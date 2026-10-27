import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import Foundation

/// Paints the fake card over the real one so it looks like it belongs in the shot.
///
/// How it hides the seams:
/// - Both cards have the same white border, so the overlay only replaces the area *inside*
///   the border and fades into it. Fingertips on the edges sit on the border and stay real.
/// - Lighting is measured from the real card's own white border every frame (a smooth
///   bilinear fit, so shadows and gradients across the card carry over), and the fake face
///   is multiplied by it.
/// - Fingers that reach past the border are found (non-white pixels near the edges) and
///   cut out of the overlay.
/// - Camera blur and sensor noise are matched so the fake isn't suspiciously crisp.
final class Compositor {
    let context: CIContext

    static let gw = 72          // analysis grid (card space), ~1.1 mm per cell
    static let gh = 100
    static let cardSize = CardFaceRenderer.size
    static let cardRect = CGRect(origin: .zero, size: CardFaceRenderer.size)

    private let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!

    private var coeffs: [[Double]]?          // 3 channels x [a, b, c, d] for a + b u + c v + d u v
    private var fingerPrev = [Float](repeating: 0, count: Compositor.gw * Compositor.gh)
    private var regionMask: CIImage?
    private var regionInset: Double = -1
    private var prevCenter: CGPoint?

    struct Analysis {
        var lighting: CIImage      // card space, predicted paper colour under the real light
        var keep: CIImage          // card space, 1 = draw overlay, 0 = finger (show camera)
        var cardness: Double       // fraction of the border that looks like white paper
        var faceUp: Bool           // interior bright enough to be a face, not a card back
        var noise: Double          // estimated sensor noise (linear units, per camera pixel)
    }

    init(context: CIContext) {
        self.context = context
    }

    func resetTemporalState() {
        coeffs = nil
        fingerPrev = [Float](repeating: 0, count: Compositor.gw * Compositor.gh)
        prevCenter = nil
    }

    // MARK: - Analysis

    func analyze(frame: CIImage, quad: Quad) -> Analysis? {
        let gw = Compositor.gw, gh = Compositor.gh

        let pc = CIFilter.perspectiveCorrection()
        pc.inputImage = frame
        pc.topLeft = quad.tl
        pc.topRight = quad.tr
        pc.bottomRight = quad.br
        pc.bottomLeft = quad.bl
        pc.crop = true
        guard let corrected = pc.outputImage else { return nil }
        let ext = corrected.extent
        guard ext.width > 8, ext.height > 8, ext.width.isFinite, ext.height.isFinite else { return nil }

        // Pre-blur so the downscale averages instead of aliasing.
        let sx = CGFloat(gw) / ext.width, sy = CGFloat(gh) / ext.height
        let sigma = max(0.0, Double(0.5 / max(sx, sy)) * 0.6)
        var src = corrected.clampedToExtent()
        if sigma > 0.3 { src = src.applyingGaussianBlur(sigma: sigma) }
        let small = src
            .transformed(by: CGAffineTransform(translationX: -ext.minX, y: -ext.minY))
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: CGRect(x: 0, y: 0, width: gw, height: gh))

        var px = [Float](repeating: 0, count: gw * gh * 4)
        px.withUnsafeMutableBytes { raw in
            context.render(small, toBitmap: raw.baseAddress!, rowBytes: gw * 16,
                           bounds: CGRect(x: 0, y: 0, width: gw, height: gh),
                           format: .RGBAf, colorSpace: linear)
        }

        @inline(__always) func rgb(_ x: Int, _ y: Int) -> (Double, Double, Double) {
            let i = (y * gw + x) * 4
            return (Double(px[i]), Double(px[i + 1]), Double(px[i + 2]))
        }
        @inline(__always) func lum(_ c: (Double, Double, Double)) -> Double {
            0.2126 * c.0 + 0.7152 * c.1 + 0.0722 * c.2
        }

        // 1. Sample the paper border (skip the rounded corners).
        struct S { var u, v, r, g, b, l: Double }
        var samples: [S] = []
        samples.reserveCapacity(400)
        let cornerX = 6, cornerY = 7
        for y in 0..<gh {
            for x in 0..<gw {
                let inRing = (x >= 1 && x <= 2) || (x >= gw - 3 && x <= gw - 2)
                    || (y >= 1 && y <= 3) || (y >= gh - 4 && y <= gh - 2)
                guard inRing else { continue }
                let nearCornerX = x < cornerX || x >= gw - cornerX
                let nearCornerY = y < cornerY || y >= gh - cornerY
                if nearCornerX && nearCornerY { continue }
                let c = rgb(x, y)
                samples.append(S(u: Double(x) / Double(gw - 1), v: Double(y) / Double(gh - 1),
                                 r: c.0, g: c.1, b: c.2, l: lum(c)))
            }
        }
        guard samples.count > 20 else { return nil }

        let sortedL = samples.map(\.l).sorted()
        let p75 = sortedL[Int(Double(sortedL.count - 1) * 0.75)]
        guard p75 > 0.002 else { return nil }
        let accepted = samples.filter { s in
            let mx = max(s.r, s.g, s.b), mn = min(s.r, s.g, s.b)
            let sat = (mx - mn) / max(mx, 1e-4)
            return s.l > 0.6 * p75 && s.l < 1.6 * p75 && sat < 0.32
        }
        let cardness = Double(accepted.count) / Double(samples.count)

        // 2. Fit a smooth bilinear lighting model per channel.
        var fresh: [[Double]] = []
        if accepted.count >= 24 {
            for ch in 0..<3 {
                fresh.append(fitBilinear(accepted.map { ($0.u, $0.v, ch == 0 ? $0.r : ch == 1 ? $0.g : $0.b) }))
            }
        } else if !accepted.isEmpty {
            let n = Double(accepted.count)
            fresh = [[accepted.map(\.r).reduce(0, +) / n, 0, 0, 0],
                     [accepted.map(\.g).reduce(0, +) / n, 0, 0, 0],
                     [accepted.map(\.b).reduce(0, +) / n, 0, 0, 0]]
        } else if let c = coeffs {
            fresh = c
        } else {
            return nil
        }
        if let prev = coeffs {
            coeffs = zip(prev, fresh).map { p, f in zip(p, f).map { 0.55 * $0 + 0.45 * $1 } }
        } else {
            coeffs = fresh
        }
        let k = coeffs!

        @inline(__always) func predict(_ u: Double, _ v: Double) -> (Double, Double, Double) {
            func e(_ c: [Double]) -> Double { max(0, c[0] + c[1] * u + c[2] * v + c[3] * u * v) }
            return (e(k[0]), e(k[1]), e(k[2]))
        }

        // Noise estimate from the border residual (scaled back up to camera-pixel size).
        var resid = 0.0
        for s in accepted {
            let p = predict(s.u, s.v)
            let d = s.l - lum(p)
            resid += d * d
        }
        resid = accepted.isEmpty ? 0 : sqrt(resid / Double(accepted.count))
        let downscale = Double(max(ext.width / CGFloat(gw), 1))
        let noise = min(0.05, resid * sqrt(downscale) * 0.5)

        // 3. Face or back? A face's interior is mostly paper; a back is mostly ink.
        var inL = 0.0, inP = 0.0
        for y in stride(from: gh / 5, to: gh * 4 / 5, by: 2) {
            for x in stride(from: gw / 5, to: gw * 4 / 5, by: 2) {
                let u = Double(x) / Double(gw - 1), v = Double(y) / Double(gh - 1)
                inL += lum(rgb(x, y))
                inP += lum(predict(u, v))
            }
        }
        let faceUp = inP > 0 && (inL / inP) > 0.38

        // 4. Fingers: skin that comes in over the card's outer edge (index corners excluded).
        // A finger has to start right at the edge of the card and is followed inward only while it
        // stays skin. Printed ink never touches the edge (there's always a white border first), so
        // the real card's pips and indices can never be mistaken for a finger and show through.
        var finger = [Float](repeating: 0, count: gw * gh)
        func isSkin(_ x: Int, _ y: Int) -> Bool {
            let c = rgb(x, y)
            let p = predict(Double(x) / Double(gw - 1), Double(y) / Double(gh - 1))
            let lc = lum(c), lp = max(lum(p), 1e-4)
            let mx = max(c.0, c.1, c.2), mn = min(c.0, c.1, c.2)
            let sat = (mx - mn) / max(mx, 1e-4)
            // printed ink: near-black and neutral, or pure saturated red
            if lc < 0.22 * lp && sat < 0.45 { return false }
            if sat > 0.88 { return false }
            if lc < 0.74 * lp { return true }
            // colour shift relative to the paper (skin reads warmer than paper)
            let dr = c.0 / max(lc, 1e-4) - p.0 / lp
            let db = c.2 / max(lc, 1e-4) - p.2 / lp
            return abs(dr) + abs(db) > 0.22
        }
        let reachLimit = 11
        let sideY0 = 27, sideY1 = gh - 28          // keep clear of the index corners
        let endX0 = 16, endX1 = gw - 17
        // Walk inward from the edge: (start x, start y, step x, step y)
        func walk(_ x0: Int, _ y0: Int, _ dx: Int, _ dy: Int) {
            // must be skin in the outermost cells
            guard isSkin(x0, y0) || isSkin(x0 + dx, y0 + dy) else { return }
            var x = x0, y = y0, n = 0
            var misses = 0
            while n <= reachLimit {
                if isSkin(x, y) { finger[y * gw + x] = 1; misses = 0 } else {
                    misses += 1
                    if misses > 1 { break }          // allow one noisy cell, then stop
                }
                x += dx; y += dy; n += 1
            }
        }
        for y in sideY0...sideY1 {
            walk(1, y, 1, 0)                       // left edge
            walk(gw - 2, y, -1, 0)                 // right edge
        }
        for x in endX0...endX1 {
            walk(x, 1, 0, 1)                       // top edge
            walk(x, gh - 2, 0, -1)                 // bottom edge
        }
        // Temporal smoothing, then a 1-cell dilation so the cut covers the whole fingertip.
        for i in finger.indices { finger[i] = 0.5 * finger[i] + 0.5 * fingerPrev[i] }
        fingerPrev = finger
        var dilated = finger
        for y in 1..<(gh - 1) {
            for x in 1..<(gw - 1) {
                var m: Float = 0
                for dy in -1...1 { for dx in -1...1 { m = max(m, finger[(y + dy) * gw + x + dx]) } }
                dilated[y * gw + x] = m
            }
        }

        // 5. Build the card-space images.
        var lightPx = [Float](repeating: 1, count: gw * gh * 4)
        var keepPx = [Float](repeating: 1, count: gw * gh * 4)
        for y in 0..<gh {
            for x in 0..<gw {
                let p = predict(Double(x) / Double(gw - 1), Double(y) / Double(gh - 1))
                let i = (y * gw + x) * 4
                lightPx[i] = Float(p.0); lightPx[i + 1] = Float(p.1); lightPx[i + 2] = Float(p.2); lightPx[i + 3] = 1
                let keep = 1 - min(1, dilated[y * gw + x] * 1.25)
                keepPx[i] = keep; keepPx[i + 1] = keep; keepPx[i + 2] = keep; keepPx[i + 3] = 1
            }
        }

        return Analysis(lighting: gridImage(lightPx, colorSpace: linear),
                        keep: gridImage(keepPx, colorSpace: nil),
                        cardness: cardness, faceUp: faceUp, noise: noise)
    }

    /// Turns a gw x gh float grid into a smooth card-sized image.
    private func gridImage(_ data: [Float], colorSpace: CGColorSpace?) -> CIImage {
        let gw = Compositor.gw, gh = Compositor.gh
        let d = data.withUnsafeBufferPointer { Data(buffer: $0) }
        let img = CIImage(bitmapData: d, bytesPerRow: gw * 16,
                          size: CGSize(width: gw, height: gh), format: .RGBAf, colorSpace: colorSpace)
        let sx = Compositor.cardSize.width / CGFloat(gw)
        let sy = Compositor.cardSize.height / CGFloat(gh)
        return img.samplingLinear()
            .clampedToExtent()
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .cropped(to: Compositor.cardRect)
    }

    /// Least-squares fit of value = a + b u + c v + d u v (with a little ridge for stability).
    private func fitBilinear(_ pts: [(Double, Double, Double)]) -> [Double] {
        var A = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 4)
        var bv = [Double](repeating: 0, count: 4)
        for (u, v, val) in pts {
            let f = [1, u, v, u * v]
            for i in 0..<4 {
                bv[i] += f[i] * val
                for j in 0..<4 { A[i][j] += f[i] * f[j] }
            }
        }
        let n = Double(pts.count)
        for i in 1..<4 { A[i][i] += 0.02 * n }
        // Gaussian elimination
        var M = A
        var r = bv
        for col in 0..<4 {
            var piv = col
            for row in col..<4 where abs(M[row][col]) > abs(M[piv][col]) { piv = row }
            M.swapAt(col, piv); r.swapAt(col, piv)
            let p = M[col][col]
            if abs(p) < 1e-12 { return [pts.map { $0.2 }.reduce(0, +) / n, 0, 0, 0] }
            for row in 0..<4 where row != col {
                let f = M[row][col] / p
                for j in col..<4 { M[row][j] -= f * M[col][j] }
                r[row] -= f * r[col]
            }
        }
        return (0..<4).map { r[$0] / M[$0][$0] }
    }

    // MARK: - Overlay

    private func region(insetScale: Double) -> CIImage {
        if let m = regionMask, abs(regionInset - insetScale) < 0.001 { return m }
        let W = Compositor.cardSize.width, H = Compositor.cardSize.height
        let ix = W * 0.032 * CGFloat(insetScale), iy = H * 0.025 * CGFloat(insetScale)
        let gen = CIFilter.roundedRectangleGenerator()
        gen.extent = Compositor.cardRect.insetBy(dx: ix, dy: iy)
        gen.radius = Float(W * 0.035)
        gen.color = CIColor.white
        let black = CIImage(color: .black).cropped(to: Compositor.cardRect)
        let m = (gen.outputImage ?? black)
            .composited(over: black)
            .clampedToExtent()
            .applyingGaussianBlur(sigma: Double(W * 0.007))
            .cropped(to: Compositor.cardRect)
        regionMask = m
        regionInset = insetScale
        return m
    }

    /// The fake face lifted slightly off pure black: printed ink never photographs as 0.
    static func prepareAlbedo(_ face: CIImage) -> CIImage {
        let f = CIFilter.colorMatrix()
        f.inputImage = face
        f.rVector = CIVector(x: 0.95, y: 0, z: 0, w: 0)
        f.gVector = CIVector(x: 0, y: 0.95, z: 0, w: 0)
        f.bVector = CIVector(x: 0, y: 0, z: 0.95, w: 0)
        f.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
        f.biasVector = CIVector(x: 0.035, y: 0.035, z: 0.035, w: 0)
        return (f.outputImage ?? face).cropped(to: Compositor.cardRect)
    }

    func overlay(frame: CIImage, quad: Quad, albedo: CIImage, analysis a: Analysis,
                 cfg: RenderConfig, opacity: Double = 1) -> CIImage {
        // Relight the fake face with the real card's measured light.
        let gain = 1 + cfg.brightness
        var lighting = a.lighting
        if abs(gain - 1) > 0.001 {
            let m = CIFilter.colorMatrix()
            m.inputImage = lighting
            m.rVector = CIVector(x: gain, y: 0, z: 0, w: 0)
            m.gVector = CIVector(x: 0, y: gain, z: 0, w: 0)
            m.bVector = CIVector(x: 0, y: 0, z: gain, w: 0)
            m.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
            lighting = m.outputImage ?? lighting
        }
        let mul = CIFilter.multiplyCompositing()
        mul.inputImage = albedo
        mul.backgroundImage = lighting
        var lit = (mul.outputImage ?? albedo).cropped(to: Compositor.cardRect)

        // Sensor noise, grain sized to roughly one camera pixel.
        let amp = a.noise * cfg.noiseScale
        if amp > 0.0005, let rnd = CIFilter.randomGenerator().outputImage {
            let s = amp * 3.464
            // Opaque noise image (alpha 1), so the colour matrix is unaffected by premultiplication.
            let nm = CIFilter.colorMatrix()
            nm.inputImage = rnd.composited(over: CIImage(color: .black))   // raw values, alpha 1
            nm.rVector = CIVector(x: s, y: 0, z: 0, w: 0)
            nm.gVector = CIVector(x: s, y: 0, z: 0, w: 0)
            nm.bVector = CIVector(x: s, y: 0, z: 0, w: 0)
            nm.aVector = CIVector(x: 0, y: 0, z: 0, w: 0)
            // After compositing over black each value is uniform x uniform (mean 0.25), so centre on that.
            nm.biasVector = CIVector(x: -s / 4, y: -s / 4, z: -s / 4, w: 1)
            let grain = max(1, Compositor.cardSize.width / max(quad.width, 1))
            if let n = nm.outputImage?.transformed(by: CGAffineTransform(scaleX: grain, y: grain)) {
                // lit + noise, keeping alpha at 1: 2 * mix(lit, noise, 0.5).
                let mix = CIFilter.dissolveTransition()
                mix.inputImage = lit
                mix.targetImage = n.cropped(to: Compositor.cardRect)
                mix.time = 0.5
                let x2 = CIFilter.colorMatrix()
                x2.inputImage = mix.outputImage
                x2.rVector = CIVector(x: 2, y: 0, z: 0, w: 0)
                x2.gVector = CIVector(x: 0, y: 2, z: 0, w: 0)
                x2.bVector = CIVector(x: 0, y: 0, z: 2, w: 0)
                x2.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
                lit = (x2.outputImage ?? lit).cropped(to: Compositor.cardRect)
            }
        }

        // Alpha = inside-the-border region x finger cut-outs (x opacity for fades).
        let mm = CIFilter.multiplyCompositing()
        mm.inputImage = region(insetScale: cfg.insetScale)
        mm.backgroundImage = a.keep
        var alpha = (mm.outputImage ?? a.keep).cropped(to: Compositor.cardRect)
        if opacity < 0.999 {
            let om = CIFilter.colorMatrix()
            om.inputImage = alpha
            let o = CGFloat(max(0, opacity))
            om.rVector = CIVector(x: o, y: 0, z: 0, w: 0)
            om.gVector = CIVector(x: 0, y: o, z: 0, w: 0)
            om.bVector = CIVector(x: 0, y: 0, z: o, w: 0)
            om.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
            alpha = om.outputImage ?? alpha
        }

        let clear = CIImage(color: .clear).cropped(to: Compositor.cardRect)
        let blend = CIFilter.blendWithMask()
        blend.inputImage = lit
        blend.backgroundImage = clear
        blend.maskImage = alpha
        var card = (blend.outputImage ?? lit).cropped(to: Compositor.cardRect)

        if cfg.debug {
            // Show finger cut-outs in red so they can be checked.
            let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0, alpha: 0.6)).cropped(to: Compositor.cardRect)
            let inv = a.keep.applyingFilter("CIColorInvert")
            let dbg = CIFilter.blendWithMask()
            dbg.inputImage = red
            dbg.backgroundImage = card
            dbg.maskImage = inv
            card = (dbg.outputImage ?? card).cropped(to: Compositor.cardRect)
        }

        // Into the camera frame.
        let pt = CIFilter.perspectiveTransform()
        pt.inputImage = card
        pt.topLeft = quad.tl
        pt.topRight = quad.tr
        pt.bottomRight = quad.br
        pt.bottomLeft = quad.bl
        guard var warped = pt.outputImage else { return frame }

        // Match the camera's softness.
        let sigma = 0.45 * cfg.blurScale * Double(max(0.7, quad.width / 500))
        if sigma > 0.15 {
            let box = quad.boundingBox.insetBy(dx: -8, dy: -8)
            warped = warped.applyingGaussianBlur(sigma: sigma).cropped(to: box)
        }
        // Match motion blur: when the card moves, the real one smears along the motion, so the
        // fake has to as well or it looks pasted on.
        let c = quad.center
        if let pc = prevCenter {
            let vx = c.x - pc.x, vy = c.y - pc.y
            let speed = hypot(vx, vy)
            if speed > 3 && speed < quad.width * 0.35 {
                let radius = min(40, 0.4 * speed * CGFloat(cfg.blurScale))
                let mb = CIFilter.motionBlur()
                mb.inputImage = warped
                mb.radius = Float(radius)
                mb.angle = Float(atan2(vy, vx))
                if let o = mb.outputImage {
                    warped = o.cropped(to: quad.boundingBox.insetBy(dx: -radius - 8, dy: -radius - 8))
                }
            }
        }
        prevCenter = c

        var out = warped.composited(over: frame)
        if cfg.debug { out = debugCorners(out, quad: quad) }
        return out.cropped(to: frame.extent)
    }

    func debugCorners(_ img: CIImage, quad: Quad, color: CIColor = CIColor(red: 0, green: 1, blue: 0)) -> CIImage {
        var out = img
        for (i, p) in quad.points.enumerated() {
            let s: CGFloat = i == 0 ? 22 : 14   // bigger dot marks the card's top-left
            let dot = CIImage(color: color).cropped(to: CGRect(x: p.x - s / 2, y: p.y - s / 2, width: s, height: s))
            out = dot.composited(over: out)
        }
        return out
    }
}
