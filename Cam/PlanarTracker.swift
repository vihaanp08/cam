import Foundation
import CoreGraphics

// Frame-to-frame card tracking that doesn't depend on the card's outline being visible:
// it follows texture points printed on the card (pips, indices) with pyramidal Lucas-Kanade
// optical flow, fits a rotation + scale + translation to them (RANSAC, so fingers and other
// stray points are ignored), then snaps each card edge to the strongest nearby edge in the image.
// All coordinates here are in "gray image" pixels: origin top-left, x right, y down.

/// A single-channel float image (values roughly 0...1).
struct GrayImage {
    let w: Int
    let h: Int
    var px: [Float]

    init(w: Int, h: Int, px: [Float]) {
        self.w = w; self.h = h; self.px = px
    }

    @inline(__always) func at(_ x: Int, _ y: Int) -> Float {
        let cx = min(max(x, 0), w - 1), cy = min(max(y, 0), h - 1)
        return px[cy * w + cx]
    }

    /// Bilinear sample with edge clamping.
    @inline(__always) func sample(_ x: Float, _ y: Float) -> Float {
        guard x.isFinite, y.isFinite else { return 0 }
        let fx = min(max(x, 0), Float(w - 1)), fy = min(max(y, 0), Float(h - 1))
        let x0 = Int(fx), y0 = Int(fy)
        let x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1)
        let ax = fx - Float(x0), ay = fy - Float(y0)
        let r0 = y0 * w, r1 = y1 * w
        let top = px[r0 + x0] + (px[r0 + x1] - px[r0 + x0]) * ax
        let bot = px[r1 + x0] + (px[r1 + x1] - px[r1 + x0]) * ax
        return top + (bot - top) * ay
    }

    /// Inside the image with a 1-pixel margin.
    @inline(__always) func contains(_ x: Double, _ y: Double) -> Bool {
        x >= 1 && y >= 1 && x <= Double(w - 2) && y <= Double(h - 2)
    }

    /// Half-size image (2x2 average).
    func down() -> GrayImage {
        let nw = max(1, w / 2), nh = max(1, h / 2)
        var out = [Float](repeating: 0, count: nw * nh)
        px.withUnsafeBufferPointer { src in
            for y in 0..<nh {
                let r0 = (2 * y) * w, r1 = min(2 * y + 1, h - 1) * w
                for x in 0..<nw {
                    let x0 = 2 * x, x1 = min(2 * x + 1, w - 1)
                    out[y * nw + x] = 0.25 * (src[r0 + x0] + src[r0 + x1] + src[r1 + x0] + src[r1 + x1])
                }
            }
        }
        return GrayImage(w: nw, h: nh, px: out)
    }

    func pyramid(levels: Int) -> [GrayImage] {
        var p = [self]
        while p.count < levels, p.last!.w > 40, p.last!.h > 40 { p.append(p.last!.down()) }
        return p
    }
}

enum PlanarTracker {

    // MARK: - Feature points (Shi-Tomasi)

    /// Corners with strong texture inside the card (the quad shrunk by `shrink`), at least `minDist` apart.
    static func goodFeatures(_ g: GrayImage, quad: [CGPoint], shrink: CGFloat = 0.14, maxCount: Int = 60,
                             quality: Float = 0.01, minDist: Float = 6) -> [CGPoint] {
        guard quad.count == 4, quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return [] }
        let inner = shrunk(quad, shrink)
        let xs = inner.map(\.x), ys = inner.map(\.y)
        let x0 = max(2, Int(xs.min()!.rounded(.down))), x1 = min(g.w - 3, Int(xs.max()!.rounded(.up)))
        let y0 = max(2, Int(ys.min()!.rounded(.down))), y1 = min(g.h - 3, Int(ys.max()!.rounded(.up)))
        guard x1 - x0 > 8, y1 - y0 > 8 else { return [] }
        let bw = x1 - x0 + 1, bh = y1 - y0 + 1

        // Gradient products
        var a = [Float](repeating: 0, count: bw * bh), b = a, c = a
        for y in 0..<bh {
            for x in 0..<bw {
                let gx = x0 + x, gy = y0 + y
                let ix = 0.5 * (g.at(gx + 1, gy) - g.at(gx - 1, gy))
                let iy = 0.5 * (g.at(gx, gy + 1) - g.at(gx, gy - 1))
                let i = y * bw + x
                a[i] = ix * ix; b[i] = ix * iy; c[i] = iy * iy
            }
        }
        // 7x7 box sums via integral images
        func integral(_ v: [Float]) -> [Float] {
            var s = [Float](repeating: 0, count: (bw + 1) * (bh + 1))
            for y in 0..<bh {
                var row: Float = 0
                for x in 0..<bw {
                    row += v[y * bw + x]
                    s[(y + 1) * (bw + 1) + x + 1] = s[y * (bw + 1) + x + 1] + row
                }
            }
            return s
        }
        let sa = integral(a), sb = integral(b), sc = integral(c)
        func box(_ s: [Float], _ x: Int, _ y: Int, _ r: Int) -> Float {
            let xa = max(0, x - r), xb = min(bw, x + r + 1), ya = max(0, y - r), yb = min(bh, y + r + 1)
            return s[yb * (bw + 1) + xb] - s[ya * (bw + 1) + xb] - s[yb * (bw + 1) + xa] + s[ya * (bw + 1) + xa]
        }
        var score = [Float](repeating: 0, count: bw * bh)
        var maxScore: Float = 0
        for y in 0..<bh {
            for x in 0..<bw {
                let p = CGPoint(x: CGFloat(x0 + x), y: CGFloat(y0 + y))
                guard inside(inner, p) else { continue }
                let A = box(sa, x, y, 3), B = box(sb, x, y, 3), C = box(sc, x, y, 3)
                let e = 0.5 * (A + C) - ((0.5 * (A - C)) * (0.5 * (A - C)) + B * B).squareRoot()
                score[y * bw + x] = e
                if e > maxScore { maxScore = e }
            }
        }
        guard maxScore > 0 else { return [] }
        let thr = quality * maxScore
        // local maxima (3x3) above threshold
        var cands: [(Float, Int, Int)] = []
        for y in 1..<(bh - 1) {
            for x in 1..<(bw - 1) {
                let s = score[y * bw + x]
                guard s > thr else { continue }
                var isMax = true
                for dy in -1...1 { for dx in -1...1 where (dx != 0 || dy != 0) {
                    if score[(y + dy) * bw + x + dx] > s { isMax = false }
                } }
                if isMax { cands.append((s, x0 + x, y0 + y)) }
            }
        }
        cands.sort { $0.0 > $1.0 }
        var out: [CGPoint] = []
        let md2 = minDist * minDist
        for (_, x, y) in cands {
            let fx = Float(x), fy = Float(y)
            if out.contains(where: { let dx = Float($0.x) - fx, dy = Float($0.y) - fy; return dx * dx + dy * dy < md2 }) { continue }
            out.append(CGPoint(x: x, y: y))
            if out.count >= maxCount { break }
        }
        return out
    }

    // MARK: - Pyramidal Lucas-Kanade

    /// Tracks `pts` from `prev` to `cur`. Returns the new position, or nil if the point was lost.
    static func lucasKanade(prev: [GrayImage], cur: [GrayImage], pts: [CGPoint],
                            halfWindow: Int = 7, iterations: Int = 10) -> [CGPoint?] {
        let levels = min(prev.count, cur.count)
        let n = (2 * halfWindow + 1) * (2 * halfWindow + 1)
        var ip = [Float](repeating: 0, count: n), ix = ip, iy = ip
        var result: [CGPoint?] = []
        result.reserveCapacity(pts.count)

        for pt in pts {
            var gx: Float = 0, gy: Float = 0      // guess carried down the pyramid (in level coords)
            var ok = true
            for L in stride(from: levels - 1, through: 0, by: -1) {
                let P = prev[L], C = cur[L]
                let scale = Float(1 << L)
                let px = Float(pt.x) / scale, py = Float(pt.y) / scale
                // prev window and its gradient
                var gxx: Float = 0, gxy: Float = 0, gyy: Float = 0
                var k = 0
                for dy in -halfWindow...halfWindow {
                    for dx in -halfWindow...halfWindow {
                        let x = px + Float(dx), y = py + Float(dy)
                        let v = P.sample(x, y)
                        let ddx = 0.5 * (P.sample(x + 1, y) - P.sample(x - 1, y))
                        let ddy = 0.5 * (P.sample(x, y + 1) - P.sample(x, y - 1))
                        ip[k] = v; ix[k] = ddx; iy[k] = ddy
                        gxx += ddx * ddx; gxy += ddx * ddy; gyy += ddy * ddy
                        k += 1
                    }
                }
                let det = gxx * gyy - gxy * gxy
                let minEig = 0.5 * (gxx + gyy) - ((0.5 * (gxx - gyy)) * (0.5 * (gxx - gyy)) + gxy * gxy).squareRoot()
                if det < 1e-12 || minEig / Float(n) < 1e-6 { ok = false; break }
                var vx: Float = 0, vy: Float = 0
                for _ in 0..<iterations {
                    var bx: Float = 0, by: Float = 0
                    k = 0
                    let ox = px + gx + vx, oy = py + gy + vy
                    for dy in -halfWindow...halfWindow {
                        for dx in -halfWindow...halfWindow {
                            let it = C.sample(ox + Float(dx), oy + Float(dy)) - ip[k]
                            bx += it * ix[k]; by += it * iy[k]
                            k += 1
                        }
                    }
                    // solve G * eta = -b
                    let ex = (-gyy * bx + gxy * by) / det
                    let ey = (gxy * bx - gxx * by) / det
                    if !ex.isFinite || !ey.isFinite || abs(vx + ex) > 64 || abs(vy + ey) > 64 { ok = false; break }
                    vx += ex; vy += ey
                    if ex * ex + ey * ey < 1e-4 { break }
                }
                if !ok { break }
                if L > 0 { gx = 2 * (gx + vx); gy = 2 * (gy + vy) } else { gx += vx; gy += vy }
            }
            let nx = Float(pt.x) + gx, ny = Float(pt.y) + gy
            if !ok || !nx.isFinite || !ny.isFinite || nx < 0 || ny < 0 || nx > Float(cur[0].w - 1) || ny > Float(cur[0].h - 1) {
                result.append(nil)
            } else {
                result.append(CGPoint(x: CGFloat(nx), y: CGFloat(ny)))
            }
        }
        return result
    }

    // MARK: - Similarity transform with RANSAC

    /// z' = c * z + t with complex c (rotation + scale) and t (translation).
    struct Similarity {
        var cr: Double, ci: Double, tx: Double, ty: Double
        func apply(_ p: CGPoint) -> CGPoint {
            let x = Double(p.x), y = Double(p.y)
            return CGPoint(x: cr * x - ci * y + tx, y: ci * x + cr * y + ty)
        }
        var scale: Double { (cr * cr + ci * ci).squareRoot() }
    }

    static func fitSimilarity(_ a: [CGPoint], _ b: [CGPoint]) -> Similarity? {
        guard a.count >= 2 else { return nil }
        let n = Double(a.count)
        let ax = a.map { Double($0.x) }.reduce(0, +) / n, ay = a.map { Double($0.y) }.reduce(0, +) / n
        let bx = b.map { Double($0.x) }.reduce(0, +) / n, by = b.map { Double($0.y) }.reduce(0, +) / n
        var nr = 0.0, ni = 0.0, d = 0.0
        for (p, q) in zip(a, b) {
            let ux = Double(p.x) - ax, uy = Double(p.y) - ay
            let vx = Double(q.x) - bx, vy = Double(q.y) - by
            // (v) * conj(u)
            nr += vx * ux + vy * uy
            ni += vy * ux - vx * uy
            d += ux * ux + uy * uy
        }
        guard d > 1e-9 else { return nil }
        let cr = nr / d, ci = ni / d
        return Similarity(cr: cr, ci: ci, tx: bx - (cr * ax - ci * ay), ty: by - (ci * ax + cr * ay))
    }

    static func ransacSimilarity(_ a: [CGPoint], _ b: [CGPoint], threshold: Double = 2.0,
                                 iterations: Int = 80) -> (Similarity, [Bool])? {
        let n = a.count
        guard n >= 4 else { return nil }
        var seed: UInt64 = 0x9E3779B97F4A7C15
        func rand(_ m: Int) -> Int {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Int((seed >> 33) % UInt64(m))
        }
        var bestInliers: [Bool] = []
        var bestCount = 0
        let t2 = threshold * threshold
        for _ in 0..<iterations {
            let i = rand(n), j = rand(n)
            if i == j { continue }
            guard let m = fitSimilarity([a[i], a[j]], [b[i], b[j]]) else { continue }
            if m.scale < 0.5 || m.scale > 2 { continue }
            var inl = [Bool](repeating: false, count: n)
            var cnt = 0
            for k in 0..<n {
                let p = m.apply(a[k])
                let dx = Double(p.x - b[k].x), dy = Double(p.y - b[k].y)
                if dx * dx + dy * dy < t2 { inl[k] = true; cnt += 1 }
            }
            if cnt > bestCount { bestCount = cnt; bestInliers = inl }
        }
        guard bestCount >= max(4, n / 3) else { return nil }
        let ia = zip(a, bestInliers).filter { $0.1 }.map { $0.0 }
        let ib = zip(b, bestInliers).filter { $0.1 }.map { $0.0 }
        guard let m = fitSimilarity(ia, ib) else { return nil }
        return (m, bestInliers)
    }

    // MARK: - Edge snapping

    /// Moves each edge of `quad` onto the strongest straight edge nearby (robust to fingers covering
    /// parts of it), and rebuilds the corners from the four lines. Returns the refined quad and how
    /// many edges were snapped.
    static func snapEdges(_ g: GrayImage, quad: [CGPoint]) -> ([CGPoint], Int) {
        guard quad.count == 4, quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return (quad, 0) }
        let q = quad.map { (Double($0.x), Double($0.y)) }
        func len(_ a: (Double, Double), _ b: (Double, Double)) -> Double { hypot(b.0 - a.0, b.1 - a.1) }
        let size = max(len(q[0], q[1]), len(q[0], q[3]))
        guard size > 8, size < 1e4 else { return (quad, 0) }
        let rad = max(4.0, 0.07 * size)
        let cx = q.map(\.0).reduce(0, +) / 4, cy = q.map(\.1).reduce(0, +) / 4
        let samples = 28
        let steps = Int(2 * rad) + 1
        var seed: UInt64 = 12345
        func rand(_ m: Int) -> Int {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            return Int((seed >> 33) % UInt64(m))
        }
        var lines: [((Double, Double), (Double, Double))] = []   // (point, unit direction)
        var snapped = 0
        let cosLimit = cos(8.0 * Double.pi / 180)
        for k in 0..<4 {
            let a = q[k], b = q[(k + 1) % 4]
            let L = len(a, b)
            guard L > 1 else { return (quad, 0) }
            var d = ((b.0 - a.0) / L, (b.1 - a.1) / L)
            var nrm = (-d.1, d.0)
            let mid = ((a.0 + b.0) / 2, (a.1 + b.1) / 2)
            if nrm.0 * (mid.0 - cx) + nrm.1 * (mid.1 - cy) < 0 { nrm = (-nrm.0, -nrm.1) }
            var pts: [(Double, Double)] = []
            var strengths: [Double] = []
            for s in 0..<samples {
                let t = 0.18 + 0.64 * Double(s) / Double(samples - 1)
                let px = a.0 + t * (b.0 - a.0), py = a.1 + t * (b.1 - a.1)
                // parts of the card outside the picture (card entering or leaving) tell us nothing
                guard g.contains(px - rad * abs(nrm.0), py - rad * abs(nrm.1)),
                      g.contains(px + rad * abs(nrm.0), py + rad * abs(nrm.1)) else { continue }
                var best = 0.0, bestOff = 0.0
                for i in 0..<steps {
                    let off = -rad + 2 * rad * Double(i) / Double(steps - 1)
                    let x = px + off * nrm.0, y = py + off * nrm.1
                    let gp = g.sample(Float(x + nrm.0), Float(y + nrm.1))
                    let gm = g.sample(Float(x - nrm.0), Float(y - nrm.1))
                    let m = abs(Double(gp - gm)) * 0.5
                    if m > best { best = m; bestOff = off }
                }
                pts.append((px + bestOff * nrm.0, py + bestOff * nrm.1))
                strengths.append(best)
            }
            // keep the stronger samples
            guard strengths.count >= 8 else { lines.append((a, d)); continue }
            let sorted = strengths.sorted()
            let p30 = sorted[Int(Double(sorted.count - 1) * 0.3)]
            let minStrength = max(p30, 6.0 / 255.0)
            let cand = zip(pts, strengths).filter { $0.1 >= minStrength }.map { $0.0 }
            guard cand.count >= 8 else { lines.append((a, d)); continue }
            var bestInl: [Bool] = []
            var bestCount = 0
            for _ in 0..<40 {
                let i = rand(cand.count), j = rand(cand.count)
                if i == j { continue }
                var u = (cand[j].0 - cand[i].0, cand[j].1 - cand[i].1)
                let nu = hypot(u.0, u.1)
                if nu < 1e-3 { continue }
                u = (u.0 / nu, u.1 / nu)
                if abs(u.0 * d.0 + u.1 * d.1) < cosLimit { continue }
                let nn = (-u.1, u.0)
                var inl = [Bool](repeating: false, count: cand.count)
                var cnt = 0
                for (m, p) in cand.enumerated() {
                    let dist = abs((p.0 - cand[i].0) * nn.0 + (p.1 - cand[i].1) * nn.1)
                    if dist < 1.5 { inl[m] = true; cnt += 1 }
                }
                if cnt > bestCount { bestCount = cnt; bestInl = inl }
            }
            guard bestCount >= Int(0.45 * Double(samples)) else { lines.append((a, d)); continue }
            let inPts = zip(cand, bestInl).filter { $0.1 }.map { $0.0 }
            let mx = inPts.map(\.0).reduce(0, +) / Double(inPts.count)
            let my = inPts.map(\.1).reduce(0, +) / Double(inPts.count)
            var sxx = 0.0, sxy = 0.0, syy = 0.0
            for p in inPts { let dx = p.0 - mx, dy = p.1 - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy }
            // principal direction of the 2x2 covariance
            let theta = 0.5 * atan2(2 * sxy, sxx - syy)
            var u = (cos(theta), sin(theta))
            if u.0 * d.0 + u.1 * d.1 < 0 { u = (-u.0, -u.1) }
            d = u
            lines.append(((mx, my), d))
            snapped += 1
        }
        guard snapped > 0 else { return (quad, 0) }
        var out: [CGPoint] = []
        for k in 0..<4 {
            let l1 = lines[(k + 3) % 4], l2 = lines[k]
            // p + t r = q + s s'
            let r = l1.1, s = l2.1
            let det = r.0 * (-s.1) - (-s.0) * r.1
            if abs(det) < 1e-6 { return (quad, 0) }
            let dx = l2.0.0 - l1.0.0, dy = l2.0.1 - l1.0.1
            let t = (dx * (-s.1) - (-s.0) * dy) / det
            let p = CGPoint(x: l1.0.0 + t * r.0, y: l1.0.1 + t * r.1)
            if hypot(Double(p.x) - q[k].0, Double(p.y) - q[k].1) > rad * 1.5 { return (quad, 0) }
            out.append(p)
        }
        return (out, snapped)
    }

    // MARK: - How much does this quad look like a playing card?

    /// The card's real width/height ratio (short over long) undoing perspective, for a camera with
    /// focal length `f` (pixels) and the principal point at the image centre. A card tilted away
    /// from the camera looks squashed in the picture but still comes out near 63/88 = 0.716 here.
    static func trueAspect(_ quad: [CGPoint], f: Double, cx: Double, cy: Double) -> Double {
        // Zhang & He (whiteboard scanning): rectangle aspect from its four imaged corners.
        func v(_ p: CGPoint) -> (Double, Double, Double) { (Double(p.x) - cx, Double(p.y) - cy, 1) }
        func cross(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> (Double, Double, Double) {
            (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0)
        }
        func dot(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> Double { a.0 * b.0 + a.1 * b.1 + a.2 * b.2 }
        let m1 = v(quad[0]), m2 = v(quad[1]), m4 = v(quad[2]), m3 = v(quad[3])   // tl, tr, br, bl
        let c14 = cross(m1, m4)
        let d2 = dot(cross(m2, m4), m3), d3 = dot(cross(m3, m4), m2)
        func simple() -> Double {
            let w = (hypot(Double(quad[1].x - quad[0].x), Double(quad[1].y - quad[0].y)) + hypot(Double(quad[2].x - quad[3].x), Double(quad[2].y - quad[3].y))) / 2
            let h = (hypot(Double(quad[3].x - quad[0].x), Double(quad[3].y - quad[0].y)) + hypot(Double(quad[2].x - quad[1].x), Double(quad[2].y - quad[1].y))) / 2
            return min(w, h) / max(max(w, h), 1e-6)
        }
        guard abs(d2) > 1e-9, abs(d3) > 1e-9 else { return simple() }
        let k2 = dot(c14, m3) / d2, k3 = dot(c14, m2) / d3
        let n2 = (k2 * m2.0 - m1.0, k2 * m2.1 - m1.1, k2 * m2.2 - m1.2)
        let n3 = (k3 * m3.0 - m1.0, k3 * m3.1 - m1.1, k3 * m3.2 - m1.2)
        let f2 = f * f
        let a = n2.0 * n2.0 + n2.1 * n2.1 + f2 * n2.2 * n2.2
        let b = n3.0 * n3.0 + n3.1 * n3.1 + f2 * n3.2 * n3.2
        guard a > 0, b > 0, a.isFinite, b.isFinite else { return simple() }
        let r = (a / b).squareRoot()
        let ratio = min(r, 1 / r)
        return ratio.isFinite ? ratio : simple()
    }

    /// Focal length guess for the gray image (about 70 degrees across the short side).
    static func focalGuess(_ g: GrayImage) -> Double { 0.75 * Double(min(g.w, g.h)) }

    /// For each edge, the fraction of points along it where the image really has an edge
    /// (a brightness step across it, either way round: the background can be brighter or darker
    /// than the card). -1 for an edge that is mostly outside the picture (unknown).
    static func edgeSupport(_ g: GrayImage, quad: [CGPoint]) -> [Double] {
        guard quad.count == 4, quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return [0, 0, 0, 0] }
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        var out: [Double] = []
        for k in 0..<4 {
            let a = quad[k], b = quad[(k + 1) % 4]
            let L = Double(hypot(b.x - a.x, b.y - a.y))
            guard L > 4 else { out.append(0); continue }
            let d = (Double(b.x - a.x) / L, Double(b.y - a.y) / L)
            var n = (-d.1, d.0)
            let mx = Double(a.x + b.x) / 2 - Double(cx), my = Double(a.y + b.y) / 2 - Double(cy)
            if n.0 * mx + n.1 * my < 0 { n = (-n.0, -n.1) }      // outward
            var hits = 0, seen = 0
            for s in 0..<20 {
                let t = 0.12 + 0.76 * Double(s) / 19
                let px = Double(a.x) + t * Double(b.x - a.x), py = Double(a.y) + t * Double(b.y - a.y)
                guard g.contains(px - 4 * abs(n.0), py - 4 * abs(n.1)), g.contains(px + 4 * abs(n.0), py + 4 * abs(n.1)) else { continue }
                seen += 1
                var best: Float = 0
                for o in [-1.5, -0.75, 0, 0.75, 1.5] {
                    let qx = px + o * n.0, qy = py + o * n.1
                    let inside = g.sample(Float(qx - 2 * n.0), Float(qy - 2 * n.1))
                    let outside = g.sample(Float(qx + 2 * n.0), Float(qy + 2 * n.1))
                    best = max(best, abs(inside - outside))
                }
                if best > 0.012 { hits += 1 }
            }
            out.append(seen < 6 ? -1 : Double(hits) / Double(seen))
        }
        return out
    }

    /// 0...1. High when the edges are real edges, the band just inside the edges is even
    /// (the white border), and the shape is card-like once perspective is undone. Parts of the card
    /// outside the picture are ignored, so a card that is only partly in view still scores well.
    /// Used to choose between detections and to notice when the tracker has latched onto something
    /// else (a card-plus-fingers blob, half a card, a patch of palm, a window).
    static func cardScore(_ g: GrayImage, quad: [CGPoint]) -> Double {
        guard quad.count == 4, quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return 0 }
        let sides = (0..<4).map { k -> Double in
            let a = quad[k], b = quad[(k + 1) % 4]
            return Double(hypot(b.x - a.x, b.y - a.y))
        }
        let size = sides.max()!
        guard size > 16, sides.min()! > 8 else { return 0 }
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        var ring: [Float] = []
        let inset = 0.04 * size
        for k in 0..<4 {
            let a = quad[k], b = quad[(k + 1) % 4]
            let L = sides[k]
            let d = (Double(b.x - a.x) / L, Double(b.y - a.y) / L)
            var n = (-d.1, d.0)
            let mx = Double(a.x + b.x) / 2 - Double(cx), my = Double(a.y + b.y) / 2 - Double(cy)
            if n.0 * mx + n.1 * my < 0 { n = (-n.0, -n.1) }
            for s in 0..<20 {
                let t = 0.12 + 0.76 * Double(s) / 19
                let px = Double(a.x) + t * Double(b.x - a.x) - inset * n.0, py = Double(a.y) + t * Double(b.y - a.y) - inset * n.1
                guard g.contains(px, py) else { continue }
                ring.append(g.sample(Float(px), Float(py)))
            }
        }
        guard ring.count >= 20 else { return 0 }
        let support = edgeSupport(g, quad: quad).filter { $0 >= 0 }
        guard support.count >= 2 else { return 0 }
        // Every visible side of a real card is an edge somewhere along it, even with fingers on it;
        // a side with almost no edge (half a card, a box drawn through the middle) is penalised.
        let edgeScore = support.reduce(0, +) / Double(support.count) * min(1, support.min()! / 0.5)
        let sorted = ring.sorted()
        let med = sorted[sorted.count / 2]
        let even = Double(ring.filter { abs($0 - med) < 0.06 }.count) / Double(ring.count)
        let r = trueAspect(quad, f: focalGuess(g), cx: Double(g.w) / 2, cy: Double(g.h) / 2)
        let aspect = exp(-pow((r - 0.716) / 0.15, 2))
        return edgeScore * even * aspect
    }

    /// A card coming into (or going out of) view is cut off by the edge of the picture. If one side
    /// of `quad` lies along the picture's border, returns the full card: the cut-off sides extended
    /// to a playing card's proportions (either way round, the scorer picks). Gray coordinates.
    static func completeTruncated(_ quad: [CGPoint], w: Int, h: Int) -> [[CGPoint]] {
        guard quad.count == 4 else { return [] }
        let tol: CGFloat = 4
        func onBorder(_ p: CGPoint) -> Int? {
            if p.x <= tol { return 0 }
            if p.x >= CGFloat(w - 1) - tol { return 1 }
            if p.y <= tol { return 2 }
            if p.y >= CGFloat(h - 1) - tol { return 3 }
            return nil
        }
        var out: [[CGPoint]] = []
        for k in 0..<4 {
            let p0 = quad[k], p1 = quad[(k + 1) % 4]
            guard let b0 = onBorder(p0), let b1 = onBorder(p1), b0 == b1 else { continue }
            let A = quad[(k + 3) % 4], B = quad[(k + 2) % 4]    // the far side, fully visible
            let L = hypot(B.x - A.x, B.y - A.y)
            let la = hypot(p0.x - A.x, p0.y - A.y), lb = hypot(p1.x - B.x, p1.y - B.y)
            guard L > 8, la > 2, lb > 2 else { continue }
            let da = CGVector(dx: (p0.x - A.x) / la, dy: (p0.y - A.y) / la)
            let db = CGVector(dx: (p1.x - B.x) / lb, dy: (p1.y - B.y) / lb)
            let visible = max(la, lb)
            for full in [L / 0.716, L * 0.716] where full > visible * 1.02 {
                var q = quad
                q[k] = CGPoint(x: A.x + da.dx * full, y: A.y + da.dy * full)
                q[(k + 1) % 4] = CGPoint(x: B.x + db.dx * full, y: B.y + db.dy * full)
                out.append(q)
            }
        }
        return out
    }

    /// Bigger versions of `quad` for each side with no real edge under it: the shape doubled
    /// across that side (the detector sometimes finds half the card, split by the middle pip).
    static func extendAcrossWeakSides(_ g: GrayImage, quad: [CGPoint]) -> [[CGPoint]] {
        let support = edgeSupport(g, quad: quad)
        var out: [[CGPoint]] = []
        for k in 0..<4 where support[k] >= 0 && support[k] < 0.5 {
            var q = quad
            let a = k, b = (k + 1) % 4, oa = (k + 3) % 4, ob = (k + 2) % 4
            q[a] = CGPoint(x: 2 * quad[a].x - quad[oa].x, y: 2 * quad[a].y - quad[oa].y)
            q[b] = CGPoint(x: 2 * quad[b].x - quad[ob].x, y: 2 * quad[b].y - quad[ob].y)
            out.append(q)
        }
        return out
    }

    // MARK: - Geometry helpers

    static func shrunk(_ q: [CGPoint], _ f: CGFloat) -> [CGPoint] {
        let cx = q.map(\.x).reduce(0, +) / 4, cy = q.map(\.y).reduce(0, +) / 4
        return q.map { CGPoint(x: cx + ($0.x - cx) * (1 - f), y: cy + ($0.y - cy) * (1 - f)) }
    }

    /// Point inside a convex quad (either winding).
    static func inside(_ q: [CGPoint], _ p: CGPoint) -> Bool {
        var sign: CGFloat = 0
        for k in 0..<4 {
            let a = q[k], b = q[(k + 1) % 4]
            let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            if cross != 0 {
                if sign == 0 { sign = cross } else if (cross > 0) != (sign > 0) { return false }
            }
        }
        return true
    }
}
