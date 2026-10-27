import Vision
import CoreVideo
import CoreImage
import CoreGraphics

/// Finds the playing card in each frame and follows it.
///
/// - Finding: Vision's document detector (copes with fingers and busy backgrounds) plus rectangle
///   detection (exact edges when it can see them).
/// - Following: optical flow on the card's own printed texture (PlanarTracker), which keeps working
///   when the card's outline is hard to see (white card on a white ceiling) or partly covered by
///   fingers. Each frame the edges are then snapped to the real card edges, so the fake card stays
///   aligned and doesn't drift.
/// - Vision re-checks every few frames and takes over if the flow tracker has latched onto
///   something else.
/// - A One Euro filter removes the last bit of jitter.
/// All returned coordinates are Core Image pixels of the full frame (origin bottom-left).
final class CardTracker {
    enum State {
        case searching
        case locked(Quad)          // fresh measurement this frame
        case coasting(Quad, Int)   // no measurement; predicted pose, frames since last seen
    }

    private let ciContext: CIContext
    private var pool: CVPixelBufferPool?
    private var poolW = 0, poolH = 0

    private var smoother = QuadSmoother()
    private var lastQuad: Quad?
    private var lastTime: Double = -1
    private var framesSinceSeen = 0
    private var frameIndex = 0
    private var disagreeCount = 0
    private(set) var stableFrames = 0
    /// What produced the last measurement ("flow", "flow+edge", "doc+rect", "doc", "rect", "none").
    private(set) var lastSource = "none"

    // Flow tracker state (gray-image coordinates, origin top-left)
    private var grayQuad: [CGPoint]?
    private var prevPyramid: [GrayImage]?
    private var points: [CGPoint] = []
    private var rowsTopDown: Bool?
    private var weakFrames = 0
    /// The last confidently tracked card (gray coordinates), to re-find it after a fast move and to
    /// prefer shapes of about the same size (the card's distance from the camera changes slowly).
    private var lastGood: (quad: [CGPoint], size: CGFloat, frame: Int)?
    private var lastGoodPyramid: [GrayImage]?
    /// Card-likeness of the current lock (PlanarTracker.cardScore), -1 when none.
    private(set) var lastScore = -1.0
    /// Below this a shape isn't accepted as a card.
    var minScore = 0.14

    /// How many frames we keep predicting the pose after losing the card.
    let maxCoastFrames = 30
    /// Vision works on a copy of the frame scaled by this factor.
    var analysisScale: CGFloat = 0.5
    /// While the flow tracker has the card, re-check with Vision this often.
    var detectEvery = 10
    /// Use Vision's document detector alongside rectangle detection.
    var useDocumentDetector = true
    /// Use the optical-flow tracker between detections.
    var useFlowTracker = true
    /// How many times the document detector threw an error (for the test harness).
    private(set) var documentErrors = 0

    init(context: CIContext) {
        ciContext = context
    }

    func reset() {
        smoother.reset()
        lastQuad = nil
        lastTime = -1
        framesSinceSeen = 0
        stableFrames = 0
        disagreeCount = 0
        lastSource = "none"
        grayQuad = nil
        prevPyramid = nil
        points = []
        weakFrames = 0
        lastGood = nil
        lastGoodPyramid = nil
    }

    /// Drop the current lock and search again (used when the locked shape doesn't look like a card face).
    func forceRedetect() {
        grayQuad = nil
        points = []
    }

    /// `time` is the frame's presentation time in seconds.
    func process(_ fullBuffer: CVPixelBuffer, time: Double) -> State {
        frameIndex += 1
        let W = CGFloat(CVPixelBufferGetWidth(fullBuffer))
        let H = CGFloat(CVPixelBufferGetHeight(fullBuffer))
        let dt = lastTime < 0 ? 1.0 / 30 : min(0.2, max(0.001, time - lastTime))
        lastTime = time

        let buffer = analysisBuffer(fullBuffer)
        let gray = useFlowTracker ? renderGray(buffer) : nil
        // CPU view of the analysis frame's colours, for the "is this paper?" checks.
        let paper = PaperReader(buffer, topDown: rowsTopDown ?? true)
        defer { paper?.unlock() }
        let pyramid = gray?.pyramid(levels: 4)
        let gw = CGFloat(gray?.w ?? 1), gh = CGFloat(gray?.h ?? 1)
        let topDown = rowsTopDown ?? true

        func toGray(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * gw / W, y: topDown ? (H - p.y) * gh / H : p.y * gh / H)
        }
        func toCI(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * W / gw, y: topDown ? H - p.y * H / gh : p.y * H / gh)
        }
        func quadCI(_ g: [CGPoint]) -> Quad { Quad(g.map(toCI)) }

        var measured: Quad?
        var source = "none"

        // 1. Follow the card's texture with optical flow.
        var trackedScore = -1.0
        if let gq = grayQuad, let prev = prevPyramid, let cur = pyramid, let g = gray {
            if let next = flowStep(prev: prev, cur: cur, quad: gq), plausible(quadCI(next.quad), w: W, h: H) {
                trackedScore = cardLikeness(g, paper, next.quad)
                // Still looks like a card? Weak frames happen (motion blur, a thumb over an edge), so
                // a weak track is kept while detection looks for something better (step 2); only a
                // long run of clearly-not-a-card frames drops it.
                weakFrames = trackedScore < minScore ? weakFrames + 1 : 0
                if weakFrames < 10 || trackedScore >= minScore * 0.5 {
                    grayQuad = next.quad
                    points = next.points
                    measured = quadCI(next.quad)
                    source = next.snapped > 0 ? "flow+edge" : "flow"
                } else {
                    grayQuad = nil
                    trackedScore = -1
                }
            } else {
                grayQuad = nil
            }
        }

        // 2. Detect when lost, every few frames as a check, and sooner while the track looks weak.
        if measured == nil || frameIndex % detectEvery == 0 || (weakFrames > 0 && frameIndex % 2 == 0) {
            let cands = detectCandidates(buffer, w: W, h: H)
            if let g = gray {
                // Snap every candidate to the real edges, score it, keep the most card-like.
                var best: (quad: [CGPoint], how: String, score: Double)?
                func consider(_ q: [CGPoint], _ how: String) {
                    var gq = q
                    let (sq, n) = PlanarTracker.snapEdges(g, quad: gq)
                    if n > 0 { gq = sq }
                    guard plausible(quadCI(gq), w: W, h: H) else { return }
                    // Prefer the card's recent size when re-finding it or when the current lock is
                    // healthy, but never let it stop a clearly better card replacing a weak lock
                    // (the weak lock may itself have been a wrong, odd-sized shape).
                    let useSize = grayQuad == nil || trackedScore >= minScore
                    let sc = cardLikeness(g, paper, gq) * (useSize ? sizeConsistency(gq) : 1)
                    if best == nil || sc > best!.score { best = (gq, how, sc) }
                }
                if grayQuad == nil, let lg = lastGood, frameIndex - lg.frame < 90 {
                    // where the card was last seen clearly (it often comes back after a fast move)
                    consider(lg.quad, "last")
                    if let r = refitInward(g, paper, lg.quad) { consider(r, "last fit") }
                    // and wherever its picture best matches now (after a fast swipe)
                    if let tpl = lastGoodPyramid, let cur = pyramid, let moved = findByTemplate(lg.quad, from: tpl, in: cur) {
                        consider(moved, "match")
                        if let r = refitInward(g, paper, moved) { consider(r, "match fit") }
                    }
                }
                if let tq = grayQuad {
                    // the current lock might be half the card, or have grown onto the fingers
                    for bigger in PlanarTracker.extendAcrossWeakSides(g, quad: tq) { consider(bigger, "flow x2") }
                    if let r = refitInward(g, paper, tq) { consider(r, "flow fit") }
                }
                for (q, how) in cands {
                    let gq = q.points.map(toGray)
                    consider(gq, how)
                    // Vision's outline often includes the fingers behind the card: pull each side in
                    // to the real paper edge.
                    if let r = refitInward(g, paper, gq) { consider(r, how + " fit") }
                    for bigger in PlanarTracker.extendAcrossWeakSides(g, quad: gq) { consider(bigger, how + "x2") }
                }
                // Taking over from a working track needs a clearly better, clearly card-like shape.
                if let b = best, b.score >= minScore,
                   measured == nil || (b.score > trackedScore + 0.12 && b.score >= 0.3) {
                    grayQuad = b.quad
                    points = PlanarTracker.goodFeatures(g, quad: b.quad)
                    measured = quadCI(b.quad)
                    source = b.how
                    weakFrames = 0
                    trackedScore = b.score
                }
            } else if measured == nil, let first = cands.first {
                measured = first.0
                source = first.1
            }
        }
        lastScore = trackedScore
        if trackedScore >= 0.25, let gq = grayQuad {
            lastGood = (gq, quadSize(gq), frameIndex)
            lastGoodPyramid = pyramid
        }
        prevPyramid = pyramid

        // 3. Sanity: reject shapes that are clearly not a card held up to the camera.
        if let m = measured, !plausible(m, w: W, h: H) {
            measured = nil
            grayQuad = nil
        }

        if var m = measured {
            m = lastQuad.map { m.aligned(to: $0) } ?? m.uprightPortrait()
            if framesSinceSeen > maxCoastFrames || lastQuad == nil { smoother.reset() }
            let s = smoother.smooth(m, dt: dt)
            lastQuad = s
            framesSinceSeen = 0
            stableFrames += 1
            lastSource = source
            return .locked(s)
        }

        grayQuad = nil
        points = []
        lastSource = "none"
        stableFrames = 0
        framesSinceSeen += 1
        guard let last = lastQuad, framesSinceSeen <= maxCoastFrames else {
            lastQuad = nil
            return .searching
        }
        // Short constant-velocity prediction, then hold still.
        let k = CGFloat(min(framesSinceSeen, 4)) * CGFloat(dt)
        let v = smoother.velocities
        let predicted = Quad(zip(last.points, v).map { CGPoint(x: $0.x + $1.dx * k * 0.5, y: $0.y + $1.dy * k * 0.5) })
        return .coasting(predicted, framesSinceSeen)
    }

    // MARK: - Optical flow step

    private struct FlowResult {
        var quad: [CGPoint]
        var points: [CGPoint]
        var snapped: Int
    }

    private func flowStep(prev: [GrayImage], cur: [GrayImage], quad: [CGPoint]) -> FlowResult? {
        var pts = points
        if pts.count < 30 {
            let fresh = PlanarTracker.goodFeatures(prev[0], quad: quad)
            for p in fresh where !pts.contains(where: { hypot($0.x - p.x, $0.y - p.y) < 6 }) {
                pts.append(p)
                if pts.count >= 60 { break }
            }
        }
        guard pts.count >= 8 else { return nil }
        let fwd = PlanarTracker.lucasKanade(prev: prev, cur: cur, pts: pts)
        var orig: [CGPoint] = [], moved: [CGPoint] = []
        for (p, q) in zip(pts, fwd) {
            if let q = q { orig.append(p); moved.append(q) }
        }
        guard moved.count >= 8 else { return nil }
        // Forward-backward check: a point must track back to where it started.
        let back = PlanarTracker.lucasKanade(prev: cur, cur: prev, pts: moved)
        var a: [CGPoint] = [], b: [CGPoint] = []
        for i in 0..<moved.count {
            if let r = back[i], hypot(r.x - orig[i].x, r.y - orig[i].y) < 1.0 {
                a.append(orig[i]); b.append(moved[i])
            }
        }
        guard a.count >= 8, let fit = PlanarTracker.ransacSimilarity(a, b) else { return nil }
        let m = fit.0, inliers = fit.1
        var nq = quad.map { m.apply($0) }
        var snapped = 0
        let (sq, n) = PlanarTracker.snapEdges(cur[0], quad: nq)
        if n > 0 { nq = sq; snapped = n }
        let kept = zip(b, inliers).filter { $0.1 }.map { $0.0 }.filter { PlanarTracker.inside(nq, $0) }
        return FlowResult(quad: nq, points: kept, snapped: snapped)
    }

    // MARK: - Re-finding the card by its picture

    /// Where did the card's image (from the last frame it was clearly tracked) move to? Searches by
    /// normalized cross-correlation at 1/8 scale over about a card's width, refines at 1/4 scale,
    /// and returns the quad shifted by that offset, or nil if nothing matches well.
    private func findByTemplate(_ quad: [CGPoint], from tpl: [GrayImage], in cur: [GrayImage]) -> [CGPoint]? {
        guard tpl.count >= 4, cur.count >= 4 else { return nil }
        let xs = quad.map(\.x), ys = quad.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max(),
              (maxX - minX).isFinite, (maxY - minY).isFinite else { return nil }
        let cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
        let hw = (maxX - minX) * 0.35, hh = (maxY - minY) * 0.35   // central part of the card
        let size = max(maxX - minX, maxY - minY)

        func ncc(_ T: GrayImage, _ C: GrayImage, _ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int, _ dx: Int, _ dy: Int) -> Float {
            var st: Float = 0, sc: Float = 0, stt: Float = 0, scc: Float = 0, stc: Float = 0
            var n: Float = 0
            for y in stride(from: y0, through: y1, by: 1) {
                for x in stride(from: x0, through: x1, by: 1) {
                    let a = T.at(x, y), b = C.at(x + dx, y + dy)
                    st += a; sc += b; stt += a * a; scc += b * b; stc += a * b; n += 1
                }
            }
            guard n > 4 else { return -1 }
            let cov = stc - st * sc / n, vt = stt - st * st / n, vc = scc - sc * sc / n
            guard vt > 1e-6, vc > 1e-6 else { return -1 }
            return cov / (vt * vc).squareRoot()
        }
        func search(level L: Int, around: (Int, Int), radius: Int) -> (Int, Int, Float) {
            let s = CGFloat(1 << L)
            let x0 = Int(((cx - hw) / s).rounded()), x1 = Int(((cx + hw) / s).rounded())
            let y0 = Int(((cy - hh) / s).rounded()), y1 = Int(((cy + hh) / s).rounded())
            var best = (around.0, around.1, Float(-2))
            for dy in (around.1 - radius)...(around.1 + radius) {
                for dx in (around.0 - radius)...(around.0 + radius) {
                    let v = ncc(tpl[L], cur[L], x0, y0, x1, y1, dx, dy)
                    if v > best.2 { best = (dx, dy, v) }
                }
            }
            return best
        }
        let r3 = max(3, Int((size / 8).rounded()))
        let coarse = search(level: 3, around: (0, 0), radius: r3)
        guard coarse.2 > 0.5 else { return nil }
        let fine = search(level: 2, around: (coarse.0 * 2, coarse.1 * 2), radius: 2)
        guard fine.2 > 0.6 else { return nil }
        let ox = CGFloat(fine.0 * 4), oy = CGFloat(fine.1 * 4)
        return quad.map { CGPoint(x: $0.x + ox, y: $0.y + oy) }
    }

    // MARK: - Card check

    /// Reads colours straight from the BGRA analysis buffer (locked while it exists).
    struct PaperReader {
        let buffer: CVPixelBuffer
        let base: UnsafePointer<UInt8>
        let w: Int, h: Int, rb: Int
        let topDown: Bool

        init?(_ buffer: CVPixelBuffer, topDown: Bool) {
            guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
            CVPixelBufferLockBaseAddress(buffer, .readOnly)
            guard let b = CVPixelBufferGetBaseAddress(buffer) else {
                CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
                return nil
            }
            self.buffer = buffer
            base = UnsafePointer(b.assumingMemoryBound(to: UInt8.self))
            w = CVPixelBufferGetWidth(buffer); h = CVPixelBufferGetHeight(buffer)
            rb = CVPixelBufferGetBytesPerRow(buffer)
            self.topDown = topDown
        }

        func unlock() { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        /// Bright and close to neutral: white card paper (even under warm or cool light), not skin.
        /// (x, y) in gray-image coordinates.
        func isPaper(_ x: CGFloat, _ y: CGFloat) -> Bool {
            guard x.isFinite, y.isFinite, abs(x) < 1e6, abs(y) < 1e6 else { return false }
            let xi = Int(x.rounded())
            var yi = Int(y.rounded())
            if !topDown { yi = h - 1 - yi }
            guard xi >= 0, xi < w, yi >= 0, yi < h else { return false }
            let o = yi * rb + xi * 4
            let bb = Double(base[o]), gg = Double(base[o + 1]), rr = Double(base[o + 2])
            let mx = max(rr, gg, bb), mn = min(rr, gg, bb)
            return mx > 64 && (mx - mn) / mx < 0.3
        }
    }

    /// PlanarTracker.cardScore (edges, even border, shape) times how much the border looks like
    /// white paper in colour, so a finger, a patch of palm or the gap between two fingers never
    /// wins over the card.
    private func cardLikeness(_ g: GrayImage, _ paper: PaperReader?, _ quad: [CGPoint]) -> Double {
        let s = PlanarTracker.cardScore(g, quad: quad)
        guard s > 0 else { return 0 }
        guard let paper = paper else { return s }
        let p = paperFraction(paper, quad)
        return s * p * p
    }

    private func quadSize(_ q: [CGPoint]) -> CGFloat {
        (0..<4).map { k in hypot(q[(k + 1) % 4].x - q[k].x, q[(k + 1) % 4].y - q[k].y) }.max() ?? 0
    }

    /// 1 for a shape the same size as the card seen recently, falling off for much smaller or
    /// bigger shapes (a finger, the gap between two fingers, a whole hand).
    private func sizeConsistency(_ q: [CGPoint]) -> Double {
        guard let lg = lastGood, frameIndex - lg.frame < 90, lg.size > 0 else { return 1 }
        let r = Double(quadSize(q) / lg.size)
        guard r > 0, r.isFinite else { return 0 }
        return exp(-pow(log(r) / 0.4, 2))
    }

    /// Fraction of points just inside the quad's edges that look like paper.
    private func paperFraction(_ paper: PaperReader, _ quad: [CGPoint]) -> Double {
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        let size = (0..<4).map { k in hypot(quad[(k + 1) % 4].x - quad[k].x, quad[(k + 1) % 4].y - quad[k].y) }.max() ?? 0
        guard size > 8, size.isFinite else { return 0 }
        var ok = 0, total = 0
        for k in 0..<4 {
            let a = quad[k], b = quad[(k + 1) % 4]
            let L = hypot(b.x - a.x, b.y - a.y)
            guard L > 1 else { continue }
            var n = CGVector(dx: -(b.y - a.y) / L, dy: (b.x - a.x) / L)
            if n.dx * ((a.x + b.x) / 2 - cx) + n.dy * ((a.y + b.y) / 2 - cy) < 0 { n = CGVector(dx: -n.dx, dy: -n.dy) }
            for s in 0..<20 {
                let t = 0.12 + 0.76 * CGFloat(s) / 19
                let x = a.x + t * (b.x - a.x) - 0.04 * size * n.dx
                let y = a.y + t * (b.y - a.y) - 0.04 * size * n.dy
                total += 1
                if paper.isPaper(x, y) { ok += 1 }
            }
        }
        return total == 0 ? 0 : Double(ok) / Double(total)
    }

    /// Moves each side of an oversized outline inward (or slightly outward) to the line with the
    /// most evidence of a card edge: a brightness step with paper on the inner side. Fixes outlines
    /// that include the fingers or thumb around the card. Gray-image coordinates.
    private func refitInward(_ g: GrayImage, _ paper: PaperReader?, _ quad: [CGPoint]) -> [CGPoint]? {
        guard let paper = paper, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        var lines: [(CGPoint, CGVector)] = []
        var moved = false
        for k in 0..<4 {
            let a = quad[k], b = quad[(k + 1) % 4]
            let L = hypot(b.x - a.x, b.y - a.y)
            guard L > 4 else { return nil }
            let d = CGVector(dx: (b.x - a.x) / L, dy: (b.y - a.y) / L)
            var n = CGVector(dx: -d.dy, dy: d.dx)
            if n.dx * ((a.x + b.x) / 2 - cx) + n.dy * ((a.y + b.y) / 2 - cy) < 0 { n = CGVector(dx: -n.dx, dy: -n.dy) }
            let p1 = quad[(k + 1) % 4], p2 = quad[(k + 2) % 4], p3 = quad[(k + 3) % 4]
            let D = (hypot(p2.x - p1.x, p2.y - p1.y) + hypot(a.x - p3.x, a.y - p3.y)) / 2
            let step = max(1, D / 60)
            var bestOff: CGFloat = 0, bestEv = 0.0
            var o = 0.05 * D
            while o >= -0.45 * D {
                var hits = 0
                for s in 0..<20 {
                    let t = 0.12 + 0.76 * CGFloat(s) / 19
                    let px = a.x + t * (b.x - a.x) + o * n.dx, py = a.y + t * (b.y - a.y) + o * n.dy
                    var gr: Float = 0
                    for e in [-1.0, 0.0, 1.0] as [CGFloat] {
                        let qx = px + e * n.dx, qy = py + e * n.dy
                        let v = abs(g.sample(Float(qx - 2 * n.dx), Float(qy - 2 * n.dy)) - g.sample(Float(qx + 2 * n.dx), Float(qy + 2 * n.dy)))
                        gr = max(gr, v)
                    }
                    if gr > 0.02 && paper.isPaper(px - 3 * n.dx, py - 3 * n.dy) { hits += 1 }
                }
                let ev = Double(hits) / 20
                if ev > bestEv + 1e-9 { bestEv = ev; bestOff = o }
                o -= step
            }
            let off = bestEv >= 0.3 ? bestOff : 0
            if abs(off) > 1 { moved = true }
            lines.append((CGPoint(x: a.x + off * n.dx, y: a.y + off * n.dy), d))
        }
        guard moved else { return nil }
        var out: [CGPoint] = []
        for k in 0..<4 {
            let (p, r) = lines[(k + 3) % 4], (q, s) = lines[k]
            let det = r.dx * (-s.dy) - (-s.dx) * r.dy
            guard abs(det) > 1e-6 else { return nil }
            let t = ((q.x - p.x) * (-s.dy) - (-s.dx) * (q.y - p.y)) / det
            out.append(CGPoint(x: p.x + t * r.dx, y: p.y + t * r.dy))
        }
        return out
    }

    // MARK: - Gray image for the flow tracker

    private func renderGray(_ buffer: CVPixelBuffer) -> GrayImage? {
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        guard w > 16, h > 16 else { return nil }
        if rowsTopDown == nil { rowsTopDown = detectRowOrder() }
        let src = CIImage(cvPixelBuffer: buffer)
        let ext = src.extent
        let luma = src.applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.299, y: 0.587, z: 0.114, w: 0),
            "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
        ])
        .applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])
        // Perceptual (gamma) values, so edge thresholds behave the same in dim and bright light.
        .applyingFilter("CIGammaAdjust", parameters: ["inputPower": 0.4545])
        .clampedToExtent()
        .applyingGaussianBlur(sigma: 1.0)
        .cropped(to: ext)
        var px = [Float](repeating: 0, count: w * h)
        px.withUnsafeMutableBytes { raw in
            ciContext.render(luma, toBitmap: raw.baseAddress!, rowBytes: w * 4, bounds: ext, format: .Rf, colorSpace: nil)
        }
        return GrayImage(w: w, h: h, px: px)
    }

    /// Whether `render(toBitmap:)` writes the top of the image first (checked once at runtime).
    private func detectRowOrder() -> Bool {
        let top = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: CGRect(x: 0, y: 1, width: 1, height: 1))
        let bottom = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let img = top.composited(over: bottom)
        var v = [Float](repeating: 0, count: 2)
        v.withUnsafeMutableBytes { raw in
            ciContext.render(img, toBitmap: raw.baseAddress!, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 2),
                             format: .Rf, colorSpace: nil)
        }
        return v[0] > v[1]
    }

    // MARK: - Detection

    /// Card candidates from Vision, most likely first: the document outline and every rectangle.
    private func detectCandidates(_ buffer: CVPixelBuffer, w: CGFloat, h: CGFloat) -> [(Quad, String)] {
        let doc = VNDetectDocumentSegmentationRequest()
        let rect = VNDetectRectanglesRequest()
        rect.minimumAspectRatio = 0.45
        rect.maximumAspectRatio = 0.95
        rect.quadratureTolerance = 30
        rect.minimumSize = 0.08
        rect.minimumConfidence = 0.2
        rect.maximumObservations = 10
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up, options: [:])
        // Run separately so one detector failing never takes the other down with it.
        try? handler.perform([rect])
        if useDocumentDetector {
            do { try handler.perform([doc]) } catch { documentErrors += 1 }
        }
        var out: [(Quad, String)] = []
        if useDocumentDetector, let d = doc.results?.first, d.confidence > 0.3 {
            let dq = quad(from: d, w: w, h: h)
            if plausible(dq, w: w, h: h) { out.append((dq, "doc")) }
        }
        let rects = (rect.results ?? [])
            .sorted { area($0) * CGFloat($0.confidence) > area($1) * CGFloat($1.confidence) }
        for r in rects {
            let q = quad(from: r, w: w, h: h)
            if plausible(q, w: w, h: h) { out.append((q, "rect")) }
        }
        if let last = lastQuad {
            // nearest to where the card was comes first
            out.sort { $0.0.center.distance(to: last.center) < $1.0.center.distance(to: last.center) }
        }
        return out
    }

    // MARK: - Helpers

    /// A reduced-size copy of the frame for Vision (same aspect ratio, so normalized results map back).
    private func analysisBuffer(_ pb: CVPixelBuffer) -> CVPixelBuffer {
        guard analysisScale < 0.999 else { return pb }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let tw = max(64, Int((CGFloat(w) * analysisScale).rounded()) & ~1)
        let th = max(64, Int((CGFloat(h) * analysisScale).rounded()) & ~1)
        if pool == nil || tw != poolW || th != poolH {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: tw,
                kCVPixelBufferHeightKey as String: th,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
            ]
            var p: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
            pool = p
            poolW = tw; poolH = th
        }
        guard let pool = pool else { return pb }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let o = out else { return pb }
        let img = CIImage(cvPixelBuffer: pb)
            .transformed(by: CGAffineTransform(scaleX: CGFloat(tw) / CGFloat(w), y: CGFloat(th) / CGFloat(h)))
        ciContext.render(img, to: o)
        return o
    }

    private func area(_ o: VNRectangleObservation) -> CGFloat {
        o.boundingBox.width * o.boundingBox.height
    }

    private func meanCornerDistance(_ a: Quad, _ b: Quad) -> CGFloat {
        let bb = b.aligned(to: a)
        return zip(a.points, bb.points).reduce(CGFloat(0)) { $0 + $1.0.distance(to: $1.1) } / 4
    }

    private func quad(from o: VNRectangleObservation, w: CGFloat, h: CGFloat) -> Quad {
        func p(_ n: CGPoint) -> CGPoint { CGPoint(x: n.x * w, y: n.y * h) }
        return Quad(tl: p(o.topLeft), tr: p(o.topRight), br: p(o.bottomRight), bl: p(o.bottomLeft))
    }

    private func plausible(_ q: Quad, w: CGFloat, h: CGFloat) -> Bool {
        let short = min(q.width, q.height), long = max(q.width, q.height)
        // In a close-up the card is never tiny: rules out pips, fingertips and other small boxes.
        guard short > max(40, 0.13 * min(w, h)), long > 0 else { return false }
        let ratio = short / long
        // A 63 x 88 mm card is 0.716; allow for perspective.
        guard ratio > 0.45, ratio < 0.95 else { return false }
        // Convex, roughly rectangular (no corner angle far from 90 degrees), opposite sides similar.
        let p = q.points
        var sign: CGFloat = 0
        for k in 0..<4 {
            let a = p[(k + 3) % 4], b = p[k], c = p[(k + 1) % 4]
            let u = CGVector(dx: a.x - b.x, dy: a.y - b.y), v = CGVector(dx: c.x - b.x, dy: c.y - b.y)
            let lu = hypot(u.dx, u.dy), lv = hypot(v.dx, v.dy)
            guard lu > 1, lv > 1 else { return false }
            let cross = u.dx * v.dy - u.dy * v.dx
            if sign == 0 { sign = cross } else if (cross > 0) != (sign > 0) { return false }
            let cosA = (u.dx * v.dx + u.dy * v.dy) / (lu * lv)
            if abs(cosA) > 0.64 { return false }         // outside 50...130 degrees
        }
        let s0 = p[0].distance(to: p[1]), s1 = p[1].distance(to: p[2])
        let s2 = p[2].distance(to: p[3]), s3 = p[3].distance(to: p[0])
        guard s0 / s2 > 0.6, s0 / s2 < 1.67, s1 / s3 > 0.6, s1 / s3 < 1.67 else { return false }
        // Not the whole frame (e.g. a doorway or the edge of a screen).
        return q.boundingBox.width < w * 0.98 || q.boundingBox.height < h * 0.98
    }
}
