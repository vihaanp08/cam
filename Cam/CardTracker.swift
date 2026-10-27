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
        if let gq = grayQuad, let prev = prevPyramid, let cur = pyramid {
            if let next = flowStep(prev: prev, cur: cur, quad: gq) {
                let q = quadCI(next.quad)
                if plausible(q, w: W, h: H) {
                    grayQuad = next.quad
                    points = next.points
                    measured = q
                    source = next.snapped > 0 ? "flow+edge" : "flow"
                } else {
                    grayQuad = nil
                }
            } else {
                grayQuad = nil
            }
        }

        // 2. Detect when lost, and every few frames to catch a wrong lock.
        if measured == nil || frameIndex % detectEvery == 0 {
            if let det = detect(buffer, w: W, h: H) {
                let (obs, how) = det
                let q = quad(from: obs, w: W, h: H)
                var adopt = false
                if let m = measured {
                    let size = max(q.width, q.height)
                    if meanCornerDistance(m, q) > size * 0.3 {
                        disagreeCount += 1
                        adopt = disagreeCount >= 2
                    } else {
                        disagreeCount = 0
                    }
                } else {
                    adopt = true
                }
                if adopt {
                    disagreeCount = 0
                    measured = q
                    source = how
                    if let g = gray {
                        var gq = q.points.map(toGray)
                        let (snappedQuad, n) = PlanarTracker.snapEdges(g, quad: gq)
                        if n > 0 { gq = snappedQuad; measured = quadCI(gq) }
                        grayQuad = gq
                        points = PlanarTracker.goodFeatures(g, quad: gq)
                    }
                }
            }
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

    private func detect(_ buffer: CVPixelBuffer, w: CGFloat, h: CGFloat) -> (VNRectangleObservation, String)? {
        let doc = VNDetectDocumentSegmentationRequest()
        let rect = VNDetectRectanglesRequest()
        rect.minimumAspectRatio = 0.45
        rect.maximumAspectRatio = 0.95
        rect.quadratureTolerance = 30
        rect.minimumSize = 0.1
        rect.minimumConfidence = 0.4
        rect.maximumObservations = 6
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up, options: [:])
        // Run separately so one detector failing never takes the other down with it.
        try? handler.perform([rect])
        if useDocumentDetector {
            do { try handler.perform([doc]) } catch { documentErrors += 1 }
        }

        let rects = (rect.results ?? []).filter { plausible(quad(from: $0, w: w, h: h), w: w, h: h) }

        if useDocumentDetector, let d = doc.results?.first, d.confidence > 0.3 {
            let dq = quad(from: d, w: w, h: h)
            if plausible(dq, w: w, h: h) {
                // Prefer a precise rectangle that matches the document outline.
                let dSize = max(dq.width, dq.height)
                let dArea = dq.width * dq.height
                let matches = rects.filter { r in
                    let rq = quad(from: r, w: w, h: h)
                    let ratio = (rq.width * rq.height) / max(dArea, 1)
                    return rq.center.distance(to: dq.center) < dSize * 0.2 && ratio > 0.7 && ratio < 1.4
                }
                if let best = matches.max(by: { $0.confidence < $1.confidence }) {
                    return (best, "doc+rect")
                }
                return (d, "doc")
            }
        }

        guard !rects.isEmpty else { return nil }
        // No document found: the rectangle nearest the last card, otherwise the biggest confident one.
        if let last = lastQuad {
            let best = rects.min { a, b in
                quad(from: a, w: w, h: h).center.distance(to: last.center) <
                quad(from: b, w: w, h: h).center.distance(to: last.center)
            }!
            return (best, "rect")
        }
        let best = rects.max { area($0) * CGFloat($0.confidence) < area($1) * CGFloat($1.confidence) }!
        return (best, "rect")
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
        guard short > 40, long > 0 else { return false }
        let ratio = short / long
        // A 63 x 88 mm card is 0.716; allow for perspective.
        guard ratio > 0.45, ratio < 0.95 else { return false }
        // Not the whole frame (e.g. a doorway or the edge of a screen).
        return q.boundingBox.width < w * 0.98 || q.boundingBox.height < h * 0.98
    }
}
