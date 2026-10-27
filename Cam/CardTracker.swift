import Vision
import CoreVideo
import CoreImage
import CoreGraphics

/// Finds the playing card in each frame and follows it.
///
/// - Detection uses two Vision detectors together: document segmentation (a neural net that copes
///   well with fingers over the card and busy backgrounds) finds the card, and rectangle detection
///   snaps to its exact edges when it can see them.
/// - Between detections, Vision's rectangle tracker follows the card.
/// - A One Euro filter removes jitter without adding much lag.
/// - Vision runs on a half-size copy of the frame; results are scaled back up.
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

    private var sequenceHandler = VNSequenceRequestHandler()
    private var trackRequest: VNTrackRectangleRequest?
    private var smoother = QuadSmoother()
    private var lastQuad: Quad?
    private var lastTime: Double = -1
    private var framesSinceSeen = 0
    private var frameIndex = 0
    private var disagreeCount = 0
    private(set) var stableFrames = 0
    /// What produced the last measurement: "track", "doc+rect", "doc", "rect" or "none". For the debug view.
    private(set) var lastSource = "none"

    /// How many frames we keep predicting the pose after losing the card.
    let maxCoastFrames = 30
    /// Vision works on a copy of the frame scaled by this factor.
    var analysisScale: CGFloat = 0.5
    /// Re-run detection this often while the tracker is following the card.
    var detectEvery = 3
    /// Use Vision's document detector alongside rectangle detection.
    var useDocumentDetector = true
    /// How many times the document detector threw an error (for the test harness).
    private(set) var documentErrors = 0

    init(context: CIContext) {
        ciContext = context
    }

    func reset() {
        trackRequest = nil
        sequenceHandler = VNSequenceRequestHandler()   // drops any trackers Vision is holding
        smoother.reset()
        lastQuad = nil
        lastTime = -1
        framesSinceSeen = 0
        stableFrames = 0
        disagreeCount = 0
        lastSource = "none"
    }

    /// `time` is the frame's presentation time in seconds.
    func process(_ fullBuffer: CVPixelBuffer, time: Double) -> State {
        frameIndex += 1
        let w = CGFloat(CVPixelBufferGetWidth(fullBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(fullBuffer))
        let dt = lastTime < 0 ? 1.0 / 30 : min(0.2, max(0.001, time - lastTime))
        lastTime = time

        let buffer = analysisBuffer(fullBuffer)
        var measured: Quad?
        var source = "none"

        // 1. Follow the card with the tracker if we have one.
        if let req = trackRequest {
            do {
                try sequenceHandler.perform([req], on: buffer, orientation: .up)
                if let obs = req.results?.first as? VNRectangleObservation, obs.confidence > 0.2 {
                    req.inputObservation = obs
                    measured = quad(from: obs, w: w, h: h)
                    source = "track"
                } else {
                    endTracking(buffer)
                }
            } catch {
                endTracking(buffer)
            }
        }

        // 2. Detect every frame while searching, and every few frames to correct tracker drift.
        if measured == nil || frameIndex % detectEvery == 0 {
            if let det = detect(buffer, w: w, h: h) {
                let (obs, how) = det
                let q = quad(from: obs, w: w, h: h)
                if let m = measured {
                    let size = max(q.width, q.height)
                    let drift = meanCornerDistance(m, q)
                    if drift > size * 0.35 {
                        // Detector and tracker disagree badly. Trust the detector if it happens twice.
                        disagreeCount += 1
                        if disagreeCount >= 2 {
                            reseedTracker(with: obs, buffer: buffer)
                            measured = q
                            source = how
                            disagreeCount = 0
                        }
                    } else {
                        disagreeCount = 0
                        if how.contains("rect") {
                            // Edge-snapped detection is more accurate than the tracker: use it.
                            measured = q
                            source = how
                        }
                        if drift > size * 0.04 { reseedTracker(with: obs, buffer: buffer) }
                    }
                } else {
                    measured = q
                    source = how
                    reseedTracker(with: obs, buffer: buffer)
                }
            }
        }

        // 3. Sanity: reject shapes that are clearly not a card held up to the camera.
        if let m = measured, !plausible(m, w: w, h: h) {
            measured = nil
            endTracking(buffer)
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

    // MARK: - Tracker management

    private func reseedTracker(with obs: VNRectangleObservation, buffer: CVPixelBuffer) {
        endTracking(buffer)
        let tr = VNTrackRectangleRequest(rectangleObservation: obs)
        tr.trackingLevel = .accurate
        trackRequest = tr
    }

    /// Tells Vision this tracking sequence is over so it frees the tracker.
    private func endTracking(_ buffer: CVPixelBuffer) {
        if let req = trackRequest {
            req.isLastFrame = true
            try? sequenceHandler.perform([req], on: buffer, orientation: .up)
        }
        trackRequest = nil
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
