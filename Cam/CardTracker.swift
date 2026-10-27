import Vision
import CoreVideo
import CoreGraphics
import QuartzCore

/// Finds the playing card in each frame and follows it.
/// Detection (re)acquires the card; Vision's rectangle tracker follows it between detections;
/// a One Euro filter removes jitter. All coordinates are Core Image pixels (origin bottom-left).
final class CardTracker {
    enum State {
        case searching
        case locked(Quad)          // fresh measurement this frame
        case coasting(Quad, Int)   // no measurement; predicted pose, frames since last seen
    }

    private var sequenceHandler = VNSequenceRequestHandler()
    private var trackRequest: VNTrackRectangleRequest?
    private var smoother = QuadSmoother()
    private var lastQuad: Quad?
    private var lastTime: CFTimeInterval = 0
    private var framesSinceSeen = 0
    private var frameIndex = 0
    private(set) var stableFrames = 0

    /// How many frames we keep predicting the pose after losing the card.
    let maxCoastFrames = 30

    /// Tells Vision this tracking sequence is over so it frees the tracker.
    private func endTracking(_ pixelBuffer: CVPixelBuffer) {
        if let req = trackRequest {
            req.isLastFrame = true
            try? sequenceHandler.perform([req], on: pixelBuffer, orientation: .up)
        }
        trackRequest = nil
    }

    func reset() {
        trackRequest = nil
        sequenceHandler = VNSequenceRequestHandler()   // drops any trackers Vision is holding
        smoother.reset()
        lastQuad = nil
        framesSinceSeen = 0
        stableFrames = 0
    }

    func process(_ pixelBuffer: CVPixelBuffer) -> State {
        frameIndex += 1
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let now = CACurrentMediaTime()
        let dt = lastTime == 0 ? 1.0 / 30 : min(0.2, max(0.001, now - lastTime))
        lastTime = now

        var measured: Quad?

        // 1. Follow the card with the tracker if we have one.
        if let req = trackRequest {
            do {
                try sequenceHandler.perform([req], on: pixelBuffer, orientation: .up)
                if let obs = req.results?.first as? VNRectangleObservation, obs.confidence > 0.35 {
                    req.inputObservation = obs
                    measured = quad(from: obs, w: w, h: h)
                } else {
                    endTracking(pixelBuffer)
                }
            } catch {
                endTracking(pixelBuffer)
            }
        }

        // 2. Detect when we have nothing, and every few frames to correct tracker drift.
        if measured == nil || frameIndex % 8 == 0 {
            if let obs = detect(pixelBuffer) {
                let q = quad(from: obs, w: w, h: h)
                let agrees = measured.map { $0.center.distance(to: q.center) < max(q.width, q.height) * 0.5 } ?? true
                if agrees {
                    measured = q
                    if let existing = trackRequest {
                        // Re-seed the running tracker instead of starting another one
                        // (Vision limits how many trackers can be alive at once).
                        existing.inputObservation = obs
                    } else {
                        let tr = VNTrackRectangleRequest(rectangleObservation: obs)
                        tr.trackingLevel = .accurate
                        trackRequest = tr
                    }
                }
            }
        }

        // 3. Sanity: reject shapes that are clearly not a card held up to the camera.
        if let m = measured, !plausible(m, w: w, h: h) {
            measured = nil
            endTracking(pixelBuffer)
        }

        if var m = measured {
            m = lastQuad.map { m.aligned(to: $0) } ?? m.uprightPortrait()
            if framesSinceSeen > maxCoastFrames { smoother.reset() }
            let s = smoother.smooth(m, dt: dt)
            lastQuad = s
            framesSinceSeen = 0
            stableFrames += 1
            return .locked(s)
        }

        stableFrames = 0
        framesSinceSeen += 1
        guard let last = lastQuad, framesSinceSeen <= maxCoastFrames else {
            lastQuad = framesSinceSeen > maxCoastFrames ? nil : lastQuad
            return .searching
        }
        // Constant-velocity prediction for a few frames, then hold still.
        let k = CGFloat(min(framesSinceSeen, 4)) * CGFloat(dt)
        let v = smoother.velocities
        let predicted = Quad(zip(last.points, v).map { CGPoint(x: $0.x + $1.dx * k * 0.5, y: $0.y + $1.dy * k * 0.5) })
        return .coasting(predicted, framesSinceSeen)
    }

    private func detect(_ pixelBuffer: CVPixelBuffer) -> VNRectangleObservation? {
        let req = VNDetectRectanglesRequest()
        req.minimumAspectRatio = 0.45
        req.maximumAspectRatio = 0.95
        req.quadratureTolerance = 28
        req.minimumSize = 0.12
        req.minimumConfidence = 0.55
        req.maximumObservations = 4
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do { try handler.perform([req]) } catch { return nil }
        guard let results = req.results, !results.isEmpty else { return nil }

        // Prefer the candidate nearest the last known card, otherwise the biggest confident one.
        if let last = lastQuad {
            let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            return results.min { a, b in
                quad(from: a, w: w, h: h).center.distance(to: last.center) <
                quad(from: b, w: w, h: h).center.distance(to: last.center)
            }
        }
        return results.max { area($0) * CGFloat($0.confidence) < area($1) * CGFloat($1.confidence) }
    }

    private func area(_ o: VNRectangleObservation) -> CGFloat {
        o.boundingBox.width * o.boundingBox.height
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
