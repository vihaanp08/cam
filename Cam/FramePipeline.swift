import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo
import CoreGraphics

/// The per-frame trick logic, kept free of camera and UI code so it can also run in the test harness:
/// find the card, decide whether the fake face should be on it, and paint it.
///
/// Flicker control: once the fake card is showing, it stays on through short tracking dropouts and
/// through frames where the card is partly covered. It only turns off when the card has really gone
/// (lost for `holdFrames` frames), when the card is turned around, or when the trick is revealed.
final class FramePipeline {
    let tracker: CardTracker
    let compositor: Compositor

    struct Status {
        var tracking: String = "searching"   // locked / coast N / searching
        var source: String = "none"          // which detector produced the card
        var reason: String = ""              // why the fake is or isn't showing
        var cardness: Double = 0
        var overlay: Bool = false
    }

    struct Output {
        var image: CIImage
        var drewOverlay: Bool
        var quad: Quad?
        var status: Status
    }

    /// Frames to keep the fake card on after the tracker loses the card (about 0.4 s at 30 fps).
    var holdFrames = 5
    /// Consecutive "not a card face" frames before the fake turns off (card turned around, etc.).
    var badFramesToDrop = 5

    private(set) var overlayOn = false
    private var badStreak = 0
    private var lastAnalysis: Compositor.Analysis?
    /// Time of the last frame where the card was seen, or -1.
    private(set) var lastCardSeen: Double = -1

    init(context: CIContext) {
        tracker = CardTracker(context: context)
        compositor = Compositor(context: context)
    }

    func reset() {
        tracker.reset()
        compositor.resetTemporalState()
        overlayOn = false
        badStreak = 0
        lastAnalysis = nil
    }

    /// Turns a card face image into the albedo the compositor paints with (and its mirror image).
    static func makeAlbedo(_ cg: CGImage) -> (normal: CIImage, mirrored: CIImage) {
        var face = CIImage(cgImage: cg)
        let sx = Compositor.cardSize.width / face.extent.width
        let sy = Compositor.cardSize.height / face.extent.height
        face = face.transformed(by: CGAffineTransform(scaleX: sx, y: sy)).cropped(to: Compositor.cardRect)
        let albedo = Compositor.prepareAlbedo(face)
        let mirrored = albedo
            .transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -Compositor.cardSize.width, y: 0))
            .cropped(to: Compositor.cardRect)
        return (albedo, mirrored)
    }

    func process(_ pb: CVPixelBuffer, time: Double, cfg: RenderConfig, armed: Bool, albedo: CIImage?) -> Output {
        let frame = CIImage(cvPixelBuffer: pb)
        let state = tracker.process(pb, time: time)
        var out = frame
        var drew = false
        var quad: Quad?
        var st = Status()
        st.source = tracker.lastSource
        if !armed { overlayOn = false; badStreak = 0 }

        switch state {
        case .locked(let q):
            st.tracking = "locked"
            quad = q
            lastCardSeen = time
            guard armed, let albedo = albedo else {
                st.reason = armed ? "no face image" : "revealed"
                if cfg.debug { out = compositor.debugCorners(frame, quad: q, color: CIColor(red: 0.3, green: 0.6, blue: 1)) }
                break
            }
            let a = compositor.analyze(frame: frame, quad: q)
            st.cardness = a?.cardness ?? 0
            let isFace = a.map { $0.faceUp || !cfg.facesOnly } ?? false
            // Easier to stay on than to turn on (hysteresis).
            // Turn on as soon as it plausibly is the card (the real card must never show).
            let threshold = overlayOn ? 0.22 : 0.3
            let good = a != nil && st.cardness > threshold && isFace
            if good {
                badStreak = 0
                lastAnalysis = a
                overlayOn = true
            } else {
                badStreak += 1
                if badStreak >= badFramesToDrop { overlayOn = false }
                // Locked onto something that isn't a card face for a while: search again.
                if isFace && badStreak % 10 == 0 { tracker.forceRedetect() }
            }
            if a == nil { st.reason = "analysis failed" }
            else if !isFace { st.reason = "card back" }
            else if !good { st.reason = String(format: "low paper %.2f", st.cardness) }
            else { st.reason = "ok" }

            if overlayOn, let use = good ? a : lastAnalysis {
                out = compositor.overlay(frame: frame, quad: q, albedo: albedo, analysis: use, cfg: cfg)
                drew = true
            } else if cfg.debug {
                out = compositor.debugCorners(frame, quad: q, color: CIColor(red: 1, green: 0.2, blue: 0.2))
            }

        case .coasting(let q, let n):
            st.tracking = "coast \(n)"
            quad = q
            // Bridge a short tracking hiccup (fast blur), but only while there is still clearly a card
            // there: a fake card painted on an empty wall would give the trick away.
            // (not when the card was going out of the picture: it has left, not glitched)
            let fe = frame.extent.insetBy(dx: frame.extent.width * 0.02, dy: frame.extent.height * 0.02)
            let wellInside = q.points.allSatisfy { fe.contains($0) }
            let stillThere = n <= holdFrames && wellInside && (compositor.analyze(frame: frame, quad: q)?.cardness ?? 0) > 0.5
            if armed, overlayOn, stillThere, let a = lastAnalysis, let albedo = albedo {
                out = compositor.overlay(frame: frame, quad: q, albedo: albedo, analysis: a, cfg: cfg)
                drew = true
                st.reason = "holding"
            } else {
                overlayOn = false
                if n > holdFrames { overlayOn = false }
                st.reason = "lost"
            }

        case .searching:
            st.tracking = "searching"
            st.reason = "no card"
            overlayOn = false
        }

        st.overlay = drew
        return Output(image: out.cropped(to: frame.extent), drewOverlay: drew, quad: quad, status: st)
    }
}
