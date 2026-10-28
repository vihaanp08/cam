import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreVideo
import CoreGraphics

/// The per-frame trick logic, kept free of camera and UI code so it can also run in the test harness:
/// find the card, decide whether the fake face should be on it, and paint it.
///
/// Head start: the picture on screen runs `cfg.headStartFrames` frames behind the camera. Every
/// frame is analysed the moment it arrives, but painted a few frames later, so by the time a frame
/// is shown the tracker has already seen the frames after it. When the card is first found, the
/// frames still waiting to be shown are filled in backwards (following the card's printing back in
/// time), so the fake is on the card from the very first frame it appears in, not a few frames later.
///
/// Flicker control: once the fake card is showing, it stays on through short tracking dropouts and
/// through frames where the card is partly covered or tilted. It only turns off when the card has
/// really gone (lost for `holdFrames` frames), when the card is turned around, or when the trick is
/// revealed.
final class FramePipeline {
    let tracker: CardTracker
    let compositor: Compositor

    struct Status {
        var tracking: String = "searching"   // locked / coast N / searching
        var source: String = "none"          // which detector produced the card
        var reason: String = ""              // why the fake is or isn't showing
        var cardness: Double = 0
        var overlay: Bool = false
        var score: Double = -1               // tracker's card-likeness for this frame
        var debug: String = ""               // what detection saw on this frame
        var raw: Quad?                       // the tracker's unsmoothed outline (test harness)
    }

    struct Output {
        var image: CIImage
        var drewOverlay: Bool
        var quad: Quad?
        var status: Status
        /// Which input frame this picture is (0 = the first frame since the last reset).
        var frameIndex: Int
        /// That frame's time.
        var time: Double
    }

    /// Frames to keep the fake card on after the tracker loses the card (about 0.4 s at 30 fps).
    var holdFrames = 5
    /// Consecutive "not a card face" frames before the fake turns off (card turned around, etc.).
    var badFramesToDrop = 8

    private(set) var overlayOn = false
    private var badStreak = 0
    private var lastAnalysis: Compositor.Analysis?
    /// Time of the last frame where the card was seen, or -1.
    private(set) var lastCardSeen: Double = -1

    /// A frame waiting to be shown, with what was decided about it.
    private struct Slot {
        let index: Int
        let time: Double
        let image: CIImage
        let view: CardTracker.FrameView?
        var quad: Quad?
        var draw: Bool
        var analysis: Compositor.Analysis?
        var status: Status
        /// How much the picture changed since the frame before (0 = not at all). Frames are only
        /// skipped or repeated, to change the head start, while the picture is nearly still, so it
        /// never shows as a jump or a stutter.
        var motion: Double = 1
    }
    private var queue: [Slot] = []
    private var inputCount = 0
    /// The previous frame's decision, and how many frames in a row the card has been followed
    /// on its own printing after the tracker let go of it.
    private var lastDecided: Slot?
    private var followStreak = 0
    /// Frames in a row with the fake on the card, and a frame counter for pacing the head start.
    private var drawnStreak = 0
    private var missStreak = 0
    private var live = false
    private var tick = 0
    private var ticksSinceAdjust = 0
    private var pool: CVPixelBufferPool?
    private var poolSize = CGSize.zero

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
        queue = []
        inputCount = 0
        lastDecided = nil
        followStreak = 0
        drawnStreak = 0
        missStreak = 0
        live = false
        ticksSinceAdjust = 0
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

    /// How much of the quad lies inside the picture (0...1), from a grid of points across it.
    static func fractionInside(_ q: Quad, _ r: CGRect) -> Double {
        var inside = 0
        for j in 0..<8 {
            for i in 0..<8 {
                let u = (CGFloat(i) + 0.5) / 8, v = (CGFloat(j) + 0.5) / 8
                let top = CGPoint(x: q.tl.x + u * (q.tr.x - q.tl.x), y: q.tl.y + u * (q.tr.y - q.tl.y))
                let bot = CGPoint(x: q.bl.x + u * (q.br.x - q.bl.x), y: q.bl.y + u * (q.br.y - q.bl.y))
                let p = CGPoint(x: top.x + v * (bot.x - top.x), y: top.y + v * (bot.y - top.y))
                if r.contains(p) { inside += 1 }
            }
        }
        return Double(inside) / 64
    }

    /// Analyse frame `pb` and return the frame to show now (`cfg.headStartFrames` frames older).
    /// `armed` decides only whether the fake is painted on the frame being shown; the analysis
    /// always runs as if armed, so re-arming puts the fake straight back on the card.
    func process(_ pb: CVPixelBuffer, time: Double, cfg: RenderConfig, armed: Bool, albedo: CIImage?) -> Output {
        let maxDelay = max(0, min(8, cfg.headStartFrames))
        // The camera reuses its buffers, so keep our own copy while the frame waits to be shown.
        let image = maxDelay > 0 ? (copy(pb).map { CIImage(cvPixelBuffer: $0) } ?? CIImage(cvPixelBuffer: pb))
                                 : CIImage(cvPixelBuffer: pb)
        var slot = decide(pb, image: image, time: time, cfg: cfg)
        slot.status.overlay = slot.draw
        queue.append(slot)
        inputCount += 1
        tick += 1
        if slot.draw { backfill() }
        drawnStreak = slot.draw ? drawnStreak + 1 : 0
        missStreak = slot.draw ? 0 : missStreak + 1
        if drawnStreak >= 6 { live = true }
        if missStreak >= 15 { live = false }    // (a short tracking hiccup doesn't count)

        // The head start is only needed for the moment the card comes into view. Once the fake has
        // been on the card for a few frames the picture catches up to live (skipping a frame now
        // and then while the picture is still), and it falls back behind again (showing the odd
        // frame twice, again only while still) once the card has gone.
        let target = (cfg.adaptiveHeadStart && live) ? 0 : maxDelay
        while queue.count > maxDelay + 1 { queue.removeFirst() }
        let behind = queue.count - 1
        guard cfg.adaptiveHeadStart else {
            // fixed head start (test harness): fill it up at the start, then keep it
            if behind < target { return show(queue[0], cfg: cfg, armed: armed, albedo: albedo, advance: false) }
            return show(queue.removeFirst(), cfg: cfg, armed: armed, albedo: albedo)
        }
        // Catching up (skipping a frame) or falling back (showing one twice) only while the picture
        // is nearly still, at most every other frame, so neither is ever seen; if the picture keeps
        // moving, do it anyway after a while.
        ticksSinceAdjust += 1
        let around = queue.prefix(2).map(\.motion).max() ?? 1
        let mayAdjust = ticksSinceAdjust >= 2 && (around < FramePipeline.stillMotion || ticksSinceAdjust >= 24)
        if behind > target && mayAdjust && queue.count >= 2 {
            ticksSinceAdjust = 0
            queue.removeFirst()
        } else if behind < target && mayAdjust {
            ticksSinceAdjust = 0
            return show(queue[0], cfg: cfg, armed: armed, albedo: albedo, advance: false)
        }
        return show(queue.removeFirst(), cfg: cfg, armed: armed, albedo: albedo)
    }

    /// Mean brightness change per pixel (0...1, at 1/8 scale) below which the picture counts as
    /// still: a hand-held phone pointed at a still scene, sensor noise.
    static let stillMotion = 0.012

    /// Mean absolute difference between two frames' smallest pyramid levels.
    private static func motion(_ a: CardTracker.FrameView?, _ b: CardTracker.FrameView?) -> Double {
        guard let a = a?.pyramid.last, let b = b?.pyramid.last, a.w == b.w, a.h == b.h, !a.px.isEmpty else { return 1 }
        var sum: Float = 0
        a.px.withUnsafeBufferPointer { pa in
            b.px.withUnsafeBufferPointer { pb in
                for i in 0..<pa.count { sum += abs(pa[i] - pb[i]) }
            }
        }
        return Double(sum) / Double(a.px.count)
    }

    /// The frames still waiting to be shown (for the test harness at the end of a clip).
    func drain(cfg: RenderConfig, armed: Bool, albedo: CIImage?) -> [Output] {
        var out: [Output] = []
        while !queue.isEmpty { out.append(show(queue.removeFirst(), cfg: cfg, armed: armed, albedo: albedo)) }
        return out
    }

    // MARK: - Deciding

    private func decide(_ pb: CVPixelBuffer, image frame: CIImage, time: Double, cfg: RenderConfig) -> Slot {
        let state = tracker.process(pb, time: time)
        var st = Status()
        st.source = tracker.lastSource
        st.score = tracker.lastScore
        st.debug = tracker.lastDebug
        if tracker.lastPictureMatch > -0.5 { st.debug += String(format: " pic %.2f", tracker.lastPictureMatch) }
        st.raw = tracker.lastRawQuad
        var slot = Slot(index: inputCount, time: time, image: frame, view: tracker.lastView,
                        quad: nil, draw: false, analysis: nil, status: st)
        slot.motion = FramePipeline.motion(lastDecided?.view, tracker.lastView)

        switch state {
        case .locked(let q):
            slot.status.tracking = "locked"
            slot.quad = q
            lastCardSeen = time
            var a = compositor.analyze(frame: frame, quad: q)
            // Held so close that hardly any of the card's border is in the picture: nothing to
            // measure the light from, so keep the last good measurement rather than giving up.
            if a == nil, overlayOn, let last = lastAnalysis, FramePipeline.fractionInside(q, frame.extent) < 0.9 { a = last }
            slot.status.cardness = a?.cardness ?? 0
            let isFace = a.map { $0.faceUp || !cfg.facesOnly } ?? false
            // Turn on as soon as it plausibly is the card (the real card must never show); once on,
            // stay on while the tracker holds the card, even when glare or a steep tilt make the
            // border look less like paper.
            let threshold = overlayOn ? 0.12 : 0.3
            let good = a != nil && slot.status.cardness > threshold && isFace
            // A card just found has to hold for a frame or two before the fake goes on, so a wrong
            // find that lasts a single frame never shows. Only while the picture runs behind (the
            // head start hides the wait, and the frames before get the fake painted on too).
            let confirmed = overlayOn || tracker.stableFrames >= Tuning.confirmFrames
                || queue.count < Tuning.confirmFrames - 1
            if good && !confirmed {
                badStreak = 0
                lastAnalysis = a
                slot.analysis = a
                slot.status.reason = "confirming"
            } else if good {
                badStreak = 0
                lastAnalysis = a
                overlayOn = true
            } else {
                badStreak += 1
                if badStreak >= badFramesToDrop { overlayOn = false }
                // Locked onto something that isn't a card face for a while: search again.
                if isFace && badStreak % 10 == 0 { tracker.forceRedetect() }
            }
            if a == nil { slot.status.reason = "analysis failed" }
            else if !isFace { slot.status.reason = "card back" }
            else if !good { slot.status.reason = String(format: "low paper %.2f", slot.status.cardness) }
            else if slot.status.reason != "confirming" { slot.status.reason = "ok" }
            if overlayOn, let use = good ? a : lastAnalysis {
                slot.draw = true
                slot.analysis = use
            }

        case .coasting(let q, let n):
            slot.status.tracking = "coast \(n)"
            slot.quad = q
            // Bridge a short tracking hiccup (fast blur), but only while there is still clearly a card
            // there: a fake card painted on an empty wall would give the trick away.
            // (a card on its way out of the picture still counts while most of it is in view, so the
            // real card doesn't flash up as it leaves; once it is mostly gone, nothing is painted)
            // (the printing must still be there too: on a white wall or ceiling the paper test
            // alone passes where the card has just been, and the fake was left hanging in the air)
            let stillThere = n <= holdFrames && FramePipeline.fractionInside(q, frame.extent) > 0.6
                && (compositor.analyze(frame: frame, quad: q)?.cardness ?? 0) > 0.5
                && (slot.view.map { CardTracker.hasPrinting($0, q) } ?? false)
            if overlayOn, stillThere, let a = lastAnalysis {
                slot.draw = true
                slot.analysis = a
                slot.status.reason = "holding"
            } else {
                overlayOn = false
                slot.status.reason = "lost"
            }

        case .searching:
            slot.status.tracking = "searching"
            slot.status.reason = "no card"
            overlayOn = false
        }

        // The tracker let go but the fake was on the card a frame ago (usually the card going out of
        // the picture fast, or a hard tilt): follow the card's own printing from that frame. While
        // any of the card can still be followed it is still there and must stay covered.
        if !slot.draw, followStreak < 8, let prev = lastDecided, prev.draw, let pq = prev.quad,
           let a = prev.view, let b = slot.view, let q = CardTracker.follow(pq, from: a, to: b),
           q.boundingBox.intersects(frame.extent.insetBy(dx: 2, dy: 2)), CardTracker.hasPrinting(b, q) {
            let an = compositor.analyze(frame: frame, quad: q)
            if an.map({ $0.cardness > 0.3 }) ?? true {
                slot.quad = q
                slot.draw = true
                slot.analysis = an ?? prev.analysis
                slot.status.reason = "followed"
                followStreak += 1
            }
        }
        if slot.status.reason != "followed" { followStreak = 0 }
        lastDecided = slot
        return slot
    }

    /// The newest frame has the fake on it; the frames before it that are still waiting to be shown
    /// may already have had part of the card in them. Follow the card back through them.
    private func backfill() {
        guard queue.count >= 2 else { return }
        var j = queue.count - 2
        while j >= 0, !queue[j].draw {
            let next = queue[j + 1]
            guard let nq = next.quad else { break }
            var q: Quad?
            if let a = next.view, let b = queue[j].view {
                q = CardTracker.follow(nq, from: a, to: b)
            }
            // A frame that was waiting for the new card to be confirmed has its own measurement
            // (used when the card can't be followed back to it).
            if q == nil, queue[j].status.reason == "confirming", let own = queue[j].quad, let a = queue[j].analysis {
                queue[j].quad = own
                queue[j].draw = true
                queue[j].analysis = a
                queue[j].status.reason = "confirmed"
                queue[j].status.overlay = true
                j -= 1
                continue
            }
            if q == nil {
                // Too blurred or too little in view to follow: continue its motion backwards, but
                // only onto the card's printing (a guess that lands beside the card is worse than
                // nothing).
                if j + 2 < queue.count, let after = queue[j + 2].quad {
                    q = Quad(zip(nq.points, after.points).map { CGPoint(x: 2 * $0.x - $1.x, y: 2 * $0.y - $1.y) })
                } else {
                    q = nq
                }
                if let guess = q, let v = queue[j].view, !CardTracker.hasPrinting(v, guess) { q = nil }
            }
            guard let found = q else { break }
            // Stop once the card is out of the picture (nothing to cover).
            let ext = queue[j].image.extent
            guard found.boundingBox.intersects(ext.insetBy(dx: 2, dy: 2)) else { break }
            let a = compositor.analyze(frame: queue[j].image, quad: found)
            // Something clearly not a card there: the card wasn't in the picture yet.
            if let a = a, a.cardness < 0.25 { break }
            queue[j].quad = found
            queue[j].draw = true
            queue[j].analysis = a ?? next.analysis
            queue[j].status.reason = "ahead"
            queue[j].status.overlay = true
            j -= 1
        }
    }

    // MARK: - Showing

    private func show(_ s: Slot, cfg: RenderConfig, armed: Bool, albedo: CIImage?, advance: Bool = true) -> Output {
        var out = s.image
        var drew = false
        var st = s.status
        if s.draw, armed, let q = s.quad, let a = s.analysis, let albedo = albedo {
            // (motion blur reads the previous shown frame: only advance it for frames really shown)
            out = compositor.overlay(frame: s.image, quad: q, albedo: albedo, analysis: a, cfg: cfg, advance: advance)
            drew = true
        } else if cfg.debug, let q = s.quad {
            out = compositor.debugCorners(s.image, quad: q,
                                          color: armed ? CIColor(red: 1, green: 0.2, blue: 0.2) : CIColor(red: 0.3, green: 0.6, blue: 1))
        }
        if !armed { st.reason = "revealed" }
        st.overlay = drew
        return Output(image: out.cropped(to: s.image.extent), drewOverlay: drew, quad: s.quad, status: st,
                      frameIndex: s.index, time: s.time)
    }

    private func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        let fmt = CVPixelBufferGetPixelFormatType(src)
        if pool == nil || poolSize != CGSize(width: w, height: h) {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: fmt,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            var p: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, [kCVPixelBufferPoolMinimumBufferCountKey as String: 12] as CFDictionary,
                                    attrs as CFDictionary, &p)
            pool = p
            poolSize = CGSize(width: w, height: h)
        }
        guard let pool = pool else { return nil }
        var dst: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst)
        guard let d = dst, !CVPixelBufferIsPlanar(src) else { return nil }
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(d, [])
        defer {
            CVPixelBufferUnlockBaseAddress(d, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let sp = CVPixelBufferGetBaseAddress(src), let dp = CVPixelBufferGetBaseAddress(d) else { return nil }
        let sr = CVPixelBufferGetBytesPerRow(src), dr = CVPixelBufferGetBytesPerRow(d)
        if sr == dr {
            memcpy(dp, sp, sr * h)
        } else {
            let n = min(sr, dr)
            for y in 0..<h { memcpy(dp + y * dr, sp + y * sr, n) }
        }
        if let attachments = CVBufferCopyAttachments(src, .shouldPropagate) {
            CVBufferSetAttachments(d, attachments, .shouldPropagate)
        }
        return d
    }
}
