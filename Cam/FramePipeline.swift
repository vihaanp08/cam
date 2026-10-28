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

    /// Frames to keep the fake card on after the tracker loses the card (0.1 s at 30 fps), and only
    /// one while the card's own printing isn't known yet: held longer, a card that had left fast
    /// left the fake behind on the wall.
    var holdFrames = 3
    /// Consecutive "not a card face" frames before the fake turns off (card turned around, etc.).
    var badFramesToDrop = 8
    /// A card coming in at the edge of the picture that isn't covered yet when its frame is due: the
    /// frame before is shown again instead (the picture holds for a moment) for up to this many
    /// frames, while the tracker makes sure of the card and the fake is painted back onto the frames
    /// waiting to be shown. A card coming in fast and blurred can take longer to be sure of than the
    /// head start, and its index, the first thing to come into view, showed. Only while the fake is
    /// to be shown, and at most once per card coming in.
    var entryHoldMax = 3
    private var entryHolds = 0
    private var entryQuiet = 100
    /// The frame shown last (shown again while holding).
    private var lastShown: Slot?

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
        /// Printing at the edge of the picture and no card followed: a card may be coming in.
        var entering = false
    }
    private var queue: [Slot] = []
    private var inputCount = 0
    /// The previous frame's decision, and how many frames in a row the card has been followed
    /// on its own printing after the tracker let go of it.
    private var lastDecided: Slot?
    /// The decision before that (for the card's motion when it can't be followed).
    private var beforeLastDecided: Slot?
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
        beforeLastDecided = nil
        followStreak = 0
        drawnStreak = 0
        missStreak = 0
        live = false
        ticksSinceAdjust = 0
        entryHolds = 0
        entryQuiet = 100
        lastShown = nil
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
    /// `armedNow`, when given, is asked again just before the frame is shown: a snap or a button press
    /// that came while this frame was being analysed already counts for it.
    func process(_ pb: CVPixelBuffer, time: Double, cfg: RenderConfig, armed: Bool, albedo: CIImage?,
                 armedNow: (() -> Bool)? = nil) -> Output {
        let maxDelay = max(0, min(8, cfg.headStartFrames))
        // The camera reuses its buffers, so keep our own copy while the frame waits to be shown.
        let image = maxDelay > 0 ? (copy(pb).map { CIImage(cvPixelBuffer: $0) } ?? CIImage(cvPixelBuffer: pb))
                                 : CIImage(cvPixelBuffer: pb)
        var slot = decide(pb, image: image, time: time, cfg: cfg)
        slot.status.overlay = slot.draw
        queue.append(slot)
        inputCount += 1
        tick += 1
        if slot.draw { backfill(redo: tracker.lastWasCorrection) }
        drawnStreak = slot.draw ? drawnStreak + 1 : 0
        missStreak = slot.draw ? 0 : missStreak + 1
        if drawnStreak >= 3 { live = true }
        // (a short tracking hiccup doesn't count; once the card has gone, the head start is built
        // back up straight away, ready for it coming back)
        if missStreak >= 10 { live = false }
        let showArmed = armedNow?() ?? armed

        // A card coming in that isn't covered yet when its frame is due: hold (see `entryHoldMax`).
        entryQuiet = slot.entering ? 0 : entryQuiet + 1
        if entryQuiet >= 15 { entryHolds = 0 }
        while queue.count > maxDelay + 1 + entryHoldMax { queue.removeFirst() }
        if holdForEntry(armed: showArmed, maxDelay: maxDelay), let last = lastShown {
            entryHolds += 1
            return show(last, cfg: cfg, armed: showArmed, albedo: albedo, advance: false)
        }

        // The head start is only needed for the moment the card comes into view. Once the fake is on
        // the card the picture catches up to live within a few frames (a frame skipped every other
        // frame, moving or not: a picture running behind the camera is what reads as lag), and it
        // falls back behind again (showing the odd frame twice, only while the picture is still, so
        // it is never seen) once the card has gone.
        let target = (cfg.adaptiveHeadStart && live) ? 0 : maxDelay
        let behind = queue.count - 1
        guard cfg.adaptiveHeadStart else {
            // fixed head start (test harness): fill it up at the start, then keep it
            if behind < target {
                lastShown = queue[0]
                return show(queue[0], cfg: cfg, armed: showArmed, albedo: albedo, advance: false)
            }
            return showNext(cfg: cfg, armed: showArmed, albedo: albedo)
        }
        ticksSinceAdjust += 1
        let around = queue.prefix(2).map(\.motion).max() ?? 1
        let mayFallBack = ticksSinceAdjust >= 2 && (around < FramePipeline.stillMotion || ticksSinceAdjust >= 24)
        if behind > target && ticksSinceAdjust >= 2 && queue.count >= 2 {
            ticksSinceAdjust = 0
            queue.removeFirst()
        } else if behind < target && mayFallBack {
            ticksSinceAdjust = 0
            lastShown = queue[0]
            return show(queue[0], cfg: cfg, armed: showArmed, albedo: albedo, advance: false)
        }
        return showNext(cfg: cfg, armed: showArmed, albedo: albedo)
    }

    /// The frame due next has a card coming in that isn't covered (yet), and the picture may still
    /// hold for it (see `entryHoldMax`).
    private func holdForEntry(armed: Bool, maxDelay: Int) -> Bool {
        guard armed, maxDelay > 0, entryHoldMax > 0, entryHolds < entryHoldMax, lastShown != nil,
              let head = queue.first else { return false }
        return head.entering && !head.draw
    }

    private func showNext(cfg: RenderConfig, armed: Bool, albedo: CIImage?) -> Output {
        let s = queue.removeFirst()
        lastShown = s
        return show(s, cfg: cfg, armed: armed, albedo: albedo)
    }

    /// Test harness (fixed head start): a frame kept back while the picture held for a card coming
    /// in is shown now, one at a time, so the head start goes back to what it was. (The app skips
    /// such frames on its way back to live; the harness scores every frame.)
    func catchUp(cfg: RenderConfig, armed: Bool, albedo: CIImage?) -> Output? {
        let maxDelay = max(0, min(8, cfg.headStartFrames))
        guard !cfg.adaptiveHeadStart, queue.count > maxDelay + 1, !holdForEntry(armed: armed, maxDelay: maxDelay) else { return nil }
        return showNext(cfg: cfg, armed: armed, albedo: albedo)
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
        slot.entering = tracker.lastEntering

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
            // head start hides the wait, and the frames before get the fake painted on too), and
            // always on the very first frame (a find there has nothing to confirm it yet).
            let confirmed = overlayOn || tracker.stableFrames >= Tuning.confirmFrames
                || (inputCount > 0 && queue.count < Tuning.confirmFrames - 1)
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
            // (and once the card's picture is known, it must be the card's own printing there: after
            // a sudden fast move the place the card has just left can still have some of it)
            let stillThere = n <= (tracker.hasPicture ? holdFrames : 1) && FramePipeline.fractionInside(q, frame.extent) > 0.6
                && (compositor.analyze(frame: frame, quad: q)?.cardness ?? 0) > 0.5
                && (slot.view.map { CardTracker.hasPrinting($0, q) } ?? false)
                && (slot.view.flatMap { tracker.pictureMatch(q, in: $0) }.map { $0 >= 0.75 } ?? true)
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
        // (a few frames at most, and only onto the card's own printing: followed longer, or onto
        // anything printed, the fake was left on the wall where the card had been)
        if !slot.draw, followStreak < (tracker.hasPicture ? 4 : 2), let prev = lastDecided, prev.draw, let pq = prev.quad,
           let b = slot.view {
            var found: Quad?
            var how = "followed"
            if let a = prev.view, let q = CardTracker.follow(pq, from: a, to: b),
               q.boundingBox.intersects(frame.extent.insetBy(dx: 2, dy: 2)), CardTracker.hasPrinting(b, q),
               (tracker.pictureMatch(q, in: b).map({ $0 >= 0.8 }) ?? true) {
                found = q
            }
            // Going out of the picture too fast and blurred to follow: it carries on the way it was
            // going, as long as that lands on its own printing (the strip of it still in view).
            // (only on its way out: a card that stops at the edge isn't there, and the guess is
            // dropped as soon as it is off the card's printing)
            if found == nil, Tuning.exitCarryOn, let pp = beforeLastDecided, pp.draw, let ppq = pp.quad {
                let guess = Quad(zip(pq.points, ppq.points).map { CGPoint(x: 2 * $0.x - $1.x, y: 2 * $0.y - $1.y) })
                let step = zip(pq.points, ppq.points).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? 0
                let ext = frame.extent
                let fast = step > 0.025 * min(ext.width, ext.height)
                let leaving = FramePipeline.fractionInside(guess, ext) < FramePipeline.fractionInside(pq, ext) - 0.02
                if fast, leaving, guess.boundingBox.intersects(ext.insetBy(dx: 2, dy: 2)),
                   CardTracker.hasPrinting(b, guess, minSamples: 8),
                   (tracker.pictureMatch(guess, in: b).map({ $0 >= 0.7 }) ?? true) {
                    found = guess
                    how = "carried on"
                }
            }
            if let q = found {
                let an = compositor.analyze(frame: frame, quad: q)
                if an.map({ $0.cardness > 0.45 }) ?? true {
                    slot.quad = q
                    slot.draw = true
                    slot.analysis = an ?? prev.analysis
                    slot.status.reason = how
                    followStreak += 1
                }
            }
        }
        if slot.status.reason != "followed" && slot.status.reason != "carried on" { followStreak = 0 }
        beforeLastDecided = lastDecided
        lastDecided = slot
        return slot
    }

    /// The newest frame has the fake on it; the frames before it that are still waiting to be shown
    /// may already have had part of the card in them. Follow the card back through them.
    /// `redo`: the outline was just corrected (it had been on part of the card, or was a poor fit),
    /// so the waiting frames already given the old outline are followed back and redone too, and
    /// the fake covers the whole card on them as well.
    private func backfill(redo: Bool = false) {
        guard queue.count >= 2 else { return }
        var j = queue.count - 2
        while j >= 0, redo, queue[j].draw {
            guard let nq = queue[j + 1].quad, let a = queue[j + 1].view, let b = queue[j].view,
                  let q = CardTracker.follow(nq, from: a, to: b), CardTracker.hasPrinting(b, q) else { return }
            queue[j].quad = q
            queue[j].status.reason = "redone"
            j -= 1
        }
        while j >= 0, !queue[j].draw {
            let next = queue[j + 1]
            guard let nq = next.quad else { break }
            var q: Quad?
            if let a = next.view, let b = queue[j].view {
                q = CardTracker.follow(nq, from: a, to: b)
                // (a follow that doesn't land on the card's printing, once that is known, is no follow)
                if let f = q, (tracker.pictureMatch(f, in: b) ?? 1) < 0.75 { q = nil }
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
                // (and, once the card's picture is known, it must be seen to be the card's printing)
                // (a card just coming into view is a strip at the edge of the picture: a little of its
                // printing in view is enough, its index is what must not show)
                if let guess = q, let v = queue[j].view {
                    let printed = CardTracker.hasPrinting(v, guess, minSamples: 8)
                    let match = tracker.hasPicture ? tracker.pictureMatch(guess, in: v) : nil
                    if !printed || (match.map { $0 < 0.75 } ?? false) { q = nil }
                }
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
