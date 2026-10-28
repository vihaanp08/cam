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
    /// The card's own printing as seen face-on (see "The card's picture" below), and when it was taken.
    private var cardPicture: [Float]?
    private var cardPictureFrame = 0
    /// The last frame the followed outline matched the card's picture.
    private var pictureSeenFrame = 0
    /// How well this frame's outline lines up with the card's picture (-1 when not checked).
    private(set) var lastPictureMatch: Float = -1
    /// Frames in a row the followed outline had all four (at least three) sides on the card's
    /// edges, and how far it moved since the frame before (gray pixels): the picture is only taken
    /// from an outline that is exact, and every later outline is lined up with it.
    private var snapStreak4 = 0
    private var snapStreak3 = 0
    private var lastMotion: CGFloat = .infinity
    /// Frames in a row a steady lock didn't match the picture at all (a different card is being held up).
    private var pictureMismatch = 0
    /// The last lock change replaced part of the card (or a poor fit) with the whole card: frames
    /// already painted with the old outline and still waiting to be shown should be redone.
    private(set) var lastWasCorrection = false
    /// The lock was taken on a card coming into view, by where its corner index is, and much of it is
    /// still out of view (see `completions`).
    private var entryLock = false
    /// This frame showed printing on paper right at the edge of the picture and no card is being
    /// followed: a card may be coming into view that isn't taken yet (FramePipeline holds the picture
    /// for a moment rather than show it uncovered).
    private(set) var lastEntering = false
    /// The followed outline has the card carrying on past one of its sides (see `printingPast`).
    private var lockPrinting = false
    /// Card-likeness of the current lock (PlanarTracker.cardScore), -1 when none.
    private(set) var lastScore = -1.0
    /// What detection saw on the last frame it ran (for the test harness and the debug view).
    private(set) var lastDebug = ""
    /// Every scored candidate of the last detection, when `keepCandidates` is on (test harness).
    var keepCandidates = false
    private(set) var lastCandidates: [(how: String, score: Double, quad: Quad, note: String)] = []
    /// This frame's measured outline before smoothing (nil when not locked), for the test harness.
    private(set) var lastRawQuad: Quad?
    /// Below this a shape isn't accepted as a card.
    var minScore = 0.14

    /// How many frames we keep predicting the pose after losing the card.
    let maxCoastFrames = 30
    /// Vision works on a copy of the frame scaled by this factor.
    var analysisScale: CGFloat = 0.5
    /// Or, when set, scaled so its long side is about this many pixels (the same detail whatever size
    /// the frames come in: the camera's 1920 x 1080 gives the usual half size).
    var analysisLongSide: CGFloat?
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

    /// One processed frame's gray pyramid, kept by the pipeline so it can look back in time.
    struct FrameView {
        let pyramid: [GrayImage]
        let W: CGFloat
        let H: CGFloat
        let topDown: Bool
        func toGray(_ p: CGPoint) -> CGPoint {
            let gw = CGFloat(pyramid[0].w), gh = CGFloat(pyramid[0].h)
            return CGPoint(x: p.x * gw / W, y: topDown ? (H - p.y) * gh / H : p.y * gh / H)
        }
        func toCI(_ p: CGPoint) -> CGPoint {
            let gw = CGFloat(pyramid[0].w), gh = CGFloat(pyramid[0].h)
            return CGPoint(x: p.x * W / gw, y: topDown ? H - p.y * H / gh : p.y * H / gh)
        }
    }
    /// The last processed frame (nil when the flow tracker is off).
    private(set) var lastView: FrameView?

    /// Where was the card (`quad` in frame `a`) in the earlier frame `b`? Follows the card's own
    /// printing backwards with optical flow, then snaps to the real edges. nil when too little of
    /// the card can be followed.
    static func follow(_ quad: Quad, from a: FrameView, to b: FrameView) -> Quad? {
        guard !a.pyramid.isEmpty, !b.pyramid.isEmpty else { return nil }
        let gq = quad.points.map(a.toGray)
        let pts = PlanarTracker.goodFeatures(a.pyramid[0], quad: gq)
        guard pts.count >= 8 else { return nil }
        let fwd = PlanarTracker.lucasKanade(prev: a.pyramid, cur: b.pyramid, pts: pts)
        var o: [CGPoint] = [], m: [CGPoint] = []
        for (p, q) in zip(pts, fwd) { if let q = q { o.append(p); m.append(q) } }
        guard m.count >= 8 else { return nil }
        let back = PlanarTracker.lucasKanade(prev: b.pyramid, cur: a.pyramid, pts: m)
        var aa: [CGPoint] = [], bb: [CGPoint] = []
        for i in 0..<m.count {
            if let r = back[i], hypot(r.x - o[i].x, r.y - o[i].y) < 1.0 { aa.append(o[i]); bb.append(m[i]) }
        }
        guard aa.count >= 8, let fit = PlanarTracker.ransacSimilarity(aa, bb),
              fit.1.filter({ $0 }).count >= 6 else { return nil }
        var nq = gq.map { fit.0.apply($0) }
        let (sq, n) = PlanarTracker.snapEdges(b.pyramid[0], quad: nq)
        if n > 0 { nq = sq }
        guard nq.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        return Quad(nq.map(b.toCI))
    }

    /// How well `quad` (Core Image coordinates) in that frame shows the card's own printing (its
    /// picture, see "The card's picture"): about 0.9-1 on the card, low anywhere else (where the card
    /// was a moment ago before a fast move, the wall beside it). nil while the picture isn't known.
    var hasPicture: Bool { cardPicture != nil }

    func pictureMatch(_ quad: Quad, in view: FrameView) -> Float? {
        guard let pic = cardPicture, !view.pyramid.isEmpty else { return nil }
        let gq = quad.points.map(view.toGray)
        guard gq.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let m = CardTracker.bestPictureMatch(pic, view.pyramid, gq).0
        return m > -1 ? m : nil
    }

    /// Is there printing (a card's pips and indices) inside `quad` in this frame? A plain wall, the
    /// ceiling or the empty space a card has just left has none, so the fake must not be held or
    /// guessed onto it. Core Image coordinates.
    /// (`minSamples`: how many of the points inside it must be in the picture; fewer for the strip of a
    /// card just coming into view)
    static func hasPrinting(_ view: FrameView, _ quad: Quad, minSamples: Int = 30) -> Bool {
        guard let g = view.pyramid.first else { return false }
        let gq = quad.points.map(view.toGray)
        guard gq.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return false }
        var v: [Float] = []
        for j in 0..<14 {
            for i in 0..<10 {
                let a = 0.12 + 0.76 * CGFloat(i) / 9, b = 0.12 + 0.76 * CGFloat(j) / 13
                let top = CGPoint(x: gq[0].x + a * (gq[1].x - gq[0].x), y: gq[0].y + a * (gq[1].y - gq[0].y))
                let bot = CGPoint(x: gq[3].x + a * (gq[2].x - gq[3].x), y: gq[3].y + a * (gq[2].y - gq[3].y))
                let x = Double(top.x + b * (bot.x - top.x)), y = Double(top.y + b * (bot.y - top.y))
                if g.contains(x, y) { v.append(g.sample(Float(x), Float(y))) }
            }
        }
        guard v.count >= max(8, minSamples) else { return false }
        let sorted = v.sorted()
        let paper = sorted[Int(Double(sorted.count - 1) * 0.8)]
        guard paper > 0.3 else { return false }
        let dark = v.filter { $0 < 0.6 * paper }.count
        return Double(dark) / Double(v.count) >= 0.03
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
        cardPicture = nil
        lastPictureMatch = -1
        snapStreak4 = 0
        snapStreak3 = 0
        lastMotion = .infinity
        pictureMismatch = 0
        lastWasCorrection = false
        entryLock = false
        lastEntering = false
        lockPrinting = false
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
        lastView = pyramid.map { FrameView(pyramid: $0, W: W, H: H, topDown: topDown) }

        func toGray(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * gw / W, y: topDown ? (H - p.y) * gh / H : p.y * gh / H)
        }
        func toCI(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * W / gw, y: topDown ? H - p.y * H / gh : p.y * H / gh)
        }
        func quadCI(_ g: [CGPoint]) -> Quad { Quad(g.map(toCI)) }

        var measured: Quad?
        var source = "none"
        var entering = false
        let previousMatch = lastPictureMatch
        lastPictureMatch = -1
        lastWasCorrection = false

        // 1. Follow the card's texture with optical flow.
        var trackedScore = -1.0
        // The followed outline lines up with the card's picture: it is the card, whatever the
        // card-likeness score says (held very close, steeply tilted, a court card's busy face).
        var verified = false
        lockPrinting = false
        if let gq = grayQuad, let prev = prevPyramid, let cur = pyramid, let g = gray {
            if let step = flowStep(prev: prev, cur: cur, quad: gq), plausible(quadCI(step.quad), w: W, h: H) {
                var next = step
                // A side with the card carrying on past it (its paper, with printing on it): a line
                // inside the card, not its edge (a court card's printed frame round its picture). Out
                // to where the card's paper ends; if that can't be found, the whole card takes over
                // as soon as detection finds it. (Not while the outline lines up with the card's
                // picture: that is the card.)
                if Tuning.printingGuard, !(cardPicture != nil && lastPictureMatch >= 0.85) {
                    if let fixed = pushOutPastPrinting(g, paper, next.quad), plausible(quadCI(fixed), w: W, h: H) {
                        next.quad = fixed
                    }
                    lockPrinting = (0..<4).contains { printingPast(paper, next.quad, side: $0) == true }
                }
                trackedScore = cardLikeness(g, paper, next.quad)
                verified = cardPicture != nil && lastPictureMatch >= 0.85
                let inView = FramePipeline.fractionInside(quadCI(next.quad), CGRect(x: 0, y: 0, width: W, height: H))
                // Still looks like a card? Weak frames happen (motion blur, a thumb over an edge), so
                // a weak track is kept while detection looks for something better (step 2); only a
                // long run of clearly-not-a-card frames drops it. (Held so close that much of it is
                // out of the picture, too little of it is in view to judge: a low score then means
                // nothing, and it is followed as long as the flow holds.)
                if trackedScore >= minScore || verified { weakFrames = 0 } else if inView > 0.85 { weakFrames += 1 }
                // Shrunk onto part of the card (paper carries on past two sides): drop it at once so
                // detection replaces it this very frame, never keep painting a card inside the card.
                // (three sides: against a plain white ceiling one or two real edges can look like that)
                // (or, while the lock is new, its corner index runs on past where a card's index ends: a
                // box round part of the card, never to be painted)
                let insideCard = !verified && (PlanarTracker.paperContinuesSides(g, quad: next.quad) >= 3
                    || (stableFrames < 15 && (indexPattern(paper, next.quad).map { $0.misScaled } ?? false)))
                // A good lock that suddenly stops looking like a card at all (a fast exit, a hard
                // blur) has lost the card: don't paint the fake wherever the flow ended up. Drop it
                // and let the short prediction and detection take over.
                // (not when most of the card is out of the picture: held very close, too little of
                // it is in view to score, but the flow is still following it)
                // (and once the card's picture is known: its printing stopped matching from one frame
                // to the next, the outline has lost the card; a fast exit, or the flow slid off)
                let pictureLost = cardPicture != nil && lastPictureMatch >= 0 && lastPictureMatch < 0.7 && previousMatch >= 0.9
                let collapsed = (!verified && trackedScore < 0.03 && lastScore >= 0.2 && inView > 0.8) || pictureLost
                if !insideCard && !collapsed && (weakFrames < 10 || trackedScore >= minScore * 0.5) {
                    grayQuad = next.quad
                    points = next.points
                    measured = quadCI(next.quad)
                    source = next.snapped > 0 ? "flow+edge" : "flow"
                } else {
                    grayQuad = nil
                    trackedScore = -1
                    verified = false
                }
            } else {
                grayQuad = nil
            }
        }
        if grayQuad == nil { snapStreak4 = 0; snapStreak3 = 0 }

        // 2. Detect when lost, every few frames as a check, and sooner while the track looks weak.
        // Also on the first frames of a new lock: if it is only part of the card (half of it, a
        // corner, the box round a court card's picture) the whole card is found before the fake
        // goes on (a new lock is confirmed for a frame or two first, see FramePipeline).
        // And more often while the lock doesn't show a card's corner indices where they should be
        // (it may have slid off the card: one side on the printing or on the palm).
        // (and every frame while the lock is new: a lock on part of the card is replaced by the whole
        // card before the fake has been shown on it for long)
        let lockPattern = grayQuad.flatMap { indexPattern(paper, $0) }
        let lockIndex: Double? = lockPattern.map { $0.both }
        let lockMisScaled: Bool = lockPattern.map { $0.misScaled } ?? false
        // (and every frame while a new card is still coming into view: its outline is a completion
        // of the part in view until all of it is in the picture)
        let comingIn = cardPicture == nil && stableFrames < 30
            && (grayQuad.map { FramePipeline.fractionInside(quadCI($0), CGRect(x: 0, y: 0, width: W, height: H)) < 0.95 } ?? false)
        let runDetect = measured == nil || frameIndex % detectEvery == 0 || (weakFrames > 0 && frameIndex % 2 == 0)
            || stableFrames < 5 || lockMisScaled || ((lockIndex ?? 1) < 0.06 && frameIndex % 3 == 0) || comingIn
            || (lockPrinting && grayQuad != nil)
        if runDetect {
            let cands = detectCandidates(buffer, w: W, h: H)
            if let g = gray {
                // Snap every candidate to the real edges, score it, keep the most card-like.
                var best: (quad: [CGPoint], how: String, score: Double)?
                var bestMatch: Float = -2
                var all: [(quad: [CGPoint], how: String, score: Double)] = []
                var found: [(quad: [CGPoint], how: String, score: Double, match: Float, note: String)] = []
                var tried = 0, shapeOK = 0
                var bestNote = ""
                // The card's size as last seen clearly, and as it is being followed now (when that lock
                // is sound and not brand new).
                var sizeRefs: [CGFloat] = []
                if let lg = lastGood, frameIndex - lg.frame < 90 { sizeRefs.append(lg.size) }
                if let tq = grayQuad, stableFrames >= 8, trackedScore >= 0.3 || verified { sizeRefs.append(quadSize(tq)) }
                let frameRect = CGRect(x: 0, y: 0, width: gw, height: gh)
                // A card cut off by the edge of the picture is completed to a whole card (either way
                // round, PlanarTracker.completeTruncated). From a sliver of it that is a guess: half the
                // card's width passes for its whole width, a card held sideways completes the other way
                // round, and the fake comes out tiny in a corner of the card or giant. So with nothing to
                // go by a completion is only taken once most of it is in view; with the card's size
                // known, it must be that size.
                // (nor a small one: a card brought up to the camera is big, and a small "card" coming in
                // at the edge is a box round its corner index and a pip)
                let minNewShort = 0.2 * CGFloat(min(g.w, g.h))
                // (`early`: taken while much of it is still out of view, by where its corner index is)
                func completions(_ q: [CGPoint]) -> [(quad: [CGPoint], early: Bool)] {
                    let all = PlanarTracker.completeTruncated(q, w: g.w, h: g.h)
                    var out: [(quad: [CGPoint], early: Bool)] = []
                    var waiting: [Int] = []
                    // (the card's recent size; or big enough and enough of it in view)
                    var sized = [Bool](repeating: false, count: all.count)
                    var inViews = [CGFloat](repeating: 0, count: all.count)
                    var usable = [Bool](repeating: false, count: all.count)
                    for (k, full) in all.enumerated() {
                        let s = quadSize(full)
                        sized[k] = sizeRefs.contains(where: { s > 0.75 * $0 && s < 1.35 * $0 })
                        let across = hypot(full[1].x - full[0].x, full[1].y - full[0].y) + hypot(full[2].x - full[3].x, full[2].y - full[3].y)
                        let along = hypot(full[3].x - full[0].x, full[3].y - full[0].y) + hypot(full[2].x - full[1].x, full[2].y - full[1].y)
                        inViews[k] = FramePipeline.fractionInside(Quad(full), frameRect)
                        usable[k] = sized[k] || (min(across, along) / 2 >= minNewShort && inViews[k] >= 0.25)
                    }
                    // Which way round the card's corner index says it is (see `entryEvidence`): once one
                    // way round is proven (an index with the blank paper past it, in view), a way round
                    // with its index boxes in view and blank, or printing past them, is wrong, whatever
                    // its size. (A card brought closer and cut off by the edge of the picture: its short
                    // side taken for the long one made a sideways card about the size it was seen at
                    // before, and that was taken by its size.)
                    let ev: [Int?] = Tuning.entryByIndex ? all.map { entryEvidence(paper, $0) } : [Int?](repeating: nil, count: all.count)
                    let proven = all.indices.contains { usable[$0] && ev[$0] == 2 }
                    for (k, full) in all.enumerated() where usable[k] {
                        if proven, let e = ev[k], e <= 0 { continue }
                        if sized[k] || inViews[k] >= 0.8 { out.append((full, false)) } else { waiting.append(k) }
                    }
                    // Coming into view: the fake must cover the card from the moment it shows, long
                    // before most of it is in the picture. A card's corner index is long and thin along
                    // the card, so of the two ways round only the right one has an index where the card
                    // shows one (see `entryIndex`): that one is taken as soon as an index is in view, as
                    // long as no other way of completing it has one too.
                    // (and only when every other way of completing it is in view enough to be seen to
                    // be wrong: blank index boxes, or printing carrying on past them. One the other way
                    // round with the paper past its index box not in view yet may be the right one.)
                    if Tuning.entryByIndex && !waiting.isEmpty {
                        for k in waiting where ev[k] == 2 {
                            if all.indices.allSatisfy({ $0 == k || ev[$0] == 0 || ev[$0] == -1 }) { out.append((all[k], true)) }
                        }
                    }
                    return out
                }
                // (an outline lying along the edge of the picture, with nothing to go by: a card cut off
                // there, which only its completions above stand for)
                let band = 0.04 * CGFloat(min(g.w, g.h))
                func cutOff(_ q: [CGPoint]) -> Bool {
                    func border(_ p: CGPoint) -> Int? {
                        if p.x <= band { return 0 }
                        if p.x >= CGFloat(g.w - 1) - band { return 1 }
                        if p.y <= band { return 2 }
                        if p.y >= CGFloat(g.h - 1) - band { return 3 }
                        return nil
                    }
                    for k in 0..<4 {
                        if let a = border(q[k]), let b = border(q[(k + 1) % 4]), a == b { return true }
                    }
                    return false
                }
                func consider(_ q: [CGPoint], _ how: String, completed: Bool = false) {
                    tried += 1
                    var gq = q
                    let (sq, n) = PlanarTracker.snapEdges(g, quad: gq)
                    if n > 0 { gq = sq }
                    guard plausible(quadCI(gq), w: W, h: H) else { return }
                    if !completed && sizeRefs.isEmpty && cutOff(gq) { return }
                    shapeOK += 1
                    // Prefer the card's recent size when re-finding it or when the current lock is
                    // healthy, but never let it stop a clearly better card replacing a weak lock
                    // (the weak lock may itself have been a wrong, odd-sized shape).
                    let useSize = grayQuad == nil || trackedScore >= minScore
                    let sz = useSize ? sizeConsistency(gq) : 1
                    var sc = cardLikeness(g, paper, gq) * sz
                    let note = partsNote + String(format: " size %.2f", sz)
                    // Fully in view but not a card face (a printed sheet on the wall, a poster), or a box
                    // round part of a card (printing carries on past where its corner index should end),
                    // and not where the card was just seen: it can't be the card.
                    if sc > 0, let ip = indexPattern(paper, gq), !ip.cardLike || ip.misScaled, !nearLastCard(gq) { sc *= 0.3 }
                    // (or the card carries on past one of its sides: a box inside the card, such as a
                    // court card's printed frame, which looks like a card of its own)
                    if Tuning.printingGuard, sc > 0, (0..<4).contains(where: { printingPast(paper, gq, side: $0) == true }) { sc *= 0.3 }
                    // How well its printing matches the card's picture, once that is known (turned
                    // the way the picture was taken)
                    var m: Float = -2
                    if let pic = cardPicture, let pyr = pyramid {
                        let (mm, turned) = CardTracker.bestPictureMatch(pic, pyr, gq)
                        gq = turned
                        m = mm
                    }
                    found.append((gq, how, sc, m, note))
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
                    for c in completions(tq) { consider(c.quad, c.early ? "flow entry" : "flow part", completed: true) }
                    // the current lock might be half the card, or have grown onto the fingers
                    for bigger in PlanarTracker.extendAcrossWeakSides(g, quad: tq) { consider(bigger, "flow x2") }
                    if let r = refitInward(g, paper, tq) { consider(r, "flow fit") }
                }
                // Cards found by their printing (works when the outline is hard to see).
                var sawPrinting = false
                if let pr = paper {
                    for q in InkFinder.candidates(pr) {
                        sawPrinting = true
                        // (printing on paper right at the edge of the picture, with no card being
                        // followed: a card may be coming into view, see `lastEntering`)
                        let edge = 0.06 * CGFloat(min(g.w, g.h))
                        if grayQuad == nil, q.contains(where: { $0.x <= edge || $0.y <= edge || $0.x >= CGFloat(g.w - 1) - edge || $0.y >= CGFloat(g.h - 1) - edge }) {
                            entering = true
                        }
                        if let f = InkFinder.refine(g, pr, q) {
                            consider(f, "ink")
                            for c in completions(f) { consider(c.quad, c.early ? "ink entry" : "ink part", completed: true) }
                        }
                    }
                }
                for (q, how) in cands {
                    let gq = q.points.map(toGray)
                    consider(gq, how)
                    // A card only partly in view (coming in or going out): complete it to a full
                    // card so the fake covers it (see `completions` for when that is believed).
                    for c in completions(gq) { consider(c.quad, how + (c.early ? " entry" : " part"), completed: true) }
                    // Vision's outline often includes the fingers behind the card: pull each side in
                    // to the real paper edge.
                    if let r = refitInward(g, paper, gq) { consider(r, how + " fit") }
                    if let pr = paper, let r = InkFinder.refine(g, pr, gq) { consider(r, how + " ref") }
                    for bigger in PlanarTracker.extendAcrossWeakSides(g, quad: gq) { consider(bigger, how + "x2") }
                }
                // Held so close that it runs off two edges of the picture (see `cornerFits`): from the
                // corner in view of the first few outlines found round the part in view.
                if Tuning.cornerSearch, grayQuad == nil || lockPrinting, let pic = cardPicture, let pyr = pyramid {
                    var fits = 0
                    for (q, _) in cands.prefix(3) where fits < 2 {
                        if let f = cornerFits(q.points.map(toGray), g: g, pyr: pyr, pic: pic) {
                            consider(f, "corner", completed: true)
                            fits += 1
                        }
                    }
                }
                // Nothing found lines up with the card's picture, and there is printing on paper in
                // view: look for the picture itself, anywhere, at any size (held close with no outline
                // of it found: its edges against a pale wall, fingers along them). Every other frame.
                if Tuning.pictureSearch, grayQuad == nil || lockPrinting, sawPrinting, frameIndex % 2 == 0,
                   let pic = cardPicture, let pyr = pyramid, let ref = lastGood?.quad,
                   !found.contains(where: { $0.match >= 0.85 }),
                   let f = pictureSearch(pic, pyr, ref: ref, w: CGFloat(g.w), h: CGFloat(g.h)) {
                    consider(f, "search", completed: true)
                }
                // Once the card's picture is known: line the most promising finds up with it (a find
                // a few pixels off, or with a corner on a finger, then fits exactly and matches well),
                // and let the match weigh in, so an outline whose printing doesn't line up with the
                // card's (half the card, the box round its picture, a printed sheet, another object)
                // can't win. A different card held up once this one has gone needs no match: a clear
                // card face, both corner indices showing, is enough.
                if let pic = cardPicture, let pyr = pyramid {
                    var promising: [Int] = found.indices.filter { k in
                        found[k].match >= 0.3 && found[k].match < 0.95 && (found[k].score > 0.02 || found[k].match >= 0.6)
                    }
                    let key: [Double] = found.map { Double($0.match) + $0.score }
                    promising.sort { key[$0] > key[$1] }
                    for k in promising.prefix(3) {
                        let q = found[k].quad
                        guard let al = CardTracker.alignPicture(pic, pyr, q, iterations: 12), al.match > found[k].match else { continue }
                        let moved: CGFloat = zip(al.quad, q).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? .infinity
                        if moved < 0.15 * quadSize(q), plausible(quadCI(al.quad), w: W, h: H) {
                            found[k].quad = al.quad
                            found[k].match = al.match
                        }
                    }
                }
                let cardGone = grayQuad == nil && frameIndex - (lastGood?.frame ?? -1000) > 15
                var bestIdx = -1
                for k in found.indices {
                    var sc = found[k].score
                    let m = found[k].match
                    if m > -2 {
                        let newCard = cardGone && (indexPattern(paper, found[k].quad)?.both ?? 0) >= 0.15
                        if !newCard { sc *= Double(max(0.15, min(1, (m - 0.45) / 0.4))) }
                        // Lined up with the card's own printing: it is the card, however little of
                        // its outline shows (going out of the picture, steeply tilted, blurred).
                        if m >= 0.85 { sc = max(sc, 0.3 + 2 * Double(m - 0.85)) }
                    }
                    all.append((found[k].quad, found[k].how, sc))
                    if bestIdx < 0 || sc > all[bestIdx].score { bestIdx = k }
                }
                if keepCandidates {
                    var list: [(how: String, score: Double, quad: Quad, note: String)] = []
                    for (k, c) in all.enumerated() {
                        var note = found[k].note
                        if found[k].match > -2 { note += String(format: " m %.2f", found[k].match) }
                        if let ip = indexPattern(paper, c.quad) {
                            note += String(format: " idx %.2f", ip.both) + (ip.cardLike ? "" : " notcard") + (ip.misScaled ? " part" : "")
                        }
                        if let e = entryEvidence(paper, c.quad) { note += " ev\(e)" }
                        let past = (0..<4).filter { printingPast(paper, c.quad, side: $0) == true }
                        if !past.isEmpty { note += " past" + past.map(String.init).joined() }
                        list.append((c.how, c.score, quadCI(c.quad), note))
                    }
                    lastCandidates = list
                }
                if bestIdx >= 0 {
                    best = all[bestIdx]
                    bestNote = found[bestIdx].note
                    bestMatch = found[bestIdx].match
                }
                // Card inside a card: the winner is a box inside another good candidate (round some
                // of the pips, or half the card). The whole card is what must be covered, so the
                // bigger one wins as long as it is clearly card-like too.
                let first = best
                if let b = first, let outer = enclosing(b.quad, in: all, minScore: max(0.3, 0.7 * b.score)) {
                    best = outer
                    bestNote += " (outer)"
                }
                // The same when only the bigger one shows a card's two corner indices: that one is
                // the whole card, even if its outline is harder to see (fingers, glare).
                let second = best
                if let b = second, (indexPattern(paper, b.quad)?.both ?? 1) < 0.06,
                   let outer = enclosingWithIndex(b.quad, in: all, minScore: max(0.15, 0.4 * b.score), paper: paper) {
                    best = outer
                    bestNote += " (outer, index)"
                }
                // The winner, as plain values from here on.
                let winQuad: [CGPoint]? = best?.quad
                let winHow: String = best?.how ?? ""
                let winScore: Double = best?.score ?? -1
                // (the match of the one that won in the end)
                if let wq = winQuad, bestIdx >= 0, wq != all[bestIdx].quad {
                    bestMatch = -2
                    for k in all.indices where all[k].quad == wq {
                        bestMatch = found[k].match
                        break
                    }
                }
                // (+0.25 a card face showing both corner indices, about 0 when one is covered or out
                // of view, below 0 when there is printing where a card has none: a printed sheet, a
                // poster, a box round part of the card)
                let bestPattern = winQuad.flatMap { indexPattern(paper, $0) }
                let bestIndex = bestPattern?.both ?? 0
                var bestDesc = "-"
                if winQuad != nil { bestDesc = winHow + String(format: " %.2f [", winScore) + bestNote + "]" }
                var notCardNote = (bestPattern.map { $0.cardLike } ?? true) ? "" : " notcard"
                if bestPattern.map({ $0.misScaled }) ?? false { notCardNote += " part" }
                if sizeRefs.isEmpty { notCardNote += " new" }
                let matchNote = bestMatch > -2 ? String(format: " m %.2f", bestMatch) : ""
                var debug = "vision \(cands.count) tried \(tried) shapeOK \(shapeOK) best " + bestDesc
                if lockPrinting { debug += " lockpast" }
                debug += String(format: " idx %.2f", bestIndex) + notCardNote + matchNote
                lastDebug = debug
                // The followed outline lines up with the card's picture: it is the card. Nothing takes
                // it over or "refines" it (that is how a lock jumped onto a sheet of paper, or onto a
                // worse fit of the card), except a clearly card-like whole card round it (the picture
                // was then of part of the card).
                let lockVerified = verified && measured != nil
                // Taking over from a working track needs a clearly better, clearly card-like shape.
                // Starting a lock from nothing needs a clearly card-like shape (a weak first lock is how
                // a fake card ends up painted on a wall).
                // The current lock sitting inside a good candidate is the card-in-card glitch: replace it.
                let lockInside = grayQuad.flatMap { enclosing($0, in: all, minScore: max(0.3, 0.7 * trackedScore)) } != nil
                // (the lock shows no corner indices but a bigger candidate round it does)
                var lockPart = false
                // (or its corner index runs on past where it should end: a box round part of the card)
                if let tq = grayQuad, let wq = winQuad, bestIndex >= 0.12, (lockIndex ?? 1) < 0.06 || lockMisScaled,
                   quadArea(wq) > 1.3 * quadArea(tq) {
                    let grown = PlanarTracker.shrunk(wq, -0.03)
                    lockPart = tq.allSatisfy { PlanarTracker.inside(grown, $0) }
                }
                // (or the card carries on past one of the lock's sides, and the winner is bigger, round
                // it, with nothing of the card past its own sides: held close, with corners out of view,
                // the indices that would show it can't be seen)
                if !lockPart, lockPrinting, let tq = grayQuad, let wq = winQuad, quadArea(wq) > 1.08 * quadArea(tq),
                   !(0..<4).contains(where: { printingPast(paper, wq, side: $0) == true }) {
                    let grown = PlanarTracker.shrunk(wq, -0.03)
                    lockPart = tq.allSatisfy { PlanarTracker.inside(grown, $0) }
                }
                // A card showing its two corner indices may start a lock at a lower score (blurred,
                // a little tilted, fingers on it): that pattern is very hard to get by chance. A shape
                // with printing where a card face has none needs a much higher score, and never takes
                // over from a card being followed.
                // (so may a card coming into view with a corner index where the card has one)
                let entering = winHow.hasSuffix(" entry")
                let startScore = bestIndex >= 0.12 ? 0.2 : (bestIndex <= -0.08 ? 0.6 : (entering ? 0.25 : 0.3))
                let mayTakeOver = bestIndex > -0.08 || trackedScore < minScore
                // Something fully in view that isn't a card face (no index in a corner with the
                // corners beside it blank: a printed sheet on the wall, a poster, a sign) never
                // starts a lock or takes one over, unless it is right where the card was just seen.
                // (its printing lining up with the card's picture overrules that)
                // (nor does a box round part of a card)
                let notCardFace = (bestPattern.map { !$0.cardLike || $0.misScaled } ?? false) && bestMatch < 0.9
                let mayStart = !notCardFace || (winQuad.map { nearLastCard($0) } ?? false)
                // The lock drifting off the card (a side slid onto the printing or the palm, or lost
                // track while out of view): a fresh find of the same card that fits it better (a
                // higher score, or the corner indices where they belong) replaces it.
                var refine = false
                if let tq = grayQuad, let wq = winQuad, trackedScore >= 0,
                   alignedDistance(wq, tq) < 0.15 * Double(quadSize(tq)) {
                    refine = winScore >= trackedScore + 0.05
                        || (bestIndex >= 0.12 && (lockIndex ?? 0) < 0.06 && winScore >= 0.7 * trackedScore)
                }
                // (the whole card round the lock: clearly bigger, card-like, not a printed sheet)
                var bigger = false
                if let tq = grayQuad, let wq = winQuad { bigger = quadArea(wq) > 1.3 * quadArea(tq) }
                let wholeRound = lockInside && bigger && winScore >= 0.3 && mayStart
                    && (!lockVerified || (bestIndex >= 0.12 && winScore >= 0.4))
                let corrects = wholeRound || (lockPart && winScore >= 0.15)
                    || (refine && winScore >= 0.2 && mayStart && !lockVerified)
                let starts = measured == nil && mayStart
                // A lock taken on a card coming into view, by where its corner index is: as more of
                // the card comes in, its index boxes show whether it was the right way round. When
                // they come up blank, anything card-like takes over straight away.
                let lockEvidence = grayQuad.flatMap { entryEvidence(paper, $0) }
                let entryWrong = entryLock && measured != nil && (lockEvidence ?? 1) <= 0
                // (the same for any lock partly out of view, once a find's corner index proves the
                // card the other way round: the lock's index boxes in view blank, or with printing
                // past them, the find's inked with the blank paper past it)
                let lockCut = grayQuad.map { FramePipeline.fractionInside(quadCI($0), CGRect(x: 0, y: 0, width: W, height: H)) < 0.95 } ?? false
                let turnedWrong = Tuning.entryByIndex && measured != nil && lockCut && (lockEvidence ?? 1) <= 0
                    && (winQuad.flatMap { entryEvidence(paper, $0) } ?? 0) == 2
                let takesOver = (winScore > trackedScore + 0.12 || entryWrong || turnedWrong) && winScore >= 0.3 && mayTakeOver && mayStart && !lockVerified
                if turnedWrong { lastDebug += " turned" }
                let needed = measured == nil ? startScore : minScore
                if let wq = winQuad, winScore >= needed, starts || takesOver || corrects {
                    entryLock = entering
                    if corrects, let tq = grayQuad {
                        lastWasCorrection = alignedDistance(wq, tq) > 0.04 * Double(quadSize(tq))
                        // (a lock on part of the card had its picture taken: that picture is wrong)
                        if lockVerified && (wholeRound || lockPart) { cardPicture = nil }
                    }
                    grayQuad = wq
                    points = PlanarTracker.goodFeatures(g, quad: wq)
                    measured = quadCI(wq)
                    source = winHow
                    weakFrames = 0
                    trackedScore = winScore
                    snapStreak4 = 0
                    snapStreak3 = 0
                    lastPictureMatch = cardPicture == nil ? -1 : bestMatch
                }
            } else if measured == nil, let first = cands.first {
                measured = first.0
                source = first.1
            }
        }
        lastScore = trackedScore
        // (an entry lock is an ordinary one once the card is all in view, or gone)
        if entryLock, grayQuad.map({ FramePipeline.fractionInside(quadCI($0), CGRect(x: 0, y: 0, width: W, height: H)) >= 0.97 }) ?? true {
            entryLock = false
        }
        if !runDetect { lastDebug = ""; lastCandidates = [] }
        lastEntering = entering && grayQuad == nil
        // (only from a lock that has held for a few frames, and that matches the card's picture once
        // that is known: a wrong lock must not become the size and picture the card is searched for by)
        // (and with nearly all of it in view: its picture is what a fast-moving card is searched for by)
        if trackedScore >= 0.25 || verified, stableFrames >= 4, let gq = grayQuad,
           cardPicture == nil || lastPictureMatch >= 0.8,
           FramePipeline.fractionInside(quadCI(gq), CGRect(x: 0, y: 0, width: W, height: H)) >= 0.9 {
            lastGood = (gq, quadSize(gq), frameIndex)
            lastGoodPyramid = pyramid
        }
        // Take the card's picture once it is clearly the whole card (both corner indices where they
        // belong), all of it in view, hardly moving, and its outline exact: all four sides on the
        // card's edges for a few frames (or three, for longer, when a finger is over the fourth).
        // Every later outline is lined up with this picture, so any error in it would stay.
        let fullyInView = grayQuad.map { q in q.allSatisfy { $0.x > 3 && $0.y > 3 && $0.x < gw - 4 && $0.y < gh - 4 } } ?? false
        // (and the card's proportions right: from an outline a little off, every later outline was
        // lined up that little off)
        let rightShape = grayQuad.flatMap { gq in gray.map { g in
            abs(PlanarTracker.trueAspect(gq, f: PlanarTracker.focalGuess(g), cx: Double(g.w) / 2, cy: Double(g.h) / 2) - 0.716) < 0.03
        } } ?? false
        if cardPicture == nil, fullyInView, rightShape, let gq = grayQuad, let pyr = pyramid, trackedScore >= 0.35, stableFrames >= 6,
           snapStreak4 >= 3 || (snapStreak3 >= 10 && trackedScore >= 0.5), lastMotion < 0.02 * quadSize(gq),
           let ip = indexPattern(paper, gq), ip.both >= 0.12,
           let pic = CardTracker.samplePicture(pyr, gq) {
            cardPicture = pic
            cardPictureFrame = frameIndex
            pictureMismatch = 0
        }
        // Forget it once no card has been seen for two seconds, or once a steady, clearly card-like
        // lock hasn't matched it for a while: the card held up now is a different one.
        if cardPicture != nil, fullyInView, stableFrames >= 10, trackedScore >= 0.35, lastPictureMatch < 0.5,
           (indexPattern(paper, grayQuad!)?.both ?? 0) >= 0.12 {
            pictureMismatch += 1
        } else {
            pictureMismatch = 0
        }
        // (kept for eight seconds after the card was last seen clearly: it is usually the same card
        // brought back, held closer or further, and its picture is what finds it then)
        // (or last followed matching its picture: held close the whole time, it is never all in view)
        if cardPicture != nil, grayQuad != nil, lastPictureMatch >= 0.85 { pictureSeenFrame = frameIndex }
        if cardPicture != nil, pictureMismatch >= 10
            || ((lastGood.map { frameIndex - $0.frame > 240 } ?? true) && frameIndex - cardPictureFrame > 240
                && frameIndex - pictureSeenFrame > 240) {
            cardPicture = nil
            pictureMismatch = 0
        }
        prevPyramid = pyramid

        // 3. Sanity: reject shapes that are clearly not a card held up to the camera.
        if let m = measured, !plausible(m, w: W, h: H) {
            measured = nil
            grayQuad = nil
        }

        if var m = measured {
            // The fake's long side must always lie along the card's long side (never drawn
            // sideways), so only the two labellings with the short sides on top and bottom are
            // allowed; of those, the one matching the last frame (or reading upright) wins.
            let r = PlanarTracker.trueWidthOverHeight(m.points, f: PlanarTracker.focalFraction * Double(min(W, H)),
                                                     cx: Double(W) / 2, cy: Double(H) / 2)
            if r > 1 { m = m.rotatedLabels(1) }
            let flipped = m.rotatedLabels(2)
            if let lq = lastQuad {
                func cost(_ q: Quad) -> CGFloat { zip(q.points, lq.points).reduce(0) { $0 + $1.0.distance(to: $1.1) } }
                if cost(flipped) < cost(m) { m = flipped }
            } else if (m.tl.y + m.tr.y) < (m.bl.y + m.br.y) {
                m = flipped
            }
            // After a gap or a jump to a different place, start fresh at the new position instead of
            // sliding there (a sliding fake card is a visible ghost).
            let jumped = lastQuad.map { meanCornerDistance(m, $0) > 0.25 * max(m.width, m.height) } ?? true
            if framesSinceSeen > 1 || jumped { smoother.reset() }
            lastRawQuad = m
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
        lastRawQuad = nil
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

    /// A card held so close that it runs off two edges of the picture at once: only one corner of it
    /// is in view, and no outline round the part in view can be completed (that needs one whole side
    /// in view). With the card's picture known, that corner and its two sides are enough: the card is
    /// tried at a range of sizes from the corner, each way round, and the size its picture matches
    /// best is lined up with it. `q`: an outline found round the part in view. Gray coordinates.
    private func cornerFits(_ q: [CGPoint], g: GrayImage, pyr: [GrayImage], pic: [Float]) -> [CGPoint]? {
        guard q.count == 4, q.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let side = CGFloat(min(g.w, g.h))
        let margin = 0.03 * side
        func inside(_ p: CGPoint) -> Bool { p.x > margin && p.y > margin && p.x < CGFloat(g.w) - margin && p.y < CGFloat(g.h) - margin }
        // (a card running off the picture: some corner of the outline at its edge)
        let inView = (0..<4).filter { inside(q[$0]) }
        guard !inView.isEmpty, inView.count < 4 else { return nil }
        var best: (m: Float, quad: [CGPoint])?
        for k in inView {
            let c = q[k], pa = q[(k + 3) % 4], pb = q[(k + 1) % 4]
            let la = hypot(pa.x - c.x, pa.y - c.y), lb = hypot(pb.x - c.x, pb.y - c.y)
            guard la > 8, lb > 8 else { continue }
            let ua = CGPoint(x: (pa.x - c.x) / la, y: (pa.y - c.y) / la), ub = CGPoint(x: (pb.x - c.x) / lb, y: (pb.y - c.y) / lb)
            // (two sides at a card's corner: not far from square)
            guard abs(ua.x * ub.x + ua.y * ub.y) < 0.5 else { continue }
            for aLong in [true, false] {
                var s = max(40, 0.15 * side)
                while s <= 1.8 * side {
                    let lenA = aLong ? s / 0.716 : s, lenB = aLong ? s : s / 0.716
                    // (never shorter than the sides seen)
                    if lenA >= 0.85 * la, lenB >= 0.85 * lb {
                        let pA = CGPoint(x: c.x + ua.x * lenA, y: c.y + ua.y * lenA)
                        let pB = CGPoint(x: c.x + ub.x * lenB, y: c.y + ub.y * lenB)
                        let pD = CGPoint(x: pA.x + ub.x * lenB, y: pA.y + ub.y * lenB)
                        var quad = [c, pB, pD, pA]
                        if PlanarTracker.signedArea2(quad) < 0 { quad = [c, pA, pD, pB] }
                        let (m, turned) = CardTracker.bestPictureMatch(pic, pyr, quad)
                        if m > (best?.m ?? -1) { best = (m, turned) }
                    }
                    s *= 1.15
                }
            }
        }
        guard let b = best, b.m >= 0.45,
              let al = CardTracker.alignPicture(pic, pyr, b.quad, iterations: 12), al.match >= 0.75 else { return nil }
        return al.quad
    }

    /// The card found by its picture alone, anywhere in the frame and at any size from a little smaller
    /// than it was last seen to three times that, turned up to 20 degrees either way from how it was
    /// last seen, or upside down: for a card held so close that no outline of it is found (its edges
    /// against a pale wall, fingers along them). The picture is tried coarsely (5 x 7 cells) on the
    /// smallest image, the best few finely (10 x 14) round where they landed, and the best of those are
    /// lined up exactly. nil unless one then matches well. `ref`: the card's outline when last seen
    /// clearly (labelled the way its picture was taken). Gray coordinates.
    private func pictureSearch(_ pic: [Float], _ pyr: [GrayImage], ref: [CGPoint], w fw: CGFloat, h fh: CGFloat) -> [CGPoint]? {
        guard pyr.count >= 2, ref.count == 4, ref.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        let level = pyr.count - 1
        let img = pyr[level], s = CGFloat(1 << level)
        let sums = CardTracker.Integral(img)
        let refW = (hypot(ref[1].x - ref[0].x, ref[1].y - ref[0].y) + hypot(ref[2].x - ref[3].x, ref[2].y - ref[3].y)) / 2
        guard refW > 8 else { return nil }
        let theta0 = atan2(ref[1].y - ref[0].y, ref[1].x - ref[0].x)
        let coarse = CardTracker.coarsePicture(pic, 5, 7), fine = CardTracker.coarsePicture(pic, 10, 14)
        typealias Place = (m: Float, c: CGPoint, S: CGFloat, th: CGFloat, flip: Bool)
        let keepN = 32
        var top: [Place] = []
        func keep(_ p: Place) {
            if top.count < keepN { top.append(p); top.sort { $0.m > $1.m }; return }
            if p.m > top[keepN - 1].m { top[keepN - 1] = p; top.sort { $0.m > $1.m } }
        }
        var S = max(0.8 * refW, 0.2 * min(fw, fh))
        let maxS = min(3.2 * refW, 2.2 * max(fw, fh))
        while S <= maxS {
            let hh = S / 0.716, step = 0.15 * S
            for flip in [false, true] {
                for dk in -2...2 {
                    let th = theta0 + CGFloat(dk) * 10 * .pi / 180
                    let ux = CGVector(dx: cos(th), dy: sin(th)), uy = CGVector(dx: -sin(th), dy: cos(th))
                    var cy = -hh / 6
                    while cy <= fh + hh / 6 {
                        var cx = -S / 6
                        while cx <= fw + S / 6 {
                            let m = CardTracker.placementMatch(coarse, 5, 7, sums, s, CGPoint(x: cx, y: cy), S, ux, uy, flip)
                            if m >= 0.4 { keep((m, CGPoint(x: cx, y: cy), S, th, flip)) }
                            cx += step
                        }
                        cy += step
                    }
                }
            }
            S *= 1.2
        }
        guard !top.isEmpty else { return nil }
        // round each of the best: a little further, bigger or smaller, turned a little, on the finer grid
        var refined: [Place] = []
        for p in top {
            var best: Place = (-2, p.c, p.S, p.th, p.flip)
            for dth in [-5.0, 0.0, 5.0] {
                let th = p.th + CGFloat(dth) * .pi / 180
                let ux = CGVector(dx: cos(th), dy: sin(th)), uy = CGVector(dx: -sin(th), dy: cos(th))
                for ds in [0.92, 1.0, 1.08] {
                    let S2 = p.S * CGFloat(ds)
                    for oy in -1...1 {
                        for ox in -1...1 {
                            let c = CGPoint(x: p.c.x + CGFloat(ox) * 0.075 * p.S, y: p.c.y + CGFloat(oy) * 0.075 * p.S)
                            let m = CardTracker.placementMatch(fine, 10, 14, sums, s, c, S2, ux, uy, p.flip)
                            if m > best.m { best = (m, c, S2, th, p.flip) }
                        }
                    }
                }
            }
            if best.m >= 0.5 { refined.append(best) }
        }
        refined.sort { $0.m > $1.m }
        // line up the best few (not the same place twice) exactly
        var tried: [CGPoint] = []
        var result: (m: Float, quad: [CGPoint])?
        for p in refined where tried.count < 3 {
            if tried.contains(where: { hypot($0.x - p.c.x, $0.y - p.c.y) < 0.1 * p.S }) { continue }
            tried.append(p.c)
            let hh = p.S / 0.716
            let ux = CGVector(dx: cos(p.th), dy: sin(p.th)), uy = CGVector(dx: -sin(p.th), dy: cos(p.th))
            let o = CGPoint(x: p.c.x - 0.5 * p.S * ux.dx - 0.5 * hh * uy.dx, y: p.c.y - 0.5 * p.S * ux.dy - 0.5 * hh * uy.dy)
            let tl = o, tr = CGPoint(x: o.x + p.S * ux.dx, y: o.y + p.S * ux.dy)
            let br = CGPoint(x: tr.x + hh * uy.dx, y: tr.y + hh * uy.dy), bl = CGPoint(x: o.x + hh * uy.dx, y: o.y + hh * uy.dy)
            let quad = p.flip ? [br, bl, tl, tr] : [tl, tr, br, bl]
            if let al = CardTracker.alignPicture(pic, pyr, quad, iterations: 12), al.match >= 0.8, al.match > (result?.m ?? -1) {
                result = (al.match, al.quad)
            }
        }
        return result?.quad
    }

    /// Running sums of an image, for the mean over any upright box in constant time.
    private struct Integral {
        let w: Int, h: Int
        var sum: [Double]
        init(_ g: GrayImage) {
            w = g.w; h = g.h
            sum = [Double](repeating: 0, count: (g.w + 1) * (g.h + 1))
            for y in 0..<g.h {
                var row = 0.0
                for x in 0..<g.w {
                    row += Double(g.px[y * g.w + x])
                    sum[(y + 1) * (w + 1) + x + 1] = sum[y * (w + 1) + x + 1] + row
                }
            }
        }
        /// Mean over the square of half-width `r` round (x, y) (clipped to the image).
        func mean(_ x: CGFloat, _ y: CGFloat, _ r: CGFloat) -> Float {
            let x0 = min(max(Int((x - r).rounded()), 0), w - 1), x1 = max(x0, min(max(Int((x + r).rounded()), 0), w - 1))
            let y0 = min(max(Int((y - r).rounded()), 0), h - 1), y1 = max(y0, min(max(Int((y + r).rounded()), 0), h - 1))
            let W = w + 1
            let t = sum[(y1 + 1) * W + x1 + 1] - sum[y0 * W + x1 + 1] - sum[(y1 + 1) * W + x0] + sum[y0 * W + x0]
            return Float(t / Double((x1 - x0 + 1) * (y1 - y0 + 1)))
        }
    }

    /// The card's picture averaged down to `gw` x `gh` cells over the same part of the card, normalised;
    /// NaN where too little of it is known.
    private static func coarsePicture(_ pic: [Float], _ gw: Int, _ gh: Int) -> [Float] {
        let bx = picW / gw, by = picH / gh
        var out = [Float](repeating: .nan, count: gw * gh)
        var sum: Float = 0, sum2: Float = 0, n = 0
        for j in 0..<gh {
            for i in 0..<gw {
                var t: Float = 0, c = 0
                for y in (j * by)..<((j + 1) * by) {
                    for x in (i * bx)..<((i + 1) * bx) {
                        let v = pic[y * picW + x]
                        if !v.isNaN { t += v; c += 1 }
                    }
                }
                guard c * 2 >= bx * by else { continue }
                let m = t / Float(c)
                out[j * gw + i] = m
                sum += m; sum2 += m * m; n += 1
            }
        }
        guard n >= 4 else { return out }
        let mean = sum / Float(n)
        let sd = max(1e-6, sum2 / Float(n) - mean * mean).squareRoot()
        for k in out.indices where !out[k].isNaN { out[k] = (out[k] - mean) / sd }
        return out
    }

    /// How well the coarse picture `cp` (`gw` x `gh`) matches the image (a pyramid level `s` times
    /// smaller than the gray image, as running sums) for a card centred at `c`, `size` wide along `ux`
    /// and its height along `uy` (gray coordinates), turned half round if `flip`: their correlation
    /// over the cells in view, each cell the image's mean over a box its size. -1 when under a third
    /// of the cells are in view.
    private static func placementMatch(_ cp: [Float], _ gw: Int, _ gh: Int, _ img: Integral, _ s: CGFloat,
                                       _ c: CGPoint, _ size: CGFloat, _ ux: CGVector, _ uy: CGVector, _ flip: Bool) -> Float {
        let hh = size / 0.716
        let ox = c.x - 0.5 * size * ux.dx - 0.5 * hh * uy.dx, oy = c.y - 0.5 * size * ux.dy - 0.5 * hh * uy.dy
        let r = 0.45 * size / CGFloat(gw) / s
        let xmax = CGFloat(img.w - 2), ymax = CGFloat(img.h - 2)
        var sa: Float = 0, sb: Float = 0, saa: Float = 0, sbb: Float = 0, sab: Float = 0, n: Float = 0
        var total = 0
        for j in 0..<gh {
            for i in 0..<gw {
                let a = cp[j * gw + i]
                if a.isNaN { continue }
                total += 1
                var u = 0.05 + 0.9 * (CGFloat(i) + 0.5) / CGFloat(gw), v = 0.05 + 0.9 * (CGFloat(j) + 0.5) / CGFloat(gh)
                if flip { u = 1 - u; v = 1 - v }
                let x = (ox + u * size * ux.dx + v * hh * uy.dx) / s, y = (oy + u * size * ux.dy + v * hh * uy.dy) / s
                guard x >= 1, y >= 1, x <= xmax, y <= ymax else { continue }
                let b = img.mean(x, y, r)
                sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b; n += 1
            }
        }
        guard total > 0, n * 3 >= Float(total), n >= 4 else { return -1 }
        let va = saa - sa * sa / n, vb = sbb - sb * sb / n
        guard va > 1e-6, vb > 1e-6 else { return -1 }
        return (sab - sa * sb / n) / (va * vb).squareRoot()
    }

    /// Every side of the outline with the card carrying on past it (see `printingPast`) moved out to
    /// where the card's paper ends, crossing the corner index and pip on the way, a fifth of the
    /// card at most. nil when no side moved. Gray coordinates.
    private func pushOutPastPrinting(_ g: GrayImage, _ paper: PaperReader?, _ quad: [CGPoint]) -> [CGPoint]? {
        guard quad.count == 4 else { return nil }
        var q = quad
        var moved = false
        for k in 0..<4 where printingPast(paper, q, side: k) == true {
            let a = q[k], p1 = q[(k + 1) % 4], p2 = q[(k + 2) % 4], p3 = q[(k + 3) % 4]
            let across = Double(hypot(p2.x - p1.x, p2.y - p1.y) + hypot(a.x - p3.x, a.y - p3.y)) / 2
            guard across > 16 else { continue }
            if let out = PlanarTracker.pushSideOut(g, quad: q, side: k, reach: 0.22 * across, throughGap: max(3, 0.1 * across)) {
                q = out
                moved = true
            }
        }
        return moved ? q : nil
    }

    /// Side k of the outline moved out onto the card's edge when it is too far in (see flowStep):
    /// only if the card's proportions say so, and come out right afterwards. Gray coordinates.
    private func pushOut(_ g: GrayImage, _ quad: [CGPoint], side k: Int) -> [CGPoint]? {
        guard quad.count == 4 else { return nil }
        func aspect(_ q: [CGPoint]) -> Double {
            PlanarTracker.trueAspect(q, f: PlanarTracker.focalGuess(g), cx: Double(g.w) / 2, cy: Double(g.h) / 2)
        }
        func len(_ i: Int) -> Double { Double(hypot(quad[(i + 1) % 4].x - quad[i].x, quad[(i + 1) % 4].y - quad[i].y)) }
        // (the side must be in view: its middle inside the picture)
        let mid = CGPoint(x: (quad[k].x + quad[(k + 1) % 4].x) / 2, y: (quad[k].y + quad[(k + 1) % 4].y) / 2)
        guard mid.x > 4, mid.y > 4, mid.x < CGFloat(g.w - 5), mid.y < CGFloat(g.h - 5) else { return nil }
        let r = aspect(quad)
        let longSide = (len(k) + len((k + 2) % 4)) > (len((k + 1) % 4) + len((k + 3) % 4))
        // a long side too far in makes the card too narrow; a short side, too short
        guard longSide ? r < 0.716 - 0.035 : r > 0.716 + 0.035 else { return nil }
        let across = (len((k + 1) % 4) + len((k + 3) % 4)) / 2      // the card's size across this side
        guard let out = PlanarTracker.pushSideOut(g, quad: quad, side: k, reach: 0.2 * across) else { return nil }
        guard abs(aspect(out) - 0.716) < 0.03 else { return nil }
        return out
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
        let m = fit.0
        var inliers = fit.1
        var nq = quad.map { m.apply($0) }
        // A card tilting toward or away from the camera changes perspective, which turn + scale +
        // shift can't follow: small per frame, but it adds up and the fake slides off the far edge.
        // Follow the card with a perspective map whenever there are enough points for a solid one
        // and it agrees with the simple fit to within a few percent (so a bad fit can't throw it).
        if a.count >= 15, let hf = PlanarTracker.ransacHomography(a, b) {
            let hq = quad.map { hf.0.apply($0) }
            let simCount = inliers.filter { $0 }.count, hCount = hf.1.filter { $0 }.count
            let size = (0..<4).map { k in hypot(nq[(k + 1) % 4].x - nq[k].x, nq[(k + 1) % 4].y - nq[k].y) }.max() ?? 0
            let dev = zip(hq, nq).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? .infinity
            // (only from points spread over the card: from a cluster in the middle, the perspective
            // part is a guess that swings the far corners about)
            let used = zip(b, hf.1).filter { $0.1 }.map { $0.0 }
            func extent(_ v: [CGFloat]) -> CGFloat { (v.max() ?? 0) - (v.min() ?? 0) }
            let spreadX: CGFloat = extent(used.map { $0.x }) / max(1, extent(quad.map { $0.x }))
            let spreadY: CGFloat = extent(used.map { $0.y }) / max(1, extent(quad.map { $0.y }))
            let spread = min(spreadX, spreadY)
            if hq.allSatisfy({ $0.x.isFinite && $0.y.isFinite }), hCount >= simCount, hCount >= 12,
               dev < 0.06 * size, spread > 0.45 {
                nq = hq
                inliers = hf.1
            }
        }
        // Nudge the sides onto the card's edges (removes slow drift); a side never jumps further
        // than this in one frame, so it can't be thrown onto some other line.
        var snapped = 0
        let side = (0..<4).map { k in hypot(nq[(k + 1) % 4].x - nq[k].x, nq[(k + 1) % 4].y - nq[k].y) }.max() ?? 0
        let flowQuad = nq
        let (sq, n, bySide) = PlanarTracker.snapEdgesBySide(cur[0], quad: nq, maxShift: max(2, 0.04 * Double(side)))
        if n > 0 { nq = sq; snapped = n }
        // One side not on an edge and the card's proportions off the way that side being too far in
        // would make them: it is stuck on the card's paper (a corner index or a thumb kept it there
        // when the card was found), the real edge further out. Move it out onto the edge, if that
        // puts the proportions right. (Before the card's picture is known: after, the printing
        // places every side.)
        if snapped == 3, cardPicture == nil, let k = bySide.firstIndex(of: false), let fixed = pushOut(cur[0], nq, side: k) {
            nq = fixed
            snapped = 4
        }
        snapStreak4 = snapped == 4 ? snapStreak4 + 1 : 0
        snapStreak3 = snapped >= 3 ? snapStreak3 + 1 : 0
        lastMotion = zip(nq, quad).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? .infinity
        // Then line the card's printing up with its picture (see "The card's picture").
        lastPictureMatch = -1
        var aligned: (quad: [CGPoint], match: Float)? = nil
        if let pic = cardPicture {
            let fromEdges = CardTracker.alignPicture(pic, cur, nq)
            let firstMatch: Float = fromEdges?.match ?? -1
            aligned = fromEdges
            // Not quite lined up from the edges (a side may have slid onto a line printed on the
            // card, steeply tilted): try again from where the printing alone said the card went.
            if firstMatch < 0.95, snapped > 0,
               let again = CardTracker.alignPicture(pic, cur, flowQuad, iterations: 12), again.match > firstMatch {
                aligned = again
            }
        }
        if let al = aligned {
            let shift = zip(al.quad, nq).map { hypot($0.x - $1.x, $0.y - $1.y) }.max() ?? .infinity
            // (lined up only by moving far from where the card was followed to: not this outline's match)
            lastPictureMatch = shift < 0.12 * side ? al.match : -1
            if snapped == 4 && shift <= 1.5 {
                // All four sides on the card's edges and the printing agrees: the edges are the
                // sharper measurement, keep them. And now and then retake the picture from them (the
                // light changes as the card moves; any small error in the first picture goes).
                let w = CGFloat(cur[0].w), h = CGFloat(cur[0].h)
                if al.match >= 0.85, snapStreak4 >= 3, lastMotion < 0.02 * side, frameIndex - cardPictureFrame >= 10,
                   nq.allSatisfy({ $0.x > 3 && $0.y > 3 && $0.x < w - 4 && $0.y < h - 4 }),
                   let fresh = CardTracker.samplePicture(cur, nq) {
                    cardPicture = fresh
                    cardPictureFrame = frameIndex
                }
            } else if snapped == 4, snapStreak4 >= 5, lastMotion < 0.02 * side, shift < 0.04 * side, al.match >= 0.75,
                      frameIndex - cardPictureFrame >= 5,
                      abs(PlanarTracker.trueAspect(nq, f: PlanarTracker.focalGuess(cur[0]), cx: Double(cur[0].w) / 2, cy: Double(cur[0].h) / 2) - 0.716) < 0.03 {
                // All four sides on the card's edges for a while, the card's proportions right and held
                // nearly still, but its printing a few pixels off them: the picture was taken from an
                // outline a little off, and every outline lined up with it since was that little off.
                // The edges are right: keep them, and take the picture again from them.
                let w = CGFloat(cur[0].w), h = CGFloat(cur[0].h)
                if nq.allSatisfy({ $0.x > 3 && $0.y > 3 && $0.x < w - 4 && $0.y < h - 4 }),
                   let fresh = CardTracker.samplePicture(cur, nq) {
                    cardPicture = fresh
                    cardPictureFrame = frameIndex
                }
            } else if al.match >= 0.6, shift < 0.12 * side {
                // A side slid onto the printing, a finger or the palm, or an edge is covered or out of
                // the picture: the printing decides where all four corners are.
                nq = al.quad
                // Then the sides that are in view go exactly onto the card's own edges, a short way
                // only. (The printing alone can be a little off in scale, most of all held close with
                // part of the card out of the picture, and the plain border has nothing to line up:
                // the long edge then sat inside the real one and a strip of the real card showed.)
                let (es, en) = PlanarTracker.snapEdges(cur[0], quad: nq, maxShift: max(1.5, 0.015 * Double(side)), fine: true)
                if en > 0 { nq = es }
            }
        }
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

        // (the template is sampled on at most a 24 x 24 grid and the search radius is capped, so the
        // cost stays the same however big the card is: uncapped, a card held close to the camera
        // took seconds per frame here)
        func ncc(_ T: GrayImage, _ C: GrayImage, _ x0: Int, _ y0: Int, _ x1: Int, _ y1: Int, _ dx: Int, _ dy: Int) -> Float {
            var st: Float = 0, sc: Float = 0, stt: Float = 0, scc: Float = 0, stc: Float = 0
            var n: Float = 0
            let sx = max(1, (x1 - x0) / 24), sy = max(1, (y1 - y0) / 24)
            for y in stride(from: y0, through: y1, by: sy) {
                for x in stride(from: x0, through: x1, by: sx) {
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
        // (out to about a card's width and a half: a card swiped out and back comes back fast)
        let r3 = min(28, max(3, Int((size / 5).rounded())))
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
        /// (r, g, b) 0...255 at integer gray-image coordinates (no bounds surprises), or nil outside.
        func colorAt(_ x: Int, _ y: Int) -> (UInt8, UInt8, UInt8)? {
            let yy = topDown ? y : h - 1 - y
            guard x >= 0, x < w, yy >= 0, yy < h else { return nil }
            let o = yy * rb + x * 4
            return (base[o + 2], base[o + 1], base[o])
        }

        /// (r, g, b) 0...255 at gray-image coordinates, or nil outside the picture.
        func color(_ x: CGFloat, _ y: CGFloat) -> (Double, Double, Double)? {
            guard x.isFinite, y.isFinite, abs(x) < 1e6, abs(y) < 1e6 else { return nil }
            let xi = Int(x.rounded())
            var yi = Int(y.rounded())
            if !topDown { yi = h - 1 - yi }
            guard xi >= 0, xi < w, yi >= 0, yi < h else { return nil }
            let o = yi * rb + xi * 4
            return (Double(base[o + 2]), Double(base[o + 1]), Double(base[o]))
        }

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
    private var partsNote = ""

    private func cardLikeness(_ g: GrayImage, _ paper: PaperReader?, _ quad: [CGPoint]) -> Double {
        let s = PlanarTracker.cardScore(g, quad: quad)
        partsNote = "edge/shape 0"
        guard s > 0 else { return 0 }
        guard let paper = paper else { return s }
        let (p, seen) = paperFraction(paper, quad)
        // A card face always has printing on it (indices, pips); a blank patch of wall or ceiling
        // with straight edges around it does not.
        let ink = inkFraction(paper, quad)
        partsNote = String(format: "edge/shape %.2f paper %.2f ink %.2f seen %.2f", s, p, ink, seen)
        // (a mild preference for the outline with less of it out of view: of two ways of completing
        // a card cut off by the edge of the picture, the one that reaches less far out is right far
        // more often, and the picture alone can't tell them apart)
        return s * p * p * min(1, ink / 0.04) * (0.6 + 0.4 * seen)
    }

    /// Fraction of points inside the quad (its middle 76%) that are printing: clearly darker than
    /// the paper, or strongly red. Only points inside the picture count.
    private func inkFraction(_ paper: PaperReader, _ quad: [CGPoint]) -> Double {
        guard quad.count == 4 else { return 0 }
        func lum(_ c: (Double, Double, Double)) -> Double { 0.299 * c.0 + 0.587 * c.1 + 0.114 * c.2 }
        var samples: [(Double, Double, Double)] = []
        for j in 0..<16 {
            for i in 0..<12 {
                let u = 0.12 + 0.76 * CGFloat(i) / 11, v = 0.12 + 0.76 * CGFloat(j) / 15
                // bilinear position inside the quad (tl, tr, br, bl)
                let top = CGPoint(x: quad[0].x + u * (quad[1].x - quad[0].x), y: quad[0].y + u * (quad[1].y - quad[0].y))
                let bot = CGPoint(x: quad[3].x + u * (quad[2].x - quad[3].x), y: quad[3].y + u * (quad[2].y - quad[3].y))
                let p = CGPoint(x: top.x + v * (bot.x - top.x), y: top.y + v * (bot.y - top.y))
                if let c = paper.color(p.x, p.y) { samples.append(c) }
            }
        }
        guard samples.count >= 20 else { return 0 }
        let lums = samples.map(lum).sorted()
        let paperLum = lums[Int(Double(lums.count - 1) * 0.8)]     // the paper is the bright majority
        guard paperLum > 40 else { return 0 }
        let inky = samples.filter { c in
            lum(c) < 0.6 * paperLum || (c.0 - max(c.1, c.2) > 60)
        }.count
        return Double(inky) / Double(samples.count)
    }

    /// A playing card carries its index (rank and suit) in two opposite corners and nothing in the
    /// other two (seen mirrored by the selfie camera it is the other two). How clearly the quad
    /// shows that: the ink in the emptier of the two index corners minus the ink in the fuller of
    /// the other two. About 0.25-0.3 for the whole card; about 0 for half a card, a corner of it,
    /// the box round a court card's picture or most other rectangles; nil when a corner is out of
    /// view. (a thumb over a corner hides it too, so this only ever decides between candidates)
    /// `cardLike`: both indices in two opposite corners and the other two corners clearly emptier
    /// (true of a card face unless a thumb covers an index; not of a printed sheet or poster).
    /// `misScaled`: printing in an index corner carries on past where a card's index ends (a card
    /// has blank paper there, beside its pips or picture): the quad is smaller than the card it is
    /// on, a box round a corner of it or half of it.
    private func indexPattern(_ paper: PaperReader?, _ quad: [CGPoint]) -> (both: Double, cardLike: Bool, misScaled: Bool)? {
        guard let paper = paper, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        // label it so u runs across the card's short side
        var q = quad
        let s01 = hypot(q[1].x - q[0].x, q[1].y - q[0].y) + hypot(q[2].x - q[3].x, q[2].y - q[3].y)
        let s03 = hypot(q[3].x - q[0].x, q[3].y - q[0].y) + hypot(q[2].x - q[1].x, q[2].y - q[1].y)
        if s01 > s03 { q = [q[1], q[2], q[3], q[0]] }
        let unit = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]
        guard let H = PlanarTracker.fitHomography(unit, q) else { return nil }
        func lum(_ c: (Double, Double, Double)) -> Double { 0.299 * c.0 + 0.587 * c.1 + 0.114 * c.2 }
        func at(_ u: CGFloat, _ v: CGFloat) -> (Double, Double, Double)? {
            let p = H.apply(CGPoint(x: u, y: v))
            guard p.x >= 0, p.y >= 0, p.x <= CGFloat(paper.w - 1), p.y <= CGFloat(paper.h - 1) else { return nil }
            return paper.color(p.x, p.y)
        }
        var lums: [Double] = []
        for j in 0..<12 { for i in 0..<8 {
            if let c = at(0.12 + 0.76 * CGFloat(i) / 7, 0.12 + 0.76 * CGFloat(j) / 11) { lums.append(lum(c)) }
        } }
        guard lums.count >= 60 else { return nil }
        lums.sort()
        let paperLum = lums[Int(Double(lums.count - 1) * 0.8)]
        guard paperLum > 40 else { return nil }
        // printing: red ink, or dark and not warm (a fingertip in shadow is dark but warm)
        func isInk(_ c: (Double, Double, Double)) -> Bool {
            if c.0 - max(c.1, c.2) > 60 { return true }
            guard lum(c) < 0.6 * paperLum else { return false }
            return !(c.0 - c.2 > 25 && c.1 > c.2 + 5)
        }
        // the index boxes of the performer's deck: 4-13% across, 6-24% down from each corner
        func inkIn(_ u0: CGFloat, _ v0: CGFloat, _ dv: CGFloat = 0.18) -> Double? {
            var n = 0, ink = 0
            for j in 0..<10 { for i in 0..<6 {
                let u = u0 + 0.085 * CGFloat(i) / 5, v = v0 + dv * CGFloat(j) / 9
                guard let c = at(u, v) else { continue }
                n += 1
                if isInk(c) { ink += 1 }
            } }
            return n >= 50 ? Double(ink) / Double(n) : nil
        }
        guard let tl = inkIn(0.045, 0.06), let tr = inkIn(0.87, 0.06),
              let bl = inkIn(0.045, 0.76), let br = inkIn(0.87, 0.76) else { return nil }
        let both = max(min(tl, br) - max(tr, bl), min(tr, bl) - max(tl, br))
        // Just past each index along the card (29-40% down: blank on a card face). Printing there
        // in a corner whose index box is inked, and no corner with a clean index, is a box round part
        // of a card: its "index" is a piece of a bigger index, of a pip or of a court card's picture.
        let marks: [Double] = [tl, tr, bl, br]
        let past: [Double?] = [inkIn(0.045, 0.29, 0.11), inkIn(0.87, 0.29, 0.11), inkIn(0.045, 0.60, 0.11), inkIn(0.87, 0.60, 0.11)]
        var clean = false, crowded = false
        for k in 0..<4 where marks[k] >= 0.12 {
            guard let p = past[k] else { continue }
            if p <= 0.06 { clean = true }
            if p >= 0.2 { crowded = true }
        }
        // (both indices: a white sheet with one dark mark in a corner looks like a card with a thumb
        // over one index, so only the full pattern counts as proof)
        return (both, both >= 0.1, crowded && !clean)
    }

    /// A card coming into view, completed from the part of it in the picture (one way round or the
    /// other, see `completions`): does the part in view show a corner index where this completion
    /// puts one? A card's index is long and thin along the card, so the wrong way round puts the index
    /// boxes beside the real index, on blank paper. true: an index box in view is inked and the paper
    /// just past it along the card is in view and blank; false: the index boxes in view are blank, or
    /// inked with printing carrying on past them; nil: no index box in view yet.
    private func entryIndex(_ paper: PaperReader?, _ quad: [CGPoint]) -> Bool? {
        entryEvidence(paper, quad).map { $0 == 2 }
    }

    /// `entryIndex` in more detail: 2 an index box in view is inked with the blank paper past it in
    /// view too; 1 an inked index box, the paper past it not in view yet; 0 the index boxes in view
    /// are blank; -1 printing carries on past an inked box (or the box is solid dark); nil no index
    /// box in view yet.
    private func entryEvidence(_ paper: PaperReader?, _ quad: [CGPoint]) -> Int? {
        guard let paper = paper, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        var q = quad
        let s01 = hypot(q[1].x - q[0].x, q[1].y - q[0].y) + hypot(q[2].x - q[3].x, q[2].y - q[3].y)
        let s03 = hypot(q[3].x - q[0].x, q[3].y - q[0].y) + hypot(q[2].x - q[1].x, q[2].y - q[1].y)
        if s01 > s03 { q = [q[1], q[2], q[3], q[0]] }
        let unit = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]
        guard let H = PlanarTracker.fitHomography(unit, q) else { return nil }
        func lum(_ c: (Double, Double, Double)) -> Double { 0.299 * c.0 + 0.587 * c.1 + 0.114 * c.2 }
        func at(_ u: CGFloat, _ v: CGFloat) -> (Double, Double, Double)? {
            let p = H.apply(CGPoint(x: u, y: v))
            guard p.x >= 0, p.y >= 0, p.x <= CGFloat(paper.w - 1), p.y <= CGFloat(paper.h - 1) else { return nil }
            return paper.color(p.x, p.y)
        }
        // the paper's brightness, from the part in view
        var lums: [Double] = []
        for j in 0..<16 { for i in 0..<12 {
            if let c = at(0.03 + 0.94 * CGFloat(i) / 11, 0.03 + 0.94 * CGFloat(j) / 15) { lums.append(lum(c)) }
        } }
        guard lums.count >= 24 else { return nil }
        lums.sort()
        let paperLum = lums[Int(Double(lums.count - 1) * 0.8)]
        guard paperLum > 40 else { return nil }
        func isInk(_ c: (Double, Double, Double)) -> Bool {
            if c.0 - max(c.1, c.2) > 60 { return true }
            guard lum(c) < 0.6 * paperLum else { return false }
            return !(c.0 - c.2 > 25 && c.1 > c.2 + 5)
        }
        // (the same boxes as `indexPattern`, but only boxes wholly in view count)
        func inkIn(_ u0: CGFloat, _ v0: CGFloat, _ dv: CGFloat = 0.18) -> Double? {
            var n = 0, ink = 0
            for j in 0..<10 { for i in 0..<6 {
                let u = u0 + 0.085 * CGFloat(i) / 5, v = v0 + dv * CGFloat(j) / 9
                guard let c = at(u, v) else { continue }
                n += 1
                if isInk(c) { ink += 1 }
            } }
            return n >= 57 ? Double(ink) / Double(n) : nil
        }
        let boxes: [(CGFloat, CGFloat)] = [(0.045, 0.06), (0.87, 0.06), (0.045, 0.76), (0.87, 0.76)]
        let pasts: [(CGFloat, CGFloat)] = [(0.045, 0.29), (0.87, 0.29), (0.045, 0.60), (0.87, 0.60)]
        var tested = 0, strong = 0, weak = 0, crowded = false
        for k in 0..<4 {
            guard let m = inkIn(boxes[k].0, boxes[k].1) else { continue }
            tested += 1
            guard m >= 0.12 else { continue }
            // (solid dark is not an index: hair or a dark sleeve where the box is off the card)
            if m > 0.65 { crowded = true; continue }
            // (an index with blank paper just past it is the proof; a box that lands on a pip or a
            // piece of the court picture, the wrong way round, has printing there)
            if let p = inkIn(pasts[k].0, pasts[k].1, 0.11) {
                if p >= 0.2 { crowded = true } else if p <= 0.06 { strong += 1 } else { weak += 1 }
            } else {
                weak += 1
            }
        }
        guard tested > 0 else { return nil }
        if crowded { return -1 }
        return strong > 0 ? 2 : (weak > 0 ? 1 : 0)
    }

    /// Does the card carry on past side k of `quad` (from corner k to k+1): its paper, with printing on
    /// it, just outside the side? Then the side is not the card's edge but a line inside the card: a
    /// court card's printed frame round its picture, with the corner index and pip outside it (taken
    /// for the card's edge, the fake was drawn short of it and the real corner index showed beside
    /// it), or a side across the middle of the card. A real edge has the background past it: a hand,
    /// a wall (no printing), anything but the card's paper with printing on it. nil when too little
    /// of the strip past the side is in view.
    private func printingPast(_ paper: PaperReader?, _ quad: [CGPoint], side k: Int) -> Bool? {
        guard let paper = paper, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let a = quad[k], b = quad[(k + 1) % 4]
        let L = hypot(b.x - a.x, b.y - a.y)
        guard L > 8 else { return nil }
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        let d = CGVector(dx: (b.x - a.x) / L, dy: (b.y - a.y) / L)
        var n = CGVector(dx: -d.dy, dy: d.dx)
        if n.dx * ((a.x + b.x) / 2 - cx) + n.dy * ((a.y + b.y) / 2 - cy) < 0 { n = CGVector(dx: -n.dx, dy: -n.dy) }
        // the card's size across this side: how far it is to the opposite side
        let p1 = quad[(k + 1) % 4], p2 = quad[(k + 2) % 4], p3 = quad[(k + 3) % 4]
        let across = (hypot(p2.x - p1.x, p2.y - p1.y) + hypot(a.x - p3.x, a.y - p3.y)) / 2
        guard across > 8 else { return nil }
        func lum(_ c: (Double, Double, Double)) -> Double { 0.299 * c.0 + 0.587 * c.1 + 0.114 * c.2 }
        func sat(_ c: (Double, Double, Double)) -> Double {
            let mx = max(c.0, c.1, c.2), mn = min(c.0, c.1, c.2)
            return mx > 0 ? (mx - mn) / mx : 0
        }
        // the card's paper: a strip just inside the side
        var inside: [Double] = []
        for s in 0..<12 {
            let t = 0.2 + 0.6 * CGFloat(s) / 11
            let x = a.x + t * (b.x - a.x) - 0.05 * across * n.dx, y = a.y + t * (b.y - a.y) - 0.05 * across * n.dy
            if let c = paper.color(x, y) { inside.append(lum(c)) }
        }
        guard inside.count >= 6 else { return nil }
        inside.sort()
        let paperLum = inside[Int(Double(inside.count - 1) * 0.8)]
        guard paperLum > 60 else { return nil }
        // past the side: 2 to 10% of the card across, along the middle of the side
        var seen = 0, paperish = 0, ink = 0
        for j in 0..<6 {
            for s in 0..<14 {
                let off = (0.02 + 0.08 * CGFloat(j) / 5) * across
                let t = 0.15 + 0.7 * CGFloat(s) / 13
                let x = a.x + t * (b.x - a.x) + off * n.dx, y = a.y + t * (b.y - a.y) + off * n.dy
                guard let c = paper.color(x, y) else { continue }
                seen += 1
                let l = lum(c)
                if l > 0.8 * paperLum && l < 1.25 * paperLum && sat(c) < 0.3 {
                    paperish += 1
                } else if c.0 - max(c.1, c.2) > 60 || (l < 0.6 * paperLum && !(c.0 - c.2 > 25 && c.1 > c.2 + 5)) {
                    ink += 1       // (printing: red, or dark and not warm like a fingertip in shadow)
                }
            }
        }
        guard seen >= 30 else { return nil }
        return Double(paperish + ink) / Double(seen) >= 0.8 && Double(ink) / Double(seen) >= 0.04
    }

    // MARK: - The card's picture

    /// The card's printing as seen face-on: a grid of gray values (zero mean, unit spread, NaN where
    /// out of view) over the card's face, taken once the card is clearly and fully in view. Lining
    /// the card's picture up with it every frame pins all four corners to the printing itself, so
    /// the outline can't slide off at a steep angle, close up, or onto a finger; and an outline that
    /// isn't the whole card (half of it, the frame round a court card's picture) or isn't this card
    /// at all doesn't match it.
    static let picW = 40, picH = 56
    private static let unitSquare = [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)]

    /// Where on the card each grid cell is (the middle 90% of the face).
    private static func picUV(_ i: Int, _ j: Int) -> CGPoint {
        CGPoint(x: 0.05 + 0.9 * (CGFloat(i) + 0.5) / CGFloat(picW), y: 0.05 + 0.9 * (CGFloat(j) + 0.5) / CGFloat(picH))
    }

    /// The pyramid level at which a card this big covers the grid about one to two pixels per cell.
    private static func pictureLevel(_ quad: [CGPoint], levels: Int) -> Int {
        let w = (hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y) + hypot(quad[2].x - quad[3].x, quad[2].y - quad[3].y)) / 2
        let h = (hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y) + hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y)) / 2
        let short = min(w, h)
        var L = 0
        while L + 1 < levels, short / CGFloat(1 << (L + 1)) >= CGFloat(picW) * 1.2 { L += 1 }
        return L
    }

    /// The card's picture inside `quad` (gray coordinates, labelled tl, tr, br, bl), normalised;
    /// nil when too little of it is in view or it is blank.
    static func samplePicture(_ pyr: [GrayImage], _ quad: [CGPoint]) -> [Float]? {
        guard !pyr.isEmpty, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let L = pictureLevel(quad, levels: pyr.count)
        let s = CGFloat(1 << L)
        let img = pyr[L]
        guard let H = PlanarTracker.fitHomography(unitSquare, quad.map { CGPoint(x: $0.x / s, y: $0.y / s) }) else { return nil }
        var out = [Float](repeating: .nan, count: picW * picH)
        var sum: Float = 0, sum2: Float = 0, n = 0
        for j in 0..<picH {
            for i in 0..<picW {
                let p = H.apply(picUV(i, j))
                guard img.contains(Double(p.x), Double(p.y)) else { continue }
                let v = img.sample(Float(p.x), Float(p.y))
                out[j * picW + i] = v
                sum += v; sum2 += v * v; n += 1
            }
        }
        guard n >= picW * picH / 3 else { return nil }
        let mean = sum / Float(n)
        let sd = (max(1e-8, sum2 / Float(n) - mean * mean)).squareRoot()
        guard sd > 0.01 else { return nil }
        for k in out.indices where !out[k].isNaN { out[k] = (out[k] - mean) / sd }
        return out
    }

    /// Correlation (-1...1) of two pictures over the cells seen in both.
    static func pictureMatch(_ a: [Float], _ b: [Float]) -> Float {
        var sa: Float = 0, sb: Float = 0, saa: Float = 0, sbb: Float = 0, sab: Float = 0, n: Float = 0
        for k in a.indices where !a[k].isNaN && !b[k].isNaN {
            sa += a[k]; sb += b[k]; saa += a[k] * a[k]; sbb += b[k] * b[k]; sab += a[k] * b[k]; n += 1
        }
        guard n >= Float(picW * picH / 3) else { return -1 }
        let va = saa - sa * sa / n, vb = sbb - sb * sb / n
        guard va > 1e-6, vb > 1e-6 else { return -1 }
        return (sab - sa * sb / n) / (va * vb).squareRoot()
    }

    /// The best match of the outline's picture with the card's, trying the outline each way round;
    /// returns the match and the outline labelled the way that matched.
    static func bestPictureMatch(_ pic: [Float], _ pyr: [GrayImage], _ quad: [CGPoint]) -> (Float, [CGPoint]) {
        var best: (Float, [CGPoint]) = (-1, quad)
        for k in 0..<4 {
            let q = Array(quad[k...] + quad[..<k])
            if let p = samplePicture(pyr, q) {
                let m = pictureMatch(pic, p)
                if m > best.0 { best = (m, q) }
            }
        }
        return best
    }

    /// Moves the four corners so the card's printing lines up with its picture: Gauss-Newton on the
    /// eight corner coordinates, with robust weights so fingers over part of the card don't pull it.
    /// Returns the new outline and how well it then matches.
    static func alignPicture(_ pic: [Float], _ pyr: [GrayImage], _ quad: [CGPoint],
                             iterations: Int = 8) -> (quad: [CGPoint], match: Float)? {
        guard !pyr.isEmpty, quad.count == 4,
              quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return nil }
        let L = pictureLevel(quad, levels: pyr.count)
        let s = CGFloat(1 << L)
        let img = pyr[L]
        var c = quad.map { CGPoint(x: $0.x / s, y: $0.y / s) }
        var uv: [CGPoint] = [], T: [Float] = []
        for j in 0..<picH {
            for i in 0..<picW where !pic[j * picW + i].isNaN {
                uv.append(picUV(i, j))
                T.append(pic[j * picW + i])
            }
        }
        guard uv.count >= picW * picH / 3 else { return nil }
        let d: CGFloat = 0.5
        for _ in 0..<max(1, iterations) {
            guard let H = PlanarTracker.fitHomography(unitSquare, c) else { return nil }
            // how each grid point moves when each corner coordinate moves by d
            var Hd: [PlanarTracker.Homography] = []
            for m in 0..<8 {
                var cc = c
                if m % 2 == 0 { cc[m / 2].x += d } else { cc[m / 2].y += d }
                guard let h = PlanarTracker.fitHomography(unitSquare, cc) else { return nil }
                Hd.append(h)
            }
            var I = [Float](repeating: .nan, count: uv.count)
            var gx = I, gy = I
            var pts = [CGPoint](repeating: .zero, count: uv.count)
            var sum: Float = 0, sum2: Float = 0, n = 0
            for k in uv.indices {
                let p = H.apply(uv[k])
                pts[k] = p
                guard img.contains(Double(p.x) - 1, Double(p.y) - 1), img.contains(Double(p.x) + 1, Double(p.y) + 1) else { continue }
                let x = Float(p.x), y = Float(p.y)
                let v = img.sample(x, y)
                I[k] = v
                gx[k] = 0.5 * (img.sample(x + 1, y) - img.sample(x - 1, y))
                gy[k] = 0.5 * (img.sample(x, y + 1) - img.sample(x, y - 1))
                sum += v; sum2 += v * v; n += 1
            }
            guard n >= picW * picH / 4 else { return nil }
            let mean = sum / Float(n)
            let sd = (max(1e-8, sum2 / Float(n) - mean * mean)).squareRoot()
            guard sd > 0.01 else { return nil }
            var A = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
            var bvec = [Double](repeating: 0, count: 8)
            for k in uv.indices where !I[k].isNaN {
                let r = Double((I[k] - mean) / sd - T[k])
                let w = abs(r) <= 1 ? 1.0 : 1 / abs(r)
                var J = [Double](repeating: 0, count: 8)
                for m in 0..<8 {
                    let q = Hd[m].apply(uv[k])
                    let dx = Double(q.x - pts[k].x) / Double(d), dy = Double(q.y - pts[k].y) / Double(d)
                    J[m] = (Double(gx[k]) * dx + Double(gy[k]) * dy) / Double(sd)
                }
                for a in 0..<8 {
                    bvec[a] -= w * J[a] * r
                    for e in 0..<8 { A[a][e] += w * J[a] * J[e] }
                }
            }
            for a in 0..<8 { A[a][a] = A[a][a] * 1.05 + 1e-6 }
            guard let step = PlanarTracker.solve(A, bvec) else { break }
            var biggest: CGFloat = 0
            for m in 0..<8 {
                let dm = CGFloat(step[m])
                guard dm.isFinite else { return nil }
                // (a step at most a pixel or two at a time: it is a refinement, not a search)
                let dmc = max(-2, min(2, dm))
                if m % 2 == 0 { c[m / 2].x += dmc } else { c[m / 2].y += dmc }
                biggest = max(biggest, abs(dmc))
            }
            if biggest < 0.02 { break }
        }
        let out = c.map { CGPoint(x: $0.x * s, y: $0.y * s) }
        guard let now = samplePicture(pyr, out) else { return nil }
        return (out, pictureMatch(pic, now))
    }

    /// Is `q` the card as last seen clearly: the same size, in about the same place (re-finding the
    /// card itself)? (not half of it or a box round its picture: those are smaller)
    private func nearLastCard(_ q: [CGPoint]) -> Bool {
        guard let lg = lastGood, frameIndex - lg.frame < 90, lg.size > 0 else { return false }
        let r = quadSize(q) / lg.size
        return r > 0.85 && r < 1.18 && alignedDistance(q, lg.quad) < 0.35 * Double(lg.size)
    }

    /// Mean corner distance between two outlines, whichever way round they are labelled.
    private func alignedDistance(_ a: [CGPoint], _ b: [CGPoint]) -> Double {
        guard a.count == 4, b.count == 4 else { return .infinity }
        var best = Double.infinity
        for rev in [false, true] {
            let bb = rev ? Array(b.reversed()) : b
            for s in 0..<4 {
                var d = 0.0
                for k in 0..<4 { d += Double(hypot(a[k].x - bb[(k + s) % 4].x, a[k].y - bb[(k + s) % 4].y)) }
                best = min(best, d / 4)
            }
        }
        return best
    }

    private func quadSize(_ q: [CGPoint]) -> CGFloat {
        (0..<4).map { k in hypot(q[(k + 1) % 4].x - q[k].x, q[(k + 1) % 4].y - q[k].y) }.max() ?? 0
    }

    private func quadArea(_ q: [CGPoint]) -> Double {
        guard q.count == 4 else { return 0 }
        var a = 0.0
        for k in 0..<4 { let p = q[k], n = q[(k + 1) % 4]; a += Double(p.x * n.y - n.x * p.y) }
        return abs(a) / 2
    }

    /// A candidate that contains `q` (all its corners inside) and is clearly bigger, scoring at least
    /// `minScore`: the best-scoring such one.
    private func enclosing(_ q: [CGPoint], in all: [(quad: [CGPoint], how: String, score: Double)],
                           minScore: Double) -> (quad: [CGPoint], how: String, score: Double)? {
        let aq = quadArea(q)
        guard aq > 0 else { return nil }
        var out: (quad: [CGPoint], how: String, score: Double)?
        for c in all where c.score >= minScore && quadArea(c.quad) > 1.3 * aq {
            let grown = PlanarTracker.shrunk(c.quad, -0.03)
            guard q.allSatisfy({ PlanarTracker.inside(grown, $0) }) else { continue }
            if out == nil || c.score > out!.score { out = c }
        }
        return out
    }

    /// A candidate that contains `q`, is clearly bigger, scores at least `minScore` and shows a
    /// card's corner indices: the best-scoring such one.
    private func enclosingWithIndex(_ q: [CGPoint], in all: [(quad: [CGPoint], how: String, score: Double)],
                                    minScore: Double, paper: PaperReader?) -> (quad: [CGPoint], how: String, score: Double)? {
        let aq = quadArea(q)
        guard aq > 0 else { return nil }
        var out: (quad: [CGPoint], how: String, score: Double)?
        for c in all where c.score >= minScore && quadArea(c.quad) > 1.3 * aq && (out == nil || c.score > out!.score) {
            let grown = PlanarTracker.shrunk(c.quad, -0.03)
            guard q.allSatisfy({ PlanarTracker.inside(grown, $0) }) else { continue }
            guard let ip = indexPattern(paper, c.quad)?.both, ip >= 0.12 else { continue }
            out = c
        }
        return out
    }

    /// 1 for a shape the same size as the card seen recently, falling off for much smaller or
    /// bigger shapes (a finger, the gap between two fingers, a whole hand).
    private func sizeConsistency(_ q: [CGPoint]) -> Double {
        guard let lg = lastGood, frameIndex - lg.frame < 90, lg.size > 0 else { return 1 }
        let r = Double(quadSize(q) / lg.size)
        guard r > 0, r.isFinite else { return 0 }
        return exp(-pow(log(r) / 0.6, 2))
    }

    /// Fraction of points just inside the quad's edges that look like paper. Only points inside the
    /// picture count: a card coming in, going out or held very close is partly out of view, and
    /// what can't be seen says nothing either way (counting it as "not paper" is what kept a card
    /// that is only partly in the picture from being taken until it was almost all in view).
    /// Also returns the fraction of those points that are in view.
    private func paperFraction(_ paper: PaperReader, _ quad: [CGPoint]) -> (Double, Double) {
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        let size = (0..<4).map { k in hypot(quad[(k + 1) % 4].x - quad[k].x, quad[(k + 1) % 4].y - quad[k].y) }.max() ?? 0
        guard size > 8, size.isFinite else { return (0, 0) }
        var ok = 0, total = 0, all = 0
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
                all += 1
                guard x >= 0, y >= 0, x <= CGFloat(paper.w - 1), y <= CGFloat(paper.h - 1) else { continue }
                total += 1
                if paper.isPaper(x, y) { ok += 1 }
            }
        }
        // (too little of it in view to tell)
        let seen = all == 0 ? 0 : Double(total) / Double(all)
        return (total < 12 ? 0 : Double(ok) / Double(total), seen)
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
        rect.minimumAspectRatio = 0.2
        rect.maximumAspectRatio = 1.0
        rect.quadratureTolerance = 40
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
        // A shape cut off by the edge of the picture is kept too: it gets completed to a full card.
        func touchesBorder(_ q: Quad) -> Bool {
            q.points.filter { $0.x < w * 0.01 || $0.x > w * 0.99 || $0.y < h * 0.01 || $0.y > h * 0.99 }.count >= 2
        }
        if useDocumentDetector, let d = doc.results?.first, d.confidence > 0.3 {
            let dq = quad(from: d, w: w, h: h)
            if plausible(dq, w: w, h: h) || touchesBorder(dq) { out.append((dq, "doc")) }
        }
        let rects = (rect.results ?? [])
            .sorted { area($0) * CGFloat($0.confidence) > area($1) * CGFloat($1.confidence) }
        for r in rects {
            let q = quad(from: r, w: w, h: h)
            if plausible(q, w: w, h: h) || touchesBorder(q) { out.append((q, "rect")) }
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
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        var scale = analysisScale
        if let side = analysisLongSide, side > 64 { scale = min(1, side / CGFloat(max(w, h, 1))) }
        guard scale < 0.999 else { return pb }
        let tw = max(64, Int((CGFloat(w) * scale).rounded()) & ~1)
        let th = max(64, Int((CGFloat(h) * scale).rounded()) & ~1)
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
        let p = q.points
        guard p.allSatisfy({ $0.x.isFinite && $0.y.isFinite && abs($0.x) < 1e5 && abs($0.y) < 1e5 }) else { return false }
        let short = min(q.width, q.height), long = max(q.width, q.height)
        // In a close-up the card is never tiny: rules out pips, fingertips and other small boxes.
        guard short > max(40, 0.13 * min(w, h)), long > 0 else { return false }
        // A card's proportions once perspective is undone (63 x 88 mm = 0.716), so tilted cards pass.
        let r = PlanarTracker.trueAspect(p, f: PlanarTracker.focalFraction * Double(min(w, h)), cx: Double(w / 2), cy: Double(h / 2))
        guard r > 0.45, r < 0.97 else { return false }
        // Convex, no corner angle wildly far from square, opposite sides not wildly different.
        var sign: CGFloat = 0
        for k in 0..<4 {
            let a = p[(k + 3) % 4], b = p[k], c = p[(k + 1) % 4]
            let u = CGVector(dx: a.x - b.x, dy: a.y - b.y), v = CGVector(dx: c.x - b.x, dy: c.y - b.y)
            let lu = hypot(u.dx, u.dy), lv = hypot(v.dx, v.dy)
            guard lu > 1, lv > 1 else { return false }
            let cross = u.dx * v.dy - u.dy * v.dx
            if sign == 0 { sign = cross } else if (cross > 0) != (sign > 0) { return false }
            let cosA = (u.dx * v.dx + u.dy * v.dy) / (lu * lv)
            if abs(cosA) > 0.82 { return false }         // outside 35...145 degrees
        }
        let s0 = p[0].distance(to: p[1]), s1 = p[1].distance(to: p[2])
        let s2 = p[2].distance(to: p[3]), s3 = p[3].distance(to: p[0])
        guard s0 / s2 > 0.5, s0 / s2 < 2, s1 / s3 > 0.5, s1 / s3 < 2 else { return false }
        // Not something filling the whole picture (a doorway, a screen).
        // (a card held very close can be much bigger than the picture, most of it out of view)
        return q.boundingBox.width < w * 2.5 && q.boundingBox.height < h * 2.5
    }
}
