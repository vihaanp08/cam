import CoreGraphics
import Foundation

/// A card outline in Core Image pixel coordinates (origin bottom-left, y up).
/// "Top" means the card's top short edge, which on screen is the edge with larger y.
struct Quad {
    var tl: CGPoint
    var tr: CGPoint
    var br: CGPoint
    var bl: CGPoint

    var points: [CGPoint] { [tl, tr, br, bl] }

    init(tl: CGPoint, tr: CGPoint, br: CGPoint, bl: CGPoint) {
        self.tl = tl; self.tr = tr; self.br = br; self.bl = bl
    }

    init(_ p: [CGPoint]) {
        tl = p[0]; tr = p[1]; br = p[2]; bl = p[3]
    }

    var center: CGPoint {
        CGPoint(x: (tl.x + tr.x + br.x + bl.x) / 4, y: (tl.y + tr.y + br.y + bl.y) / 4)
    }

    /// Average length of the top and bottom edges.
    var width: CGFloat { (tl.distance(to: tr) + bl.distance(to: br)) / 2 }
    /// Average length of the left and right edges.
    var height: CGFloat { (tl.distance(to: bl) + tr.distance(to: br)) / 2 }

    var boundingBox: CGRect {
        let xs = points.map(\.x), ys = points.map(\.y)
        return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
    }

    /// Rotate the labels by one step: the old right edge becomes the top.
    func rotatedLabels(_ steps: Int) -> Quad {
        var p = points
        let s = ((steps % 4) + 4) % 4
        if s > 0 { p = Array(p[s...] + p[..<s]) }
        return Quad(p)
    }

    /// Picks the labelling (cyclic shift) that best matches a previous quad,
    /// so the overlay never jumps by 90 degrees between frames.
    func aligned(to prev: Quad) -> Quad {
        var best = self
        var bestCost = CGFloat.greatestFiniteMagnitude
        for s in 0..<4 {
            let q = rotatedLabels(s)
            let cost = zip(q.points, prev.points).reduce(CGFloat(0)) { $0 + $1.0.distance(to: $1.1) }
            if cost < bestCost { bestCost = cost; best = q }
        }
        return best
    }

    /// Labels a fresh detection so the card reads upright in portrait:
    /// short edges on top and bottom, and the top edge is the higher one on screen.
    func uprightPortrait() -> Quad {
        var q = self
        if q.width > q.height { q = q.rotatedLabels(1) }
        let topMid = (q.tl.y + q.tr.y) / 2
        let bottomMid = (q.bl.y + q.br.y) / 2
        if topMid < bottomMid { q = q.rotatedLabels(2) }
        return q
    }

    func lerp(to other: Quad, _ t: CGFloat) -> Quad {
        Quad(zip(points, other.points).map { CGPoint(x: $0.x + ($1.x - $0.x) * t, y: $0.y + ($1.y - $0.y) * t) })
    }

    /// Mirror horizontally inside an image of the given width (labels swap so the card stays upright).
    func mirrored(width w: CGFloat) -> Quad {
        func m(_ p: CGPoint) -> CGPoint { CGPoint(x: w - p.x, y: p.y) }
        return Quad(tl: m(tr), tr: m(tl), br: m(bl), bl: m(br))
    }
}

extension CGPoint {
    func distance(to p: CGPoint) -> CGFloat { hypot(x - p.x, y - p.y) }
}

/// One Euro filter (Casiez et al.): low jitter when still, low lag when moving.
struct OneEuroFilter {
    var minCutoff: Double
    var beta: Double
    var dCutoff: Double = 1.0
    private var xPrev: Double?
    private var dxPrev: Double = 0
    private(set) var velocity: Double = 0

    init(minCutoff: Double, beta: Double) {
        self.minCutoff = minCutoff
        self.beta = beta
    }

    private func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1.0 / (2 * Double.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    }

    mutating func filter(_ x: Double, dt: Double) -> Double {
        guard let prev = xPrev, dt > 0 else {
            xPrev = x
            return x
        }
        let dx = (x - prev) / dt
        let aD = alpha(dCutoff, dt)
        let dxHat = aD * dx + (1 - aD) * dxPrev
        dxPrev = dxHat
        velocity = dxHat
        let cutoff = minCutoff + beta * abs(dxHat)
        let a = alpha(cutoff, dt)
        let xHat = a * x + (1 - a) * prev
        xPrev = xHat
        return xHat
    }

    mutating func reset() {
        xPrev = nil
        dxPrev = 0
        velocity = 0
    }
}

/// Smooths all 8 corner coordinates of a quad.
struct QuadSmoother {
    private var filters = (0..<8).map { _ in OneEuroFilter(minCutoff: 1.4, beta: 0.012) }

    mutating func smooth(_ q: Quad, dt: Double) -> Quad {
        var out: [CGPoint] = []
        for (i, p) in q.points.enumerated() {
            let x = filters[i * 2].filter(Double(p.x), dt: dt)
            let y = filters[i * 2 + 1].filter(Double(p.y), dt: dt)
            out.append(CGPoint(x: x, y: y))
        }
        return Quad(out)
    }

    /// Per-corner velocity in pixels per second, for short extrapolation while coasting.
    var velocities: [CGVector] {
        (0..<4).map { CGVector(dx: filters[$0 * 2].velocity, dy: filters[$0 * 2 + 1].velocity) }
    }

    mutating func reset() {
        for i in filters.indices { filters[i].reset() }
    }
}
