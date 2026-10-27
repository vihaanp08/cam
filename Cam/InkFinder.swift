import Foundation
import CoreGraphics

/// A second way to find the card that doesn't depend on its outline: look for printing (black or
/// red pips and indices) sitting on white paper, group nearby marks into one card, and box them.
/// Works when the card's edges are hard to see (white card on a white wall, fingers over the edges,
/// motion blur), which is exactly when Vision's rectangle detector tends to give up.
/// All coordinates are gray-image pixels (origin top-left), the same as CardTracker uses.
enum InkFinder {

    /// Rough card outlines (tl, tr, br, bl), one per group of printed marks.
    static func candidates(_ paper: CardTracker.PaperReader) -> [[CGPoint]] {
        let step = 2
        let w = paper.w / step, h = paper.h / step
        guard w > 32, h > 32 else { return [] }
        let n = w * h
        var lum = [Float](repeating: 0, count: n)
        var isInk = [Bool](repeating: false, count: n)
        var isPaper = [Bool](repeating: false, count: n)
        var sat = [Float](repeating: 0, count: n)
        var redish = [Bool](repeating: false, count: n)

        for y in 0..<h {
            for x in 0..<w {
                guard let c = paper.colorAt(x * step, y * step) else { continue }
                let r = Float(c.0), g = Float(c.1), b = Float(c.2)
                let mx = max(r, g, b), mn = min(r, g, b)
                let i = y * w + x
                lum[i] = 0.299 * r + 0.587 * g + 0.114 * b
                sat[i] = mx > 0 ? (mx - mn) / mx : 0
                redish[i] = r - max(g, b) > 45 && abs(g - b) < 40
            }
        }
        // Local paper brightness: maximum over an 11 x 11 neighbourhood (separable).
        let ref = maxFilter(lum, w: w, h: h, radius: 5)
        for i in 0..<n {
            let rf = ref[i]
            isPaper[i] = lum[i] > 0.75 * rf && sat[i] < 0.25 && lum[i] > 100
            isInk[i] = rf > 120 && ((lum[i] < 0.62 * rf && sat[i] < 0.35) || redish[i])
        }

        // Connected marks.
        var label = [Int32](repeating: -1, count: n)
        struct Comp { var count = 0; var x0 = Int.max, y0 = Int.max, x1 = Int.min, y1 = Int.min; var sx = 0.0, sy = 0.0 }
        var comps: [Comp] = []
        var stack: [Int] = []
        for start in 0..<n where isInk[start] && label[start] < 0 {
            let id = Int32(comps.count)
            var c = Comp()
            label[start] = id
            stack.append(start)
            while let p = stack.popLast() {
                let px = p % w, py = p / w
                c.count += 1; c.sx += Double(px); c.sy += Double(py)
                c.x0 = min(c.x0, px); c.x1 = max(c.x1, px); c.y0 = min(c.y0, py); c.y1 = max(c.y1, py)
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let qx = px + dx, qy = py + dy
                        guard qx >= 0, qx < w, qy >= 0, qy < h else { continue }
                        let q = qy * w + qx
                        if isInk[q] && label[q] < 0 { label[q] = id; stack.append(q) }
                    }
                }
            }
            comps.append(c)
        }

        // Keep pip-like marks: modest size, fairly solid, surrounded by paper.
        let area = Double(n)
        var keep: [Int] = []
        for (i, c) in comps.enumerated() {
            let bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1
            guard Double(c.count) >= 0.00012 * area, Double(c.count) <= 0.025 * area else { continue }
            guard Double(bw) <= 0.22 * Double(w), Double(bh) <= 0.22 * Double(h) else { continue }
            guard Double(c.count) >= 0.25 * Double(bw * bh) else { continue }
            let m = max(3, Int(0.5 * Double(max(bw, bh))))
            var ring = 0, ringPaper = 0
            for y in max(0, c.y0 - m)...min(h - 1, c.y1 + m) {
                for x in max(0, c.x0 - m)...min(w - 1, c.x1 + m) where x < c.x0 || x > c.x1 || y < c.y0 || y > c.y1 {
                    ring += 1
                    if isPaper[y * w + x] { ringPaper += 1 }
                }
            }
            guard ring > 0, Double(ringPaper) / Double(ring) >= 0.6 else { continue }
            keep.append(i)
        }
        guard !keep.isEmpty else { return [] }

        // Group marks that sit on the same piece of paper.
        var parent = Array(0..<keep.count)
        func find(_ a: Int) -> Int { var a = a; while parent[a] != a { parent[a] = parent[parent[a]]; a = parent[a] }; return a }
        func center(_ k: Int) -> (Double, Double) { let c = comps[keep[k]]; return (c.sx / Double(c.count), c.sy / Double(c.count)) }
        func diag(_ k: Int) -> Double { let c = comps[keep[k]]; return hypot(Double(c.x1 - c.x0 + 1), Double(c.y1 - c.y0 + 1)) }
        func paperBetween(_ a: (Double, Double), _ b: (Double, Double)) -> Bool {
            var ok = 0, tot = 0
            for s in 0..<9 {
                let t = 0.2 + 0.6 * Double(s) / 8
                let x = Int((a.0 + t * (b.0 - a.0)).rounded()), y = Int((a.1 + t * (b.1 - a.1)).rounded())
                guard x >= 0, x < w, y >= 0, y < h else { continue }
                tot += 1
                if isPaper[y * w + x] || isInk[y * w + x] { ok += 1 }
            }
            return tot > 0 && Double(ok) / Double(tot) >= 0.75
        }
        for a in 0..<keep.count {
            for b in (a + 1)..<keep.count {
                let ca = center(a), cb = center(b)
                let d = hypot(ca.0 - cb.0, ca.1 - cb.1)
                if d < 3 * max(diag(a), diag(b)) + 6 && paperBetween(ca, cb) {
                    parent[find(a)] = find(b)
                }
            }
        }
        var groups: [Int: [Int]] = [:]
        for a in 0..<keep.count { groups[find(a), default: []].append(a) }

        var out: [[CGPoint]] = []
        for (_, members) in groups {
            // pixel positions of the group's marks (subsampled to keep it quick)
            var pts: [(Double, Double)] = []
            let ids = Set(members.map { Int32(keep[$0]) })
            var total = 0
            for k in members { total += comps[keep[k]].count }
            if members.count < 2 && Double(total) < 0.002 * area { continue }
            let stride = max(1, total / 1500)
            var seen = 0
            for k in members {
                let c = comps[keep[k]]
                for y in c.y0...c.y1 {
                    for x in c.x0...c.x1 where ids.contains(label[y * w + x]) {
                        seen += 1
                        if seen % stride == 0 { pts.append((Double(x), Double(y))) }
                    }
                }
            }
            guard pts.count >= 8 else { continue }
            let cx = pts.map(\.0).reduce(0, +) / Double(pts.count), cy = pts.map(\.1).reduce(0, +) / Double(pts.count)
            // orientation of the smallest box around the marks (the pip grid lines up with the card)
            var best: (area: Double, u: (Double, Double), v: (Double, Double), u0: Double, u1: Double, v0: Double, v1: Double)?
            var deg = 0.0
            while deg < 90 {
                let th = deg * Double.pi / 180
                let u = (cos(th), sin(th)), v = (-sin(th), cos(th))
                var u0 = Double.infinity, u1 = -Double.infinity, v0 = Double.infinity, v1 = -Double.infinity
                for p in pts {
                    let a = (p.0 - cx) * u.0 + (p.1 - cy) * u.1, b = (p.0 - cx) * v.0 + (p.1 - cy) * v.1
                    u0 = min(u0, a); u1 = max(u1, a); v0 = min(v0, b); v1 = max(v1, b)
                }
                let ar = (u1 - u0) * (v1 - v0)
                if best == nil || ar < best!.area { best = (ar, u, v, u0, u1, v0, v1) }
                deg += 2
            }
            guard let bx = best else { continue }
            // The printing covers most of the face; the card edge is a little further out.
            let gu = 0.06 * (bx.u1 - bx.u0) + 2, gv = 0.06 * (bx.v1 - bx.v0) + 2
            func corner(_ a: Double, _ b: Double) -> CGPoint {
                CGPoint(x: (cx + a * bx.u.0 + b * bx.v.0) * Double(step), y: (cy + a * bx.u.1 + b * bx.v.1) * Double(step))
            }
            out.append([corner(bx.u0 - gu, bx.v0 - gv), corner(bx.u1 + gu, bx.v0 - gv),
                        corner(bx.u1 + gu, bx.v1 + gv), corner(bx.u0 - gu, bx.v1 + gv)])
        }
        return out
    }

    /// Moves each side of a rough outline (from the printing) out to the nearest strong edge with
    /// paper inside it: the real card edge. Gray-image coordinates.
    static func fitEdges(_ g: GrayImage, _ paper: CardTracker.PaperReader, _ quad: [CGPoint]) -> [CGPoint]? {
        guard quad.count == 4 else { return nil }
        let cx = quad.map(\.x).reduce(0, +) / 4, cy = quad.map(\.y).reduce(0, +) / 4
        var lines: [(CGPoint, CGVector)] = []
        for k in 0..<4 {
            let a = quad[k], b = quad[(k + 1) % 4]
            let L = hypot(b.x - a.x, b.y - a.y)
            guard L > 4 else { return nil }
            let d = CGVector(dx: (b.x - a.x) / L, dy: (b.y - a.y) / L)
            var nrm = CGVector(dx: -d.dy, dy: d.dx)
            if nrm.dx * ((a.x + b.x) / 2 - cx) + nrm.dy * ((a.y + b.y) / 2 - cy) < 0 { nrm = CGVector(dx: -nrm.dx, dy: -nrm.dy) }
            let p1 = quad[(k + 1) % 4], p2 = quad[(k + 2) % 4], p3 = quad[(k + 3) % 4]
            let D = (hypot(p2.x - p1.x, p2.y - p1.y) + hypot(a.x - p3.x, a.y - p3.y)) / 2
            let stepO = max(1, D / 60)
            var bestOff: CGFloat = 0, bestEv = 0.0
            var o = -0.06 * D
            while o <= 0.2 * D {
                var hits = 0, seen = 0
                for s in 0..<20 {
                    let t = 0.1 + 0.8 * CGFloat(s) / 19
                    let px = a.x + t * (b.x - a.x) + o * nrm.dx, py = a.y + t * (b.y - a.y) + o * nrm.dy
                    guard g.contains(Double(px - 4 * abs(nrm.dx)), Double(py - 4 * abs(nrm.dy))),
                          g.contains(Double(px + 4 * abs(nrm.dx)), Double(py + 4 * abs(nrm.dy))) else { continue }
                    seen += 1
                    var gr: Float = 0
                    for e in [-1.0, 0.0, 1.0] as [CGFloat] {
                        let qx = px + e * nrm.dx, qy = py + e * nrm.dy
                        gr = max(gr, abs(g.sample(Float(qx - 2 * nrm.dx), Float(qy - 2 * nrm.dy)) - g.sample(Float(qx + 2 * nrm.dx), Float(qy + 2 * nrm.dy))))
                    }
                    if gr > 0.02 && paper.isPaper(px - 3 * nrm.dx, py - 3 * nrm.dy) { hits += 1 }
                }
                let ev = seen >= 5 ? Double(hits) / Double(seen) : 0
                if ev > bestEv + 1e-9 { bestEv = ev; bestOff = o }
                o += stepO
            }
            let off = bestEv >= 0.3 ? bestOff : 0
            lines.append((CGPoint(x: a.x + off * nrm.dx, y: a.y + off * nrm.dy), d))
        }
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

    private static func maxFilter(_ v: [Float], w: Int, h: Int, radius r: Int) -> [Float] {
        var tmp = [Float](repeating: 0, count: v.count), out = tmp
        for y in 0..<h {
            for x in 0..<w {
                var m: Float = 0
                for xx in max(0, x - r)...min(w - 1, x + r) { m = max(m, v[y * w + xx]) }
                tmp[y * w + x] = m
            }
        }
        for y in 0..<h {
            for x in 0..<w {
                var m: Float = 0
                for yy in max(0, y - r)...min(h - 1, y + r) { m = max(m, tmp[yy * w + x]) }
                out[y * w + x] = m
            }
        }
        return out
    }
}
