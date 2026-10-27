import UIKit

/// Draws a clean, generic playing-card face as "albedo" (paper = pure white),
/// so the compositor can relight it to match the real card in the camera.
/// Card space is 63 x 88 mm, rendered at 8 px per mm.
enum CardFaceRenderer {
    static let size = CGSize(width: 504, height: 704)

    static let black = UIColor(white: 0.14, alpha: 1)
    static let red = UIColor(red: 0.80, green: 0.07, blue: 0.13, alpha: 1)

    static func render(rank: Int, suit: Int) -> CGImage? {
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = 1
        fmt.opaque = true
        fmt.preferredRange = .standard
        let img = UIGraphicsImageRenderer(size: size, format: fmt).image { ctx in
            let c = ctx.cgContext
            UIColor.white.setFill()
            c.fill(CGRect(origin: .zero, size: size))
            let color = (suit == 1 || suit == 3) ? red : black
            drawIndex(c, rank: rank, suit: suit, color: color)
            // Opposite corner index, rotated 180 degrees.
            c.saveGState()
            c.translateBy(x: size.width, y: size.height)
            c.rotate(by: .pi)
            drawIndex(c, rank: rank, suit: suit, color: color)
            c.restoreGState()

            if rank >= 11 {
                drawCourt(c, rank: rank, suit: suit, color: color)
            } else {
                drawPips(c, rank: rank, suit: suit, color: color)
            }
        }
        return img.cgImage
    }

    /// A face traced from the performer's own deck (see RealFaces).
    static func render(face: RealFaces.Face) -> CGImage? {
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = 1
        fmt.opaque = true
        fmt.preferredRange = .standard
        let img = UIGraphicsImageRenderer(size: size, format: fmt).image { ctx in
            let c = ctx.cgContext
            UIColor.white.setFill()
            c.fill(CGRect(origin: .zero, size: size))
            let path = CGMutablePath()
            for poly in face.polys where poly.count >= 6 {
                path.move(to: CGPoint(x: CGFloat(poly[0]) + 0.5, y: CGFloat(poly[1]) + 0.5))
                var i = 2
                while i + 1 < poly.count {
                    path.addLine(to: CGPoint(x: CGFloat(poly[i]) + 0.5, y: CGFloat(poly[i + 1]) + 0.5))
                    i += 2
                }
                path.closeSubpath()
            }
            // Ink colour measured against the paper in the photo (paper = white here).
            let k: CGFloat = 255 / 250
            c.setFillColor(UIColor(red: min(1, CGFloat(face.ink.0) * k / 255),
                                   green: min(1, CGFloat(face.ink.1) * k / 255),
                                   blue: min(1, CGFloat(face.ink.2) * k / 255), alpha: 1).cgColor)
            c.setShouldAntialias(true)
            c.addPath(path)
            c.fillPath(using: .evenOdd)
        }
        return img.cgImage
    }

    // MARK: - Pieces

    private static func suitImage(_ suit: Int, color: UIColor, pointSize: CGFloat) -> UIImage? {
        let names = ["suit.spade.fill", "suit.heart.fill", "suit.club.fill", "suit.diamond.fill"]
        let cfg = UIImage.SymbolConfiguration(pointSize: pointSize, weight: .regular)
        return UIImage(systemName: names[suit], withConfiguration: cfg)?
            .withTintColor(color, renderingMode: .alwaysOriginal)
    }

    /// Draws a suit symbol whose height is `h`, centred on `center`, optionally upside down.
    private static func drawSuit(_ c: CGContext, _ suit: Int, color: UIColor, center: CGPoint, h: CGFloat, flipped: Bool) {
        guard let img = suitImage(suit, color: color, pointSize: h * 2) else { return }
        let aspect = img.size.width / max(img.size.height, 1)
        let rect = CGRect(x: -h * aspect / 2, y: -h / 2, width: h * aspect, height: h)
        c.saveGState()
        c.translateBy(x: center.x, y: center.y)
        if flipped { c.rotate(by: .pi) }
        img.draw(in: rect)
        c.restoreGState()
    }

    private static func drawIndex(_ c: CGContext, rank: Int, suit: Int, color: UIColor) {
        let label = AppSettings.rankNames[rank - 1]
        let fontSize: CGFloat = 70
        let font = UIFont(name: "TimesNewRomanPS-BoldMT", size: fontSize) ?? .boldSystemFont(ofSize: fontSize)
        let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: color, .kern: label == "10" ? -6 : 0]
        let str = NSAttributedString(string: label, attributes: attrs)
        let textSize = str.size()
        let squeeze: CGFloat = label == "10" ? 0.62 : 0.8
        let centerX: CGFloat = 46
        let top: CGFloat = 22

        c.saveGState()
        c.translateBy(x: centerX, y: top)
        c.scaleBy(x: squeeze, y: 1)
        str.draw(at: CGPoint(x: -textSize.width / 2, y: 0))
        c.restoreGState()

        drawSuit(c, suit, color: color, center: CGPoint(x: centerX, y: top + textSize.height + 18), h: 38, flipped: false)
    }

    private static func drawPips(_ c: CGContext, rank: Int, suit: Int, color: UIColor) {
        let W = size.width, H = size.height
        if rank == 1 {
            drawSuit(c, suit, color: color, center: CGPoint(x: W / 2, y: H / 2), h: suit == 0 ? 210 : 150, flipped: false)
            return
        }
        let L: CGFloat = 0.31, M: CGFloat = 0.5, R: CGFloat = 0.69
        var spots: [(CGFloat, CGFloat)] = []
        let top: CGFloat = 0.19, bot: CGFloat = 0.81
        switch rank {
        case 2: spots = [(M, top), (M, bot)]
        case 3: spots = [(M, top), (M, 0.5), (M, bot)]
        case 4: spots = [(L, top), (R, top), (L, bot), (R, bot)]
        case 5: spots = [(L, top), (R, top), (M, 0.5), (L, bot), (R, bot)]
        case 6: spots = [(L, top), (R, top), (L, 0.5), (R, 0.5), (L, bot), (R, bot)]
        case 7: spots = [(L, top), (R, top), (M, 0.345), (L, 0.5), (R, 0.5), (L, bot), (R, bot)]
        case 8: spots = [(L, top), (R, top), (M, 0.345), (L, 0.5), (R, 0.5), (M, 0.655), (L, bot), (R, bot)]
        case 9: spots = [(L, top), (R, top), (L, 0.397), (R, 0.397), (M, 0.5), (L, 0.603), (R, 0.603), (L, bot), (R, bot)]
        case 10: spots = [(L, top), (R, top), (M, 0.295), (L, 0.397), (R, 0.397), (L, 0.603), (R, 0.603), (M, 0.705), (L, bot), (R, bot)]
        default: break
        }
        let pipH: CGFloat = 84
        for (x, y) in spots {
            drawSuit(c, suit, color: color, center: CGPoint(x: x * W, y: y * H), h: pipH, flipped: y > 0.5)
        }
    }

    /// Simple generic court design (a framed panel with the letter and suit).
    /// For a deck-exact look, import a photo of the real card in Settings.
    private static func drawCourt(_ c: CGContext, rank: Int, suit: Int, color: UIColor) {
        let W = size.width, H = size.height
        let frame = CGRect(x: W * 0.17, y: H * 0.125, width: W * 0.66, height: H * 0.75)
        c.setStrokeColor(color.cgColor)
        c.setLineWidth(4)
        c.stroke(frame)
        c.setLineWidth(1.5)
        c.stroke(frame.insetBy(dx: 9, dy: 9))

        let letter = AppSettings.rankNames[rank - 1]
        let font = UIFont(name: "TimesNewRomanPS-BoldMT", size: 190) ?? .boldSystemFont(ofSize: 190)
        let str = NSAttributedString(string: letter, attributes: [.font: font, .foregroundColor: color])
        let s = str.size()
        for flipped in [false, true] {
            c.saveGState()
            c.translateBy(x: W / 2, y: H / 2)
            if flipped { c.rotate(by: .pi) }
            str.draw(at: CGPoint(x: -s.width / 2, y: -H * 0.33))
            drawSuit(c, suit, color: color, center: CGPoint(x: 0, y: -H * 0.06), h: 60, flipped: false)
            c.restoreGState()
        }
        c.setLineWidth(2)
        c.move(to: CGPoint(x: frame.minX + 9, y: frame.midY))
        c.addLine(to: CGPoint(x: frame.maxX - 9, y: frame.midY))
        c.strokePath()
    }
}
