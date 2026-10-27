import UIKit
import Vision
import CoreImage
import CoreImage.CIFilterBuiltins

/// Turns a photo of a real card into a flat, evenly lit face image the app can use as the fake.
/// Shoot the card flat on a dark surface, straight from above, in soft even light.
enum CardImporter {
    private static let ctx = CIContext()

    static var facesDir: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("faces")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func listFaces() -> [String] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: facesDir.path)) ?? []
        return names.filter { $0.hasSuffix(".png") }.sorted()
    }

    static func loadFace(named name: String) -> CGImage? {
        guard let img = UIImage(contentsOfFile: facesDir.appendingPathComponent(name).path) else { return nil }
        return img.cgImage
    }

    static func deleteFace(named name: String) {
        try? FileManager.default.removeItem(at: facesDir.appendingPathComponent(name))
    }

    enum ImportError: Error { case noImage, noCard }

    /// Finds the card, flattens it, makes the paper white, saves it. Returns the file name.
    static func importPhoto(_ image: UIImage) throws -> String {
        guard var ci = CIImage(image: image) else { throw ImportError.noImage }
        ci = ci.oriented(CGImagePropertyOrientation(uiOrientation: image.imageOrientation))
        let ext = ci.extent
        ci = ci.transformed(by: CGAffineTransform(translationX: -ext.minX, y: -ext.minY))

        let req = VNDetectRectanglesRequest()
        req.minimumAspectRatio = 0.5
        req.maximumAspectRatio = 0.9
        req.minimumSize = 0.15
        req.quadratureTolerance = 25
        req.maximumObservations = 1
        try VNImageRequestHandler(ciImage: ci, options: [:]).perform([req])
        guard let obs = req.results?.first else { throw ImportError.noCard }

        let w = ci.extent.width, h = ci.extent.height
        func p(_ n: CGPoint) -> CGPoint { CGPoint(x: n.x * w, y: n.y * h) }
        let q = Quad(tl: p(obs.topLeft), tr: p(obs.topRight), br: p(obs.bottomRight), bl: p(obs.bottomLeft)).uprightPortrait()

        let pc = CIFilter.perspectiveCorrection()
        pc.inputImage = ci
        pc.topLeft = q.tl; pc.topRight = q.tr; pc.bottomRight = q.br; pc.bottomLeft = q.bl
        pc.crop = true
        guard let flat = pc.outputImage else { throw ImportError.noCard }
        let fe = flat.extent
        let size = CardFaceRenderer.size
        let scaled = flat
            .transformed(by: CGAffineTransform(translationX: -fe.minX, y: -fe.minY))
            .transformed(by: CGAffineTransform(scaleX: size.width / fe.width, y: size.height / fe.height))
            .cropped(to: CGRect(origin: .zero, size: size))

        // Measure the paper (bright end of the border) and scale it to pure white.
        let border = CGRect(x: size.width * 0.06, y: size.height * 0.45, width: size.width * 0.02, height: size.height * 0.1)
        let avg = CIFilter.areaAverage()
        avg.inputImage = scaled
        avg.extent = border
        var paper: [Float] = [1, 1, 1, 1]
        if let a = avg.outputImage {
            paper.withUnsafeMutableBytes { raw in
                ctx.render(a, toBitmap: raw.baseAddress!, rowBytes: 16, bounds: CGRect(origin: a.extent.origin, size: CGSize(width: 1, height: 1)),
                           format: .RGBAf, colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
            }
        }
        let m = CIFilter.colorMatrix()
        m.inputImage = scaled
        m.rVector = CIVector(x: CGFloat(1 / max(paper[0], 0.05)), y: 0, z: 0, w: 0)
        m.gVector = CIVector(x: 0, y: CGFloat(1 / max(paper[1], 0.05)), z: 0, w: 0)
        m.bVector = CIVector(x: 0, y: 0, z: CGFloat(1 / max(paper[2], 0.05)), w: 0)
        m.aVector = CIVector(x: 0, y: 0, z: 0, w: 1)
        guard let normalized = m.outputImage?.cropped(to: CGRect(origin: .zero, size: size)),
              let cg = ctx.createCGImage(normalized, from: CGRect(origin: .zero, size: size),
                                         format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else {
            throw ImportError.noCard
        }
        let name = "face-\(Int(Date().timeIntervalSince1970)).png"
        guard let data = UIImage(cgImage: cg).pngData() else { throw ImportError.noImage }
        try data.write(to: facesDir.appendingPathComponent(name))
        return name
    }
}

extension CGImagePropertyOrientation {
    init(uiOrientation o: UIImage.Orientation) {
        switch o {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
