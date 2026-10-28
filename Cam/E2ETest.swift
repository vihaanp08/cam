import AVFoundation
import CoreImage
import UIKit

/// End-to-end test mode, for the simulator (which has no camera). Plays a video, or a folder of
/// frame_00000.jpg ... images, through the real camera engine exactly as camera frames would go,
/// logs every frame's result, saves what the screen shows every few frames, and presses the volume
/// button at chosen frames. Started by launch arguments:
///
///     -e2e <video file or folder inside Documents>  [-e2ePress 150,210]
///
/// Writes Documents/e2e_<name>/log.csv, shot_NNNNN.jpg and, when finished, done.txt.
final class E2ETest {
    static func argument(_ name: String) -> String? {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
        return a[i + 1]
    }

    private let engine: CameraEngine
    private let pressButton: () -> Void
    private let queue = DispatchQueue(label: "cam.e2e", qos: .userInitiated)
    private var log = "i,drew,armed,tracking,source,reason,cardness,h,x0,y0,x1,y1,x2,y2,x3,y3\n"
    private var pool: CVPixelBufferPool?
    private var poolSize = CGSize.zero

    init(engine: CameraEngine, pressButton: @escaping () -> Void) {
        self.engine = engine
        self.pressButton = pressButton
    }

    func run(name: String, presses: Set<Int>) {
        engine.testMode = true
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let src = docs.appendingPathComponent(name)
        let stem = (name as NSString).deletingPathExtension
        let out = docs.appendingPathComponent("e2e_" + stem)
        try? FileManager.default.removeItem(at: out)
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        var index = 0
        engine.onTestFrame = { [weak self] r, armed in
            guard let self = self else { return }
            let h = r.image.extent.height
            let q = r.quad.map { $0.points.map { String(format: "%.1f,%.1f", $0.x, $0.y) }.joined(separator: ",") } ?? ",,,,,,,"
            let reason = r.status.reason.replacingOccurrences(of: ",", with: ";")
            self.log += "\(index),\(r.drewOverlay ? 1 : 0),\(armed ? 1 : 0),\(r.status.tracking),\(r.status.source),\(reason),"
                + String(format: "%.2f", r.status.cardness) + ",\(Int(h)),\(q)\n"
        }

        queue.async {
            // Let the screen come up first.
            Thread.sleep(forTimeInterval: 2.0)
            let next = self.frameSource(src)
            var i = 0
            while let pb = next() {
                let t0 = CACurrentMediaTime()
                index = i
                if presses.contains(i) { DispatchQueue.main.sync { self.pressButton() } }
                self.engine.processTestFrame(pb, time: Double(i) / 30)
                if i % 20 == 0, let img = self.engine.latestImage {
                    self.save(img, to: out.appendingPathComponent(String(format: "shot_%05d.jpg", i)))
                }
                i += 1
                // about real time, so the screen and the timers behave as with the camera
                let left = 1.0 / 30 - (CACurrentMediaTime() - t0)
                if left > 0 { Thread.sleep(forTimeInterval: left) }
            }
            try? self.log.write(to: out.appendingPathComponent("log.csv"), atomically: true, encoding: .utf8)
            try? "done \(i) frames\n".write(to: out.appendingPathComponent("done.txt"), atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Frames

    /// Hands out the frames one at a time as upright BGRA pixel buffers (nil at the end).
    private func frameSource(_ url: URL) -> () -> CVPixelBuffer? {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return { nil } }
        if isDir.boolValue {
            let names = ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? [])
                .filter { $0.hasPrefix("frame_") && $0.hasSuffix(".jpg") }.sorted()
            var k = 0
            return { [weak self] in
                guard let self = self, k < names.count else { return nil }
                defer { k += 1 }
                guard let img = CIImage(contentsOf: url.appendingPathComponent(names[k])) else { return nil }
                return self.render(img)
            }
        }
        // A video, at 30 fps, turned upright like the camera's frames.
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first,
              let reader = try? AVAssetReader(asset: asset) else { return { nil } }
        let t = track.preferredTransform
        let angle = Int((atan2(t.b, t.a) * 180 / .pi).rounded())
        let orientation: CGImagePropertyOrientation = angle == 90 ? .right : angle == -90 ? .left : (angle == 180 || angle == -180) ? .down : .up
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        reader.add(output)
        reader.startReading()
        var lastKept = -1.0
        return { [weak self] in
            // (the reader has to stay alive while frames are read: the closure keeps it)
            guard let self = self, reader.status == .reading else { return nil }
            while let sb = output.copyNextSampleBuffer() {
                guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
                let ts = CMSampleBufferGetPresentationTimeStamp(sb).seconds
                if lastKept >= 0 && ts - lastKept < 1.0 / 30 - 0.004 { continue }
                lastKept = ts
                return self.render(CIImage(cvPixelBuffer: pb).oriented(orientation))
            }
            return nil
        }
    }

    private func render(_ img: CIImage) -> CVPixelBuffer? {
        let e = img.extent.integral
        let size = CGSize(width: e.width, height: e.height)
        if pool == nil || poolSize != size {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: Int(e.width),
                kCVPixelBufferHeightKey as String: Int(e.height),
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            var p: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
            pool = p
            poolSize = size
        }
        guard let pool = pool else { return nil }
        var out: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out)
        guard let pb = out else { return nil }
        engine.ciContext.render(img.transformed(by: CGAffineTransform(translationX: -e.minX, y: -e.minY)), to: pb,
                                bounds: CGRect(origin: .zero, size: size), colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return pb
    }

    private func save(_ img: CIImage, to url: URL) {
        let e = img.extent
        let s = 540 / max(e.width, 1)
        let small = img.transformed(by: CGAffineTransform(translationX: -e.minX, y: -e.minY))
            .transformed(by: CGAffineTransform(scaleX: s, y: s))
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let data = engine.ciContext.jpegRepresentation(of: small, colorSpace: cs, options: [:]) else { return }
        try? data.write(to: url)
    }
}
