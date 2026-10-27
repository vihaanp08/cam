// Offline test harness for the card-change pipeline. Runs on macOS (GitHub Actions) with the
// same tracker / compositor / pipeline code as the app, and reports how well the fake card holds.
//
//   harness synth <framesDir> <fakeFace.png> <outPrefix> <analysisScale>
//   harness video <movie>     <fakeFace.png> <outPrefix> <analysisScale>
//
// Writes <outPrefix>.json (metrics), <outPrefix>.jpg (contact sheet) and, for videos,
// <outPrefix>.mp4 (the result with the fake card painted in).

import Foundation
import CoreImage
import CoreVideo
import CoreGraphics
import CoreText
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Shared helpers

let ciContext = CIContext(options: [
    .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
    .cacheIntermediates: false
])
let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func loadCGImage(_ path: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fail("cannot read \(path)") }
    return img
}

func makeBuffer(_ w: Int, _ h: Int) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
    ]
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    guard let b = pb else { fail("cannot make pixel buffer") }
    return b
}

func render(_ img: CIImage) -> CGImage? {
    ciContext.createCGImage(img, from: img.extent, format: .RGBA8, colorSpace: sRGB)
}

func writeJPEG(_ img: CGImage, _ path: String, quality: Double = 0.72) {
    guard let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                    UTType.jpeg.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dst, img, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    CGImageDestinationFinalize(dst)
}

func writeJSON(_ obj: [String: Any], _ path: String) {
    let data = try! JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
    try! data.write(to: URL(fileURLWithPath: path))
}

func dist(_ a: CGPoint, _ b: CGPoint) -> Double { Double(hypot(a.x - b.x, a.y - b.y)) }

/// Mean corner distance, allowing for any corner labelling (rotation / reversed order).
func quadError(_ q: [CGPoint], _ g: [CGPoint]) -> Double {
    var best = Double.infinity
    for rev in [false, true] {
        let qq = rev ? Array(q.reversed()) : q
        for s in 0..<4 {
            var sum = 0.0
            for k in 0..<4 { sum += dist(qq[(k + s) % 4], g[k]) }
            best = min(best, sum / 4)
        }
    }
    return best
}

func percentile(_ v: [Double], _ p: Double) -> Double {
    guard !v.isEmpty else { return 0 }
    let s = v.sorted()
    return s[min(s.count - 1, Int(Double(s.count - 1) * p))]
}

func r2(_ x: Double) -> Double { (x * 100).rounded() / 100 }

/// One processed frame kept for the contact sheet.
struct Thumb {
    var index: Int
    var image: CGImage
    var label: String
    var tracked: [CGPoint]?   // top-left origin, full-frame pixels
    var truth: [CGPoint]?
    var frameSize: CGSize
}

func contactSheet(_ thumbs: [Thumb], columns: Int, cellWidth: CGFloat) -> CGImage? {
    guard let first = thumbs.first else { return nil }
    let aspect = first.frameSize.height / first.frameSize.width
    let cw = cellWidth, ch = (cellWidth * aspect).rounded()
    let rows = (thumbs.count + columns - 1) / columns
    let W = Int(cw) * columns, H = Int(ch + 18) * rows
    guard let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0, space: sRGB,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(gray: 0.1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
    for (n, t) in thumbs.enumerated() {
        let col = n % columns, row = n / columns
        let x0 = CGFloat(col) * cw
        let yTop = CGFloat(row) * (ch + 18)              // from the top
        let cell = CGRect(x: x0, y: CGFloat(H) - yTop - ch, width: cw, height: ch)
        ctx.draw(t.image, in: cell)
        let sx = cw / t.frameSize.width, sy = ch / t.frameSize.height
        func toCell(_ p: CGPoint) -> CGPoint { CGPoint(x: cell.minX + p.x * sx, y: cell.maxY - p.y * sy) }
        func poly(_ pts: [CGPoint], _ color: CGColor) {
            ctx.setStrokeColor(color); ctx.setLineWidth(1.5)
            ctx.beginPath(); ctx.move(to: toCell(pts[0]))
            for p in pts.dropFirst() { ctx.addLine(to: toCell(p)) }
            ctx.closePath(); ctx.strokePath()
        }
        if let g = t.truth { poly(g, CGColor(red: 0, green: 1, blue: 0, alpha: 0.9)) }
        if let q = t.tracked { poly(q, CGColor(red: 1, green: 0, blue: 1, alpha: 0.9)) }
        let attr = NSAttributedString(string: t.label, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Menlo" as CFString, 10, nil),
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 1, alpha: 1)
        ])
        let line = CTLineCreateWithAttributedString(attr)
        ctx.textPosition = CGPoint(x: x0 + 3, y: CGFloat(H) - yTop - ch - 13)
        CTLineDraw(line, ctx)
    }
    return ctx.makeImage()
}

// MARK: - Run

let args = CommandLine.arguments
guard args.count >= 6 else { fail("usage: harness synth|video <input> <fakeFace.png> <outPrefix> <analysisScale>") }
let mode = args[1], input = args[2], facePath = args[3], outPrefix = args[4]
let analysisScale = CGFloat(Double(args[5]) ?? 0.5)

let albedo = FramePipeline.makeAlbedo(loadCGImage(facePath)).normal
let pipe = FramePipeline(context: ciContext)
pipe.tracker.analysisScale = analysisScale
var cfg = RenderConfig()
cfg.mirror = false

var drew: [Bool] = []
var quads: [[CGPoint]?] = []
var sources: [String: Int] = [:]
var reasons: [String: Int] = [:]
var msPerFrame: [Double] = []
var thumbs: [Thumb] = []
var frameSize = CGSize.zero

var trace = "i,drew,source,reason,score,x0,y0,x1,y1,x2,y2,x3,y3\n"

func record(_ i: Int, _ out: FramePipeline.Output, _ h: CGFloat, ms: Double) {
    drew.append(out.drewOverlay)
    let q = out.quad.map { q in q.points.map { CGPoint(x: $0.x, y: h - $0.y) } }
    quads.append(q)
    let qs = q.map { $0.map { String(format: "%.1f,%.1f", $0.x, $0.y) }.joined(separator: ",") } ?? ",,,,,,,"
    trace += "\(i),\(out.drewOverlay ? 1 : 0),\(out.status.source),\(out.status.reason),\(String(format: "%.3f", pipe.tracker.lastScore)),\(qs)\n"
    sources[out.status.source, default: 0] += 1
    let r = out.status.reason.hasPrefix("low paper") ? "low paper" : out.status.reason
    reasons[r, default: 0] += 1
    msPerFrame.append(ms)
}

if mode == "synth" {
    struct GTFrame: Decodable { let i: Int; let visible: Bool; let corners: [[Double]]; let still: Bool }
    struct GT: Decodable { let width: Int; let height: Int; let fps: Double; let frames: [GTFrame] }
    let gtData = try! Data(contentsOf: URL(fileURLWithPath: input + "/gt.json"))
    let gt = try! JSONDecoder().decode(GT.self, from: gtData)
    frameSize = CGSize(width: gt.width, height: gt.height)
    let pb = makeBuffer(gt.width, gt.height)
    let sheetEvery = max(1, gt.frames.count / 12)
    var errors: [Double] = []
    var errorByFrame: [Int: Double] = [:]
    var outputs: [Int: CIImage] = [:]

    for f in gt.frames {
        let cg = loadCGImage(String(format: "%@/frame_%05d.jpg", input, f.i))
        ciContext.render(CIImage(cgImage: cg), to: pb)
        let t0 = Date()
        let out = pipe.process(pb, time: Double(f.i) / gt.fps, cfg: cfg, armed: true, albedo: albedo)
        let ms = Date().timeIntervalSince(t0) * 1000
        record(f.i, out, CGFloat(gt.height), ms: ms)
        let truth = f.corners.map { CGPoint(x: $0[0], y: $0[1]) }
        if out.drewOverlay, f.visible, let q = quads.last!, q.count == 4 {
            let e = quadError(q, truth)
            errors.append(e)
            errorByFrame[f.i] = e
        }
        if f.i % sheetEvery == 0 {
            // Keep the composited image now (the pixel buffer is reused for the next frame).
            if let img = render(out.image) { outputs[f.i] = CIImage(cgImage: img) }
        }
    }

    // Metrics
    let firstLock = drew.firstIndex(of: true) ?? gt.frames.count
    var visibleAfterLock = 0, covered = 0, exposures: [Int] = [], flickers = 0, falseDraw = 0
    for f in gt.frames where f.i >= firstLock {
        // allow 3 frames to re-acquire after the card comes back into view
        let justReturned = (1...3).contains { f.i - $0 >= 0 && !gt.frames[f.i - $0].visible }
        if f.visible && !justReturned {
            visibleAfterLock += 1
            if drew[f.i] { covered += 1 } else { exposures.append(f.i) }
            if f.i > 0 && drew[f.i - 1] && !drew[f.i] { flickers += 1 }
        }
        if !f.visible && drew[f.i] { falseDraw += 1 }
    }
    // Jitter during the still hold: spread of each tracked corner around its mean.
    var stillCorners: [[CGPoint]] = []
    for f in gt.frames where f.still && drew[f.i] { if let q = quads[f.i] { stillCorners.append(q) } }
    var jitter = 0.0
    if stillCorners.count > 3 {
        let ref = stillCorners[0]
        let aligned = stillCorners.map { q -> [CGPoint] in
            // align labelling to the first frame
            var best = q; var bestE = Double.infinity
            for s in 0..<4 { let r = (0..<4).map { q[($0 + s) % 4] }; let e = zip(r, ref).map { dist($0, $1) }.reduce(0, +); if e < bestE { bestE = e; best = r } }
            return best
        }
        var total = 0.0
        for k in 0..<4 {
            let xs = aligned.map { Double($0[k].x) }, ys = aligned.map { Double($0[k].y) }
            let mx = xs.reduce(0, +) / Double(xs.count), my = ys.reduce(0, +) / Double(ys.count)
            var sq = 0.0
            for (x, y) in zip(xs, ys) {
                let dx = x - mx, dy = y - my
                sq += dx * dx + dy * dy
            }
            total += (sq / Double(xs.count)).squareRoot()
        }
        jitter = total / 4
    }
    let worst = errorByFrame.sorted { $0.value > $1.value }.prefix(5).map { "#\($0.key): \(r2($0.value)) px" }
    let metrics: [String: Any] = [
        "frames": gt.frames.count,
        "first_lock_frame": firstLock,
        "coverage_pct": r2(100 * Double(covered) / Double(max(1, visibleAfterLock))),
        "exposed_frames": exposures.count,
        "exposed_frame_list": Array(exposures.prefix(40)),
        "flicker_events": flickers,
        "false_draw_frames": falseDraw,
        "corner_err_mean_px": r2(errors.isEmpty ? -1 : errors.reduce(0, +) / Double(errors.count)),
        "corner_err_p95_px": r2(percentile(errors, 0.95)),
        "corner_err_max_px": r2(errors.max() ?? -1),
        "still_jitter_px": r2(jitter),
        "ms_per_frame_mean": r2(msPerFrame.reduce(0, +) / Double(max(1, msPerFrame.count))),
        "ms_per_frame_p95": r2(percentile(msPerFrame, 0.95)),
        "sources": sources,
        "reasons": reasons,
        "document_detector_errors": pipe.tracker.documentErrors,
        "worst_frames": worst,
        "frame": "\(gt.width)x\(gt.height)"
    ]
    writeJSON(metrics, outPrefix + ".json")
    try? trace.write(toFile: outPrefix + "_trace.csv", atomically: true, encoding: .utf8)

    // Contact sheet: evenly spaced frames, then the worst-error frames.
    for (i, img) in outputs.sorted(by: { $0.key < $1.key }) {
        guard let cg = render(img) else { continue }
        let f = gt.frames[i]
        let e = errorByFrame[i].map { String(format: " err %.1f", $0) } ?? ""
        thumbs.append(Thumb(index: i, image: cg, label: "#\(i) \(drew[i] ? "FAKE" : "real")\(e)",
                            tracked: quads[i], truth: f.corners.map { CGPoint(x: $0[0], y: $0[1]) }, frameSize: frameSize))
    }
    if let sheet = contactSheet(thumbs, columns: 6, cellWidth: 200) { writeJPEG(sheet, outPrefix + ".jpg") }
    print(String(data: try! JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)

} else if mode == "video" {
    let asset = AVURLAsset(url: URL(fileURLWithPath: input))
    guard let track = asset.tracks(withMediaType: .video).first else { fail("no video track") }
    let t = track.preferredTransform
    let angle = atan2(t.b, t.a) * 180 / .pi
    let orientation: CGImagePropertyOrientation
    switch Int(angle.rounded()) {
    case 90: orientation = .right
    case -90: orientation = .left
    case 180, -180: orientation = .down
    default: orientation = .up
    }
    let reader = try! AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ])
    reader.add(output)
    reader.startReading()

    var pb: CVPixelBuffer?
    var lastKept = -1.0
    var index = 0
    var collected: [(Int, CGImage)] = []
    var writer: AVAssetWriter?
    var writerInput: AVAssetWriterInput?
    var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    let outMovie = outPrefix + ".mp4"
    try? FileManager.default.removeItem(atPath: outMovie)
    var startTime: CMTime?
    var outW = 0, outH = 0

    while let sb = output.copyNextSampleBuffer() {
        guard let src = CMSampleBufferGetImageBuffer(sb) else { continue }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        let ts = pts.seconds
        if lastKept >= 0 && ts - lastKept < 1.0 / 30 - 0.004 { continue }   // run at 30 fps like the app
        lastKept = ts
        let img = CIImage(cvPixelBuffer: src).oriented(orientation)
        let e = img.extent
        let upright = img.transformed(by: CGAffineTransform(translationX: -e.minX, y: -e.minY))
        let w = Int(e.width), h = Int(e.height)
        if pb == nil {
            pb = makeBuffer(w, h)
            frameSize = CGSize(width: w, height: h)
            outW = (w / 2) & ~1; outH = (h / 2) & ~1
            writer = try? AVAssetWriter(outputURL: URL(fileURLWithPath: outMovie), fileType: .mp4)
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: outW, AVVideoHeightKey: outH,
                AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 2_500_000]
            ])
            input.expectsMediaDataInRealTime = false
            writerInput = input
            adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH
            ])
            if let wr = writer, wr.canAdd(input) { wr.add(input); wr.startWriting(); wr.startSession(atSourceTime: .zero) }
        }
        ciContext.render(upright, to: pb!)
        let t0 = Date()
        let out = pipe.process(pb!, time: ts, cfg: cfg, armed: true, albedo: albedo)
        let ms = Date().timeIntervalSince(t0) * 1000
        record(index, out, CGFloat(h), ms: ms)

        // Write the result at half size, with the tracked corners drawn (magenta) when tracking.
        if let ad = adaptor, let wi = writerInput, let pool = ad.pixelBufferPool {
            var ob: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &ob)
            if let ob = ob {
                var frameOut = out.image.transformed(by: CGAffineTransform(scaleX: CGFloat(outW) / CGFloat(w), y: CGFloat(outH) / CGFloat(h)))
                if !out.drewOverlay, let q = out.quad {
                    // show where the tracker thinks the card is, so misses are visible
                    for p in q.points {
                        let s = CGPoint(x: p.x * CGFloat(outW) / CGFloat(w), y: p.y * CGFloat(outH) / CGFloat(h))
                        frameOut = CIImage(color: CIColor(red: 1, green: 0, blue: 1)).cropped(to: CGRect(x: s.x - 4, y: s.y - 4, width: 8, height: 8)).composited(over: frameOut)
                    }
                }
                ciContext.render(frameOut.cropped(to: CGRect(x: 0, y: 0, width: outW, height: outH)), to: ob)
                if startTime == nil { startTime = pts }
                let rel = CMTimeSubtract(pts, startTime!)
                while !wi.isReadyForMoreMediaData { usleep(2000) }
                ad.append(ob, withPresentationTime: rel)
            }
        }
        if index % 10 == 0, let cg = render(out.image) { collected.append((index, cg)) }
        index += 1
    }
    if let wi = writerInput, let wr = writer {
        wi.markAsFinished()
        let sem = DispatchSemaphore(value: 0)
        wr.finishWriting { sem.signal() }
        sem.wait()
    }

    // Metrics without ground truth
    let firstLock = drew.firstIndex(of: true) ?? drew.count
    var flickers = 0, longestGap = 0, gap = 0, covered = 0
    for i in firstLock..<drew.count {
        if drew[i] { covered += 1; gap = 0 } else { gap += 1; longestGap = max(longestGap, gap) }
        if i > firstLock && drew[i - 1] && !drew[i] { flickers += 1 }
    }
    // Jitter proxy: second difference of corners over consecutive drawn frames.
    var accel: [Double] = []
    if drew.count > 2 {
        for i in 2..<drew.count where drew[i] && drew[i - 1] && drew[i - 2] {
            if let a = quads[i - 2], let b = quads[i - 1], let c = quads[i] {
                let aa = (0..<4).map { k in dist(CGPoint(x: a[k].x - 2 * b[k].x + c[k].x, y: a[k].y - 2 * b[k].y + c[k].y), .zero) }
                accel.append(aa.reduce(0, +) / 4)
            }
        }
    }
    let metrics: [String: Any] = [
        "frames": drew.count,
        "first_lock_frame": firstLock,
        "coverage_after_lock_pct": r2(100 * Double(covered) / Double(max(1, drew.count - firstLock))),
        "flicker_events": flickers,
        "longest_gap_frames": longestGap,
        "corner_accel_median_px": r2(percentile(accel, 0.5)),
        "corner_accel_p95_px": r2(percentile(accel, 0.95)),
        "ms_per_frame_mean": r2(msPerFrame.reduce(0, +) / Double(max(1, msPerFrame.count))),
        "sources": sources,
        "reasons": reasons,
        "document_detector_errors": pipe.tracker.documentErrors,
        "frame": "\(Int(frameSize.width))x\(Int(frameSize.height))",
        "orientation_angle": r2(Double(angle))
    ]
    writeJSON(metrics, outPrefix + ".json")
    try? trace.write(toFile: outPrefix + "_trace.csv", atomically: true, encoding: .utf8)
    let pick = stride(from: 0, to: collected.count, by: max(1, collected.count / 12)).map { collected[$0] }
    for (i, cg) in pick.prefix(12) {
        thumbs.append(Thumb(index: i, image: cg, label: "#\(i) \(drew[i] ? "FAKE" : "real")",
                            tracked: drew[i] ? nil : quads[i], truth: nil, frameSize: frameSize))
    }
    if let sheet = contactSheet(thumbs, columns: 4, cellWidth: frameSize.width > frameSize.height ? 300 : 200) {
        writeJPEG(sheet, outPrefix + ".jpg")
    }
    print(String(data: try! JSONSerialization.data(withJSONObject: metrics, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)
} else {
    fail("unknown mode \(mode)")
}
