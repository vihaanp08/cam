import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import Metal
import QuartzCore
import UIKit

/// Owns the camera and runs the per-frame pipeline:
/// camera frame -> find card -> measure light -> paint fake face -> hand image to the screen.
final class CameraEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let metalDevice: MTLDevice
    let ciContext: CIContext

    private let sessionQueue = DispatchQueue(label: "cam.session")
    private let videoQueue = DispatchQueue(label: "cam.video", qos: .userInteractive)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var input: AVCaptureDeviceInput?
    private(set) var position: AVCaptureDevice.Position = .front

    private let tracker = CardTracker()
    private let compositor: Compositor

    // Shared state (guarded by `lock`)
    private let lock = NSLock()
    private var _latest: CIImage?
    private var _config = RenderConfig()
    private var _albedo: CIImage?
    private var _albedoMirrored: CIImage?
    private var _armed = true
    private var _gateUntilReady = true
    private var _gateStart: CFTimeInterval = CACurrentMediaTime()
    private var _gateImage: CIImage?

    // Video-queue-only state
    private var lastAnalysis: Compositor.Analysis?
    private var framesSinceOverlay = 1000
    private var lastCardSeen: CFTimeInterval = 0
    private var gateFade = 0
    private var lastStatusSent: CFTimeInterval = 0
    private var lastQuadForBlur: Quad?

    /// Main-thread callbacks
    var onStatus: ((EngineStatus) -> Void)?
    /// Reassigned from main on every settings change and read on the video queue, so it is lock-guarded.
    private var _onAutoRearm: (() -> Void)?
    var onAutoRearm: (() -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return _onAutoRearm }
        set { lock.lock(); _onAutoRearm = newValue; lock.unlock() }
    }

    struct EngineStatus {
        var tracking: String
        var cardness: Double
        var armed: Bool
    }

    override init() {
        let device = MTLCreateSystemDefaultDevice()!
        let ctx = CIContext(mtlDevice: device, options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            .cacheIntermediates: false
        ])
        metalDevice = device
        ciContext = ctx
        compositor = Compositor(context: ctx)
        super.init()
    }

    // MARK: - Public controls (any thread)

    var latestImage: CIImage? {
        lock.lock(); defer { lock.unlock() }
        return _latest
    }

    var isArmed: Bool {
        lock.lock(); defer { lock.unlock() }
        return _armed
    }

    func setArmed(_ armed: Bool) {
        lock.lock(); _armed = armed; lock.unlock()
    }

    func setConfig(_ cfg: RenderConfig) {
        lock.lock()
        let mirrorChanged = cfg.mirror != _config.mirror
        _config = cfg
        lock.unlock()
        if mirrorChanged { sessionQueue.async { self.applyConnectionSettings() } }
    }

    func setFace(_ cg: CGImage) {
        var face = CIImage(cgImage: cg)
        // Normalise to the standard card size.
        let sx = Compositor.cardSize.width / face.extent.width
        let sy = Compositor.cardSize.height / face.extent.height
        face = face.transformed(by: CGAffineTransform(scaleX: sx, y: sy)).cropped(to: Compositor.cardRect)
        let albedo = Compositor.prepareAlbedo(face)
        let mirrored = albedo
            .transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -Compositor.cardSize.width, y: 0))
            .cropped(to: Compositor.cardRect)
        lock.lock()
        _albedo = albedo
        _albedoMirrored = mirrored
        lock.unlock()
    }

    /// Hide the live view behind `cover` until the card is locked (or a short timeout),
    /// so the real face is never seen while tracking starts up.
    private func beginGate(cover: CIImage?) {
        lock.lock()
        _gateUntilReady = true
        _gateStart = CACurrentMediaTime()
        _gateImage = cover
        lock.unlock()
    }

    // MARK: - Session

    func start() {
        sessionQueue.async {
            if self.session.inputs.isEmpty { self.configure(position: self.position) }
            self.beginGate(cover: nil)
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    func switchCamera() {
        // Freeze a blurred copy of the current view as cover, like a normal camera flip.
        let cover = blurredSnapshot()
        beginGate(cover: cover)
        sessionQueue.async {
            let newPos: AVCaptureDevice.Position = self.position == .front ? .back : .front
            self.configure(position: newPos)
            self.videoQueue.async {
                self.tracker.reset()
                self.compositor.resetTemporalState()
                self.lastAnalysis = nil
            }
        }
    }

    private func configure(position: AVCaptureDevice.Position) {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.automaticallyConfiguresApplicationAudioSession = false
        if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        } else {
            session.sessionPreset = .high
        }

        if let old = input { session.removeInput(old) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let newInput = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(newInput) else { return }
        session.addInput(newInput)
        input = newInput
        self.position = position

        do {
            try device.lockForConfiguration()
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            if device.isExposureModeSupported(.continuousAutoExposure) { device.exposureMode = .continuousAutoExposure }
            if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { device.whiteBalanceMode = .continuousAutoWhiteBalance }
            device.unlockForConfiguration()
        } catch {
            print("Device config error: \(error)")
        }

        if !session.outputs.contains(videoOutput) {
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        }
        applyConnectionSettings()
    }

    private func applyConnectionSettings() {
        guard let conn = videoOutput.connection(with: .video) else { return }
        if conn.isVideoRotationAngleSupported(90) { conn.videoRotationAngle = 90 }
        lock.lock(); let mirror = _config.mirror; lock.unlock()
        if conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = mirror
        }
    }

    // MARK: - Snapshots

    private func blurredSnapshot() -> CIImage? {
        guard let img = latestImage else { return nil }
        let ext = img.extent
        let small = img.transformed(by: CGAffineTransform(scaleX: 0.25, y: 0.25))
        let blurred = small.clampedToExtent().applyingGaussianBlur(sigma: 10).cropped(to: small.extent)
        guard let cg = ciContext.createCGImage(blurred, from: small.extent) else { return nil }
        return CIImage(cgImage: cg)
            .transformed(by: CGAffineTransform(scaleX: ext.width / small.extent.width, y: ext.height / small.extent.height))
            .cropped(to: ext)
    }

    /// The current on-screen picture (fake card included), cropped to the visible aspect.
    func snapshot(visibleAspect: CGFloat) -> UIImage? {
        guard let img = latestImage else { return nil }
        let ext = img.extent
        var crop = ext
        if ext.width / ext.height < visibleAspect {
            let h = ext.width / visibleAspect
            crop = CGRect(x: ext.minX, y: ext.midY - h / 2, width: ext.width, height: h)
        } else {
            let w = ext.height * visibleAspect
            crop = CGRect(x: ext.midX - w / 2, y: ext.minY, width: w, height: ext.height)
        }
        guard let cg = ciContext.createCGImage(img, from: crop,
                                               format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else { return nil }
        return UIImage(cgImage: cg)
    }

    // MARK: - Per-frame pipeline

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let frame = CIImage(cvPixelBuffer: pb)

        lock.lock()
        let cfg = _config
        let armed = _armed
        let albedo = cfg.mirror ? _albedoMirrored : _albedo
        let gating = _gateUntilReady
        let gateStart = _gateStart
        let gateImage = _gateImage
        lock.unlock()

        let state = tracker.process(pb)
        let now = CACurrentMediaTime()
        var out = frame
        var drewOverlay = false
        var trackingLabel = "searching"
        var cardness = 0.0

        switch state {
        case .locked(let q):
            trackingLabel = "locked"
            lastCardSeen = now
            lastQuadForBlur = q
            if armed, let albedo = albedo, let a = compositor.analyze(frame: frame, quad: q) {
                cardness = a.cardness
                let isCard = a.cardness > 0.45 && (a.faceUp || !cfg.facesOnly)
                if isCard {
                    out = compositor.overlay(frame: frame, quad: q, albedo: albedo, analysis: a, cfg: cfg)
                    lastAnalysis = a
                    drewOverlay = true
                } else if cfg.debug {
                    out = compositor.debugCorners(frame, quad: q, color: CIColor(red: 1, green: 0.2, blue: 0.2))
                }
            } else if cfg.debug {
                out = compositor.debugCorners(frame, quad: q, color: CIColor(red: 0.3, green: 0.6, blue: 1))
            }

        case .coasting(let q, let n):
            trackingLabel = "coasting \(n)"
            lastQuadForBlur = q
            if armed, framesSinceOverlay < 30, let albedo = albedo, let a = lastAnalysis {
                if n <= 8 {
                    // Brief dropout: keep the fake card in place (lighting from the last good frame).
                    out = compositor.overlay(frame: frame, quad: q, albedo: albedo, analysis: a, cfg: cfg)
                    drewOverlay = true
                } else {
                    // Card moving too fast to track: blur where it was, like natural motion blur,
                    // fading out over the next ~half second.
                    let strength = max(0, 1 - Double(n - 8) / 16)
                    if strength > 0 { out = blurRegion(frame, around: q, strength: strength) }
                }
            }

        case .searching:
            trackingLabel = "searching"
        }

        framesSinceOverlay = drewOverlay ? 0 : framesSinceOverlay + 1

        // Start-up / camera-flip cover.
        if gating {
            let elapsed = now - gateStart
            let ready: Bool
            if armed {
                ready = (drewOverlay && tracker.stableFrames >= 3) || elapsed > 2.0
            } else {
                ready = elapsed > 0.35
            }
            if ready {
                lock.lock(); _gateUntilReady = false; lock.unlock()
                gateFade = 6
            }
            let cover = gateImage ?? CIImage(color: .black).cropped(to: frame.extent)
            if !ready { out = cover.cropped(to: frame.extent) }
        }
        if gateFade > 0, let cover = gateImage {
            let t = CGFloat(gateFade) / 6
            out = fade(cover, over: out, alpha: t, extent: frame.extent)
            gateFade -= 1
        } else if gateFade > 0 {
            let black = CIImage(color: .black).cropped(to: frame.extent)
            out = fade(black, over: out, alpha: CGFloat(gateFade) / 6, extent: frame.extent)
            gateFade -= 1
        }

        // Auto re-arm once the card has been out of view for a couple of seconds.
        if !armed, let cb = onAutoRearm, now - lastCardSeen > 2.0, lastCardSeen > 0 {
            lastCardSeen = 0
            DispatchQueue.main.async { cb() }
        }

        lock.lock()
        _latest = out.cropped(to: frame.extent)
        lock.unlock()

        if now - lastStatusSent > 0.2, let cb = onStatus {
            lastStatusSent = now
            let st = EngineStatus(tracking: trackingLabel, cardness: cardness, armed: armed)
            DispatchQueue.main.async { cb(st) }
        }
    }

    private func fade(_ top: CIImage, over base: CIImage, alpha: CGFloat, extent: CGRect) -> CIImage {
        // Cross-dissolve (both images are opaque): base * (1 - alpha) + top * alpha.
        let d = CIFilter.dissolveTransition()
        d.inputImage = base
        d.targetImage = top.cropped(to: extent)
        d.time = Float(alpha)
        return (d.outputImage ?? base).cropped(to: extent)
    }

    private func blurRegion(_ frame: CIImage, around q: Quad, strength: Double) -> CIImage {
        let box = q.boundingBox.insetBy(dx: -q.width * 0.25, dy: -q.height * 0.25).intersection(frame.extent)
        guard !box.isNull, box.width > 4, box.height > 4 else { return frame }
        let sigma = 22 * strength
        let blurred = frame.clampedToExtent().applyingGaussianBlur(sigma: sigma).cropped(to: box)
        // Feathered mask so the blurred patch has no hard edge.
        let r = CIFilter.radialGradient()
        r.center = CGPoint(x: box.midX, y: box.midY)
        r.radius0 = Float(min(box.width, box.height) * 0.35)
        r.radius1 = Float(max(box.width, box.height) * 0.6)
        r.color0 = CIColor.white
        r.color1 = CIColor.black
        guard let mask = r.outputImage?.cropped(to: box) else { return frame }
        let b = CIFilter.blendWithMask()
        b.inputImage = blurred
        b.backgroundImage = frame
        b.maskImage = mask
        return (b.outputImage ?? frame).cropped(to: frame.extent)
    }
}
