import AVFoundation
import CoreMotion
import CoreImage
import CoreImage.CIFilterBuiltins
import Metal
import QuartzCore
import UIKit
import Vision

/// Owns the camera and feeds each frame through the FramePipeline (find card, paint fake face),
/// then hands the result to the screen.
final class CameraEngine: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let metalDevice: MTLDevice
    let ciContext: CIContext

    private let sessionQueue = DispatchQueue(label: "cam.session")
    private let videoQueue = DispatchQueue(label: "cam.video", qos: .userInteractive)
    private let videoOutput = AVCaptureVideoDataOutput()
    private var input: AVCaptureDeviceInput?
    private var device: AVCaptureDevice?

    private let pipeline: FramePipeline

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
    private var _position: AVCaptureDevice.Position = .front
    private var _onAutoRearm: (() -> Void)?
    private var _lastFakeShown: CFTimeInterval = 0
    /// Rotation that makes this camera's frames upright with the phone held in portrait.
    /// 90 for sensors mounted sideways (most), 0 for the newer portrait-mounted square selfie sensor.
    private var _portraitAngle: CGFloat = 90
    private var _learnedAngles: [Int: CGFloat] = CameraEngine.loadLearnedAngles()   // by camera position
    /// Phone clearly held upright (from the motion sensor). Only then is the system's rotation
    /// trustworthy: lying down, face up or face down it can be anything.
    private var _deviceIsPortrait = false
    private let motion = CMMotionManager()

    /// The camera's mounting never changes, so a rotation learned once (phone upright) is kept for
    /// good, even across launches.
    private static func loadLearnedAngles() -> [Int: CGFloat] {
        var out: [Int: CGFloat] = [:]
        for pos in [AVCaptureDevice.Position.front, .back] {
            if let v = UserDefaults.standard.object(forKey: "rotationAngle\(pos.rawValue)") as? Double { out[pos.rawValue] = CGFloat(v) }
        }
        return out
    }

    // Session-queue-only state
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?
    private var orientationObserver: NSObjectProtocol?

    // Video-queue-only state
    private var gateFade = 0
    private var lastStatusSent: CFTimeInterval = 0
    private var lastFrameTime: Double = -1
    private var fps: Double = 30
    private var rearmFiredFor: Double = -2
    private var faceCheckCounter = 0
    private var faceTurns: [Int] = []

    /// Main-thread callback with tracking info for the debug view.
    var onStatus: ((EngineStatus) -> Void)?
    /// Main-thread callback when the camera (and its zoom options) changes.
    var onCameraChanged: ((AVCaptureDevice.Position, [ZoomStop]) -> Void)?

    var onAutoRearm: (() -> Void)? {
        get { lock.lock(); defer { lock.unlock() }; return _onAutoRearm }
        set { lock.lock(); _onAutoRearm = newValue; lock.unlock() }
    }

    struct EngineStatus {
        var tracking: String
        var source: String
        var reason: String
        var cardness: Double
        var armed: Bool
        var fps: Double
    }

    /// A zoom button: its label and the device zoom factor it maps to.
    struct ZoomStop: Equatable {
        var label: String
        var factor: CGFloat
    }

    override init() {
        let device = MTLCreateSystemDefaultDevice()!
        let ctx = CIContext(mtlDevice: device, options: [
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            .cacheIntermediates: false
        ])
        metalDevice = device
        ciContext = ctx
        pipeline = FramePipeline(context: ctx)
        super.init()
    }

    // MARK: - Public controls (any thread)

    var latestImage: CIImage? {
        lock.lock(); defer { lock.unlock() }
        return _latest
    }

    /// The fake card is on screen right now (a reveal only makes sense then).
    var isShowingFake: Bool {
        lock.lock(); defer { lock.unlock() }
        return CACurrentMediaTime() - _lastFakeShown < 0.4
    }

    var isArmed: Bool {
        lock.lock(); defer { lock.unlock() }
        return _armed
    }

    var position: AVCaptureDevice.Position {
        lock.lock(); defer { lock.unlock() }
        return _position
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
        let a = FramePipeline.makeAlbedo(cg)
        lock.lock()
        _albedo = a.normal
        _albedoMirrored = a.mirrored
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
        if !motion.isDeviceMotionActive && motion.isDeviceMotionAvailable {
            // Learn the camera rotation only while the phone is clearly upright in portrait (gravity
            // pointing down the phone). Lying down or flat, the system's angle is unreliable, and
            // following it is what turned the picture sideways and kept resetting the tracker.
            motion.deviceMotionUpdateInterval = 0.1
            motion.startDeviceMotionUpdates(to: OperationQueue()) { [weak self] m, _ in
                guard let self = self, let g = m?.gravity else { return }
                let upright = g.y < -0.8 && abs(g.x) < 0.45 && abs(g.z) < 0.6
                self.lock.lock()
                let was = self._deviceIsPortrait
                self._deviceIsPortrait = upright
                self.lock.unlock()
                if upright && !was { self.sessionQueue.async { self.learnPortraitAngle() } }
            }
        }
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
            self.videoQueue.async { self.pipeline.reset() }
        }
    }

    func setZoom(_ factor: CGFloat, animated: Bool = true) {
        sessionQueue.async {
            guard let d = self.device else { return }
            let f = min(max(factor, d.minAvailableVideoZoomFactor), min(d.maxAvailableVideoZoomFactor, 32))
            do {
                try d.lockForConfiguration()
                if animated { d.ramp(toVideoZoomFactor: f, withRate: 12) } else { d.videoZoomFactor = f }
                d.unlockForConfiguration()
            } catch {}
            self.videoQueue.async { self.pipeline.reset() }
        }
    }

    /// Point in normalized frame coordinates (0...1, origin top-left of the upright, as-displayed frame).
    func focus(at p: CGPoint) {
        sessionQueue.async {
            guard let d = self.device else { return }
            // Device coordinates are in the sensor's own orientation: undo the rotation.
            var x = p.x
            if self.position == .front && self.currentConfig.mirror { x = 1 - x }
            self.lock.lock(); let angle = self._portraitAngle; self.lock.unlock()
            let devicePoint: CGPoint
            switch Int(angle.rounded()) % 360 {
            case 0: devicePoint = CGPoint(x: x, y: p.y)
            case 180: devicePoint = CGPoint(x: 1 - x, y: 1 - p.y)
            case 270: devicePoint = CGPoint(x: 1 - p.y, y: x)
            default: devicePoint = CGPoint(x: p.y, y: 1 - x)
            }
            do {
                try d.lockForConfiguration()
                // Keep focus/exposure continuous (just biased to the tapped point) so tracking never
                // gets stuck with a locked, wrong focus.
                if d.isFocusPointOfInterestSupported, d.isFocusModeSupported(.continuousAutoFocus) {
                    d.focusPointOfInterest = devicePoint; d.focusMode = .continuousAutoFocus
                }
                if d.isExposurePointOfInterestSupported, d.isExposureModeSupported(.continuousAutoExposure) {
                    d.exposurePointOfInterest = devicePoint; d.exposureMode = .continuousAutoExposure
                }
                d.unlockForConfiguration()
            } catch {}
        }
    }

    private var currentConfig: RenderConfig {
        lock.lock(); defer { lock.unlock() }
        return _config
    }

    private func configure(position: AVCaptureDevice.Position) {
        session.beginConfiguration()
        session.automaticallyConfiguresApplicationAudioSession = false
        if session.canSetSessionPreset(.hd1920x1080) {
            session.sessionPreset = .hd1920x1080
        } else {
            session.sessionPreset = .high
        }

        if let old = input { session.removeInput(old) }
        let d: AVCaptureDevice?
        if position == .back {
            d = AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        } else {
            d = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front)
        }
        guard let dev = d, let newInput = try? AVCaptureDeviceInput(device: dev), session.canAddInput(newInput) else {
            session.commitConfiguration()
            return
        }
        session.addInput(newInput)
        input = newInput
        device = dev
        lock.lock(); _position = position; lock.unlock()

        // Which way up are this camera's frames? Newer phones mount the selfie sensor in portrait,
        // older ones sideways, so ask the system instead of assuming 90 degrees.
        rotationObservation = nil
        let coordinator = AVCaptureDevice.RotationCoordinator(device: dev, previewLayer: nil)
        rotationCoordinator = coordinator
        lock.lock(); let portraitNow = _deviceIsPortrait; lock.unlock()
        let angle = coordinator.videoRotationAngleForHorizonLevelCapture
        lock.lock()
        if portraitNow {
            // Held upright: the system's horizon-level angle is the one we need.
            _portraitAngle = angle
            _learnedAngles[position.rawValue] = angle
            UserDefaults.standard.set(Double(angle), forKey: "rotationAngle\(position.rawValue)")
        } else if let known = _learnedAngles[position.rawValue] {
            _portraitAngle = known
        } else {
            _portraitAngle = position == .back ? 90 : angle
        }
        lock.unlock()
        rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.new]) { [weak self] _, _ in
            self?.sessionQueue.async { self?.learnPortraitAngle() }
        }

        if !session.outputs.contains(videoOutput) {
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
            if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }
        }
        applyConnectionSettings()
        session.commitConfiguration()

        // Zoom stops: "1x" is the main wide lens; on multi-camera devices that is a switch-over factor.
        var stops: [ZoomStop] = []
        var oneX: CGFloat = 1
        do {
            try dev.lockForConfiguration()
            dev.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            dev.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
            if dev.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { dev.whiteBalanceMode = .continuousAutoWhiteBalance }
            if position == .back {
                let switches = dev.virtualDeviceSwitchOverVideoZoomFactors.map { CGFloat(truncating: $0) }
                if let first = switches.first, dev.isVirtualDevice, dev.constituentDevices.first?.deviceType == .builtInUltraWideCamera {
                    oneX = first
                    stops.append(ZoomStop(label: ".5", factor: 1))
                }
                stops.append(ZoomStop(label: "1", factor: oneX))
                for m: CGFloat in [2, 4, 8] where oneX * m <= dev.maxAvailableVideoZoomFactor {
                    stops.append(ZoomStop(label: "\(Int(m))", factor: oneX * m))
                }
            }
            dev.videoZoomFactor = oneX
            dev.unlockForConfiguration()
        } catch {
            print("Device config error: \(error)")
        }
        let cb = onCameraChanged
        DispatchQueue.main.async { cb?(position, stops) }
    }

    /// If the face in the picture is turned a quarter, a half or three quarters round, the picture is
    /// on its side: turn the camera's rotation by that much. (Whichever way round the turn has to be
    /// applied, at most two steps put the face upright.) The result is remembered.
    private func checkFaceTurn(_ pb: CVPixelBuffer) {
        let req = VNDetectFaceRectanglesRequest()
        req.revision = VNDetectFaceRectanglesRequestRevision3
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up)
        try? handler.perform([req])
        guard let face = req.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
              face.boundingBox.width > 0.12 || face.boundingBox.height > 0.12,
              let roll = face.roll?.doubleValue else { return }
        let quarters = Int((roll * 180 / .pi / 90).rounded())
        let turn = ((quarters % 4) + 4) % 4 * 90
        faceTurns.append(turn)
        if faceTurns.count > 4 { faceTurns.removeFirst() }
        guard turn != 0, faceTurns.count == 4, faceTurns.filter({ $0 == turn }).count >= 3 else { return }
        faceTurns = []
        sessionQueue.async {
            self.lock.lock()
            let angle = (Int(self._portraitAngle.rounded()) + turn) % 360
            self._portraitAngle = CGFloat(angle)
            self._learnedAngles[self._position.rawValue] = CGFloat(angle)
            UserDefaults.standard.set(Double(angle), forKey: "rotationAngle\(self._position.rawValue)")
            self.lock.unlock()
            self.applyConnectionSettings()
            self.videoQueue.async { self.pipeline.reset() }
        }
    }

    /// While the phone is upright, the system's horizon-level angle is exactly the rotation the
    /// portrait-only app needs; keep it (session queue).
    private func learnPortraitAngle() {
        guard let c = rotationCoordinator else { return }
        lock.lock()
        let portrait = _deviceIsPortrait
        let changed = portrait && _portraitAngle != c.videoRotationAngleForHorizonLevelCapture
        if portrait {
            _portraitAngle = c.videoRotationAngleForHorizonLevelCapture
            _learnedAngles[_position.rawValue] = _portraitAngle
            UserDefaults.standard.set(Double(_portraitAngle), forKey: "rotationAngle\(_position.rawValue)")
        }
        lock.unlock()
        if changed {
            applyConnectionSettings()
            videoQueue.async { self.pipeline.reset() }
        }
    }

    private func applyConnectionSettings() {
        guard let conn = videoOutput.connection(with: .video) else { return }
        lock.lock(); let angle = _portraitAngle; lock.unlock()
        if conn.isVideoRotationAngleSupported(angle) { conn.videoRotationAngle = angle }
        lock.lock(); let mirror = _config.mirror && _position == .front; lock.unlock()
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

    /// Maps a rectangle in the full-screen preview (aspect-filled) to normalized frame coordinates
    /// (origin top-left).
    func normalizedRect(viewRect: CGRect, viewSize: CGSize) -> CGRect {
        guard let ext = latestImage?.extent, ext.width > 0, viewSize.width > 0 else {
            return CGRect(x: 0, y: 0, width: 1, height: 1)
        }
        let s = max(viewSize.width / ext.width, viewSize.height / ext.height)
        let dw = ext.width * s, dh = ext.height * s
        let ox = (viewSize.width - dw) / 2, oy = (viewSize.height - dh) / 2
        let r = CGRect(x: (viewRect.minX - ox) / dw, y: (viewRect.minY - oy) / dh,
                       width: viewRect.width / dw, height: viewRect.height / dh)
        return r.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    /// Maps a point in the full-screen preview to normalized frame coordinates (origin top-left), clamped.
    func normalizedPoint(viewPoint p: CGPoint, viewSize: CGSize) -> CGPoint {
        guard let ext = latestImage?.extent, ext.width > 0, viewSize.width > 0 else { return CGPoint(x: 0.5, y: 0.5) }
        let s = max(viewSize.width / ext.width, viewSize.height / ext.height)
        let dw = ext.width * s, dh = ext.height * s
        let ox = (viewSize.width - dw) / 2, oy = (viewSize.height - dh) / 2
        return CGPoint(x: min(max((p.x - ox) / dw, 0), 1), y: min(max((p.y - oy) / dh, 0), 1))
    }

    /// The picture currently on screen (fake card included), cropped to `crop`
    /// (normalized, origin top-left). Selfies are saved un-mirrored, like the built-in camera.
    func snapshot(crop: CGRect) -> UIImage? {
        guard var img = latestImage else { return nil }
        let ext = img.extent
        let r = CGRect(x: ext.minX + crop.minX * ext.width,
                       y: ext.minY + (1 - crop.maxY) * ext.height,
                       width: crop.width * ext.width,
                       height: crop.height * ext.height).integral.intersection(ext)
        lock.lock(); let unmirror = _config.mirror && _position == .front; lock.unlock()
        img = img.cropped(to: r)
        if unmirror {
            img = img.transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -(r.minX * 2 + r.width), y: 0))
        }
        guard let cg = ciContext.createCGImage(img, from: img.extent,
                                               format: .RGBA8,
                                               colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else { return nil }
        return UIImage(cgImage: cg)
    }

    // MARK: - Per-frame pipeline

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

        lock.lock()
        let cfg = _config
        let armed = _armed
        let mirrored = cfg.mirror && _position == .front
        let albedo = mirrored ? _albedoMirrored : _albedo
        let gating = _gateUntilReady
        let gateStart = _gateStart
        let gateImage = _gateImage
        let autoRearm = _onAutoRearm
        lock.unlock()

        if lastFrameTime > 0, t > lastFrameTime { fps = 0.9 * fps + 0.1 / (t - lastFrameTime) }
        lastFrameTime = t

        let result = pipeline.process(pb, time: t, cfg: cfg, armed: armed, albedo: albedo)
        var out = result.image
        let extent = out.extent
        let now = CACurrentMediaTime()

        // Start-up / camera-flip cover.
        if gating {
            let elapsed = now - gateStart
            let ready: Bool
            if armed {
                ready = (result.drewOverlay && pipeline.tracker.stableFrames >= 2) || elapsed > 1.2
            } else {
                ready = elapsed > 0.35
            }
            if ready {
                lock.lock(); _gateUntilReady = false; lock.unlock()
                gateFade = 6
            } else {
                out = (gateImage ?? CIImage(color: .black)).cropped(to: extent)
            }
        }
        if gateFade > 0 {
            let cover = (gateImage ?? CIImage(color: .black)).cropped(to: extent)
            out = fade(cover, over: out, alpha: CGFloat(gateFade) / 6, extent: extent)
            gateFade -= 1
        }

        // Selfie camera: the performer's face shows which way up the picture should be, whatever way
        // the phone is held (lying down, flat, at any angle). Twice a second, check it.
        faceCheckCounter += 1
        lock.lock(); let isFront = _position == .front; lock.unlock()
        if isFront && faceCheckCounter % 15 == 0 { checkFaceTurn(pb) }

        // Auto re-arm once the card has been out of view for a couple of seconds.
        if !armed, let cb = autoRearm, pipeline.lastCardSeen > 0, t - pipeline.lastCardSeen > 2.0,
           rearmFiredFor != pipeline.lastCardSeen {
            rearmFiredFor = pipeline.lastCardSeen
            DispatchQueue.main.async { cb() }
        }

        lock.lock()
        _latest = out
        if result.drewOverlay { _lastFakeShown = now }
        lock.unlock()

        if now - lastStatusSent > 0.2, let cb = onStatus {
            lastStatusSent = now
            let s = result.status
            let st = EngineStatus(tracking: s.tracking, source: s.source, reason: s.reason,
                                  cardness: s.cardness, armed: armed, fps: fps)
            DispatchQueue.main.async { cb(st) }
        }
    }

    private func fade(_ top: CIImage, over base: CIImage, alpha: CGFloat, extent: CGRect) -> CIImage {
        let d = CIFilter.dissolveTransition()
        d.inputImage = base
        d.targetImage = top.cropped(to: extent)
        d.time = Float(alpha)
        return (d.outputImage ?? base).cropped(to: extent)
    }
}
