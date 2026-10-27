import SwiftUI
import AVFoundation
import AudioToolbox
import Combine
import Photos
import UIKit

/// Glue between the UI, the settings, the camera engine and the triggers.
final class CameraModel: ObservableObject {
    let engine = CameraEngine()
    let settings = AppSettings.shared
    private let snap = SnapDetector()
    private var bag = Set<AnyCancellable>()
    weak var previewView: PreviewView?

    @Published var showSettings = false
    @Published var armed = true
    @Published var status = CameraEngine.EngineStatus(tracking: "searching", source: "none", reason: "",
                                                      cardness: 0, armed: true, fps: 0)
    @Published var snapLevel: Float = 0
    @Published var lastPhoto: UIImage?
    @Published var flash = false
    @Published var focusPoint: CGPoint?
    @Published var settingsVisible = false { didSet { updateMeter() } }

    // Camera chrome state (mostly cosmetic, like the built-in camera's toggles)
    @Published var position: AVCaptureDevice.Position = .front
    @Published var zoomStops: [CameraEngine.ZoomStop] = []
    @Published var zoomIndex = 0
    @Published var flashMode = 0          // 0 auto, 1 on, 2 off
    @Published var liveOn = true
    @Published var centerStageOn = true
    @Published var modeIndex = 4          // index into `modes`
    let modes = ["TIME-LAPSE", "SLO-MO", "CINEMATIC", "VIDEO", "PHOTO", "PORTRAIT", "PANO"]
    @Published var recording = false
    var isVideoMode: Bool { ["TIME-LAPSE", "SLO-MO", "CINEMATIC", "VIDEO"].contains(modes[modeIndex]) }

    /// Read from the audio thread; only feed the level meter when someone can see it.
    private var meterActive = false
    private func updateMeter() { meterActive = settings.debug || settingsVisible }

    /// Screen size and the 3:4 photo area on it (set by the layout).
    var viewSize = CGSize(width: 402, height: 874)
    var photoRect = CGRect(x: 0, y: 128, width: 402, height: 536)

    init() {
        engine.onStatus = { [weak self] st in self?.status = st }
        engine.onCameraChanged = { [weak self] pos, stops in
            guard let self = self else { return }
            self.position = pos
            self.zoomStops = stops
            self.zoomIndex = stops.firstIndex { $0.label == "1" } ?? 0
        }
        // Layout checks in the simulator (no camera there): show the front or back camera controls.
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-demoBack") {
            position = .back
            zoomStops = [(".5", 1.0), ("1", 2.0), ("2", 4.0), ("4", 8.0), ("8", 16.0)].map { CameraEngine.ZoomStop(label: $0.0, factor: $0.1) }
            zoomIndex = 1
        } else if args.contains("-demoFront") {
            position = .front
        }
        snap.onSnap = { [weak self] in self?.trigger(source: "snap") }
        snap.onLevel = { [weak self] lvl in
            guard let self = self, self.meterActive else { return }
            DispatchQueue.main.async { self.snapLevel = max(lvl, self.snapLevel * 0.85) }
        }

        settings.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.applySettings() }
            }
            .store(in: &bag)
        applySettings()
        loadLatestLibraryPhoto()
    }

    func applySettings() {
        engine.setConfig(settings.renderConfig)
        if settings.autoRearm {
            engine.onAutoRearm = { [weak self] in self?.rearm() }
        } else {
            engine.onAutoRearm = nil
        }
        snap.sensitivity = settings.snapSensitivity
        if settings.snapEnabled { snap.start() } else { snap.stop() }
        let buttonsOn = settings.buttonsEnabled
        MainActor.assumeIsolated { previewView?.setHardwareButtonsEnabled(buttonsOn) }
        updateMeter()
        refreshFace()
    }

    private var lastFaceKey = ""
    func refreshFace() {
        let key = "\(settings.rank)-\(settings.suit)-\(settings.importedFace ?? "")"
        guard key != lastFaceKey else { return }
        lastFaceKey = key
        if let name = settings.importedFace, let cg = CardImporter.loadFace(named: name) {
            engine.setFace(cg)
        } else if let real = RealFaces.face(rank: settings.rank, suit: settings.suit),
                  let cg = CardFaceRenderer.render(face: real) {
            // traced from your own deck: exactly the real card's pips and indices
            engine.setFace(cg)
        } else if let cg = CardFaceRenderer.render(rank: settings.rank, suit: settings.suit) {
            engine.setFace(cg)
        }
    }

    func start() {
        AVCaptureDeviceAuthorization.request { [weak self] ok in
            guard ok, let self = self else { return }
            self.engine.start()
        }
    }

    func stop() {
        engine.stop()
    }

    // MARK: - Triggers

    func trigger(source: String) {
        guard engine.isArmed else { return }
        // A snap only counts while the fake card is actually on screen, so a stray noise with no
        // card in view can't quietly disarm the trick. (Buttons always work.)
        if source == "snap" && !engine.isShowingFake { return }
        engine.setArmed(false)
        armed = false
        if settings.haptics {
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
        }
    }

    func rearm() {
        engine.setArmed(true)
        armed = true
        if settings.haptics {
            UISelectionFeedbackGenerator().selectionChanged()
        }
    }

    func hardwareButtonPressed() {
        guard settings.buttonsEnabled else { return }
        trigger(source: "button")
    }

    /// `p` in view points, `size` the full preview size.
    func handleTap(at p: CGPoint, in size: CGSize) {
        withAnimation(.easeOut(duration: 0.15)) { focusPoint = p }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) { [weak self] in
            withAnimation(.easeIn(duration: 0.25)) { self?.focusPoint = nil }
        }
        engine.focus(at: engine.normalizedPoint(viewPoint: p, viewSize: size))
        if settings.tapEnabled { trigger(source: "tap") }
    }

    // MARK: - Camera chrome

    func flipCamera() {
        engine.switchCamera()
    }

    func selectZoom(_ i: Int) {
        guard zoomStops.indices.contains(i) else { return }
        zoomIndex = i
        engine.setZoom(zoomStops[i].factor)
        UISelectionFeedbackGenerator().selectionChanged()
    }

    func selectMode(_ i: Int) {
        guard modes.indices.contains(i) else { return }
        recording = false
        withAnimation(.spring(response: 0.3, dampingFraction: 0.85)) { modeIndex = i }
        UISelectionFeedbackGenerator().selectionChanged()
    }

    func openPhotos() {
        if let url = URL(string: "photos-redirect://") {
            UIApplication.shared.open(url)
        }
    }

    private func loadLatestLibraryPhoto() {
        PHPhotoLibrary.requestAuthorization(for: .readWrite) { [weak self] status in
            guard status == .authorized || status == .limited else { return }
            let opts = PHFetchOptions()
            opts.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
            opts.fetchLimit = 1
            guard let asset = PHAsset.fetchAssets(with: .image, options: opts).firstObject else { return }
            let ro = PHImageRequestOptions()
            ro.deliveryMode = .highQualityFormat
            ro.isNetworkAccessAllowed = true
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: 200, height: 200),
                                                  contentMode: .aspectFill, options: ro) { img, _ in
                guard let img = img else { return }
                DispatchQueue.main.async {
                    if self?.lastPhoto == nil { self?.lastPhoto = img }
                }
            }
        }
    }

    // MARK: - Shutter

    func takePhoto() {
        if isVideoMode {
            withAnimation(.easeInOut(duration: 0.2)) { recording.toggle() }
            return
        }
        AudioServicesPlaySystemSound(1108)   // the standard camera shutter sound
        withAnimation(.linear(duration: 0.05)) { flash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { [weak self] in
            withAnimation(.easeOut(duration: 0.2)) { self?.flash = false }
        }
        guard let img = engine.snapshot(crop: engine.normalizedRect(viewRect: photoRect, viewSize: viewSize)) else { return }
        withAnimation(.easeOut(duration: 0.25)) { lastPhoto = img }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else { return }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: img)
            } completionHandler: { _, err in
                if let err = err { print("Save failed: \(err)") }
            }
        }
    }
}

/// Camera permission helper.
enum AVCaptureDeviceAuthorization {
    static func request(_ done: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            done(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { ok in DispatchQueue.main.async { done(ok) } }
        default:
            done(false)
        }
    }
}
