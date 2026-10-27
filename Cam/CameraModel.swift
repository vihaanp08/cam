import SwiftUI
import AVFoundation
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
    @Published var trackingStatus = "searching"
    @Published var cardness = 0.0
    @Published var snapLevel: Float = 0
    @Published var lastPhoto: UIImage?
    @Published var flash = false
    @Published var focusPoint: CGPoint?
    @Published var flashMode = 0         // purely cosmetic: 0 off, 1 auto, 2 on
    @Published var settingsVisible = false { didSet { updateMeter() } }

    /// Read from the audio thread; only feed the level meter when someone can see it.
    private var meterActive = false
    private func updateMeter() { meterActive = settings.debug || settingsVisible }

    /// Visible preview aspect (width / height), shared with the layout.
    let previewAspect: CGFloat = 3.0 / 4.0

    init() {
        engine.onStatus = { [weak self] st in
            self?.trackingStatus = st.tracking
            self?.cardness = st.cardness
        }
        snap.onSnap = { [weak self] in self?.trigger(source: "snap") }
        snap.onLevel = { [weak self] lvl in
            guard let self = self, self.meterActive else { return }
            DispatchQueue.main.async { self.snapLevel = max(lvl, self.snapLevel * 0.85) }
        }

        // Push setting changes into the pipeline.
        settings.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                DispatchQueue.main.async { self?.applySettings() }
            }
            .store(in: &bag)
        applySettings()
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
        // PreviewView is main-actor isolated (UIView); applySettings always runs on main.
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
        engine.setArmed(false)
        armed = false
        if settings.haptics {
            let g = UIImpactFeedbackGenerator(style: .light)
            g.impactOccurred()
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

    func handleTap(at p: CGPoint) {
        withAnimation(.easeOut(duration: 0.15)) { focusPoint = p }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            withAnimation(.easeIn(duration: 0.2)) { self?.focusPoint = nil }
        }
        if settings.tapEnabled { trigger(source: "tap") }
    }

    func flipCamera() {
        engine.switchCamera()
    }

    // MARK: - Shutter

    func takePhoto() {
        withAnimation(.linear(duration: 0.05)) { flash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) { [weak self] in
            withAnimation(.easeOut(duration: 0.2)) { self?.flash = false }
        }
        guard let img = engine.snapshot(visibleAspect: previewAspect) else { return }
        lastPhoto = img
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
