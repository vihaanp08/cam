import UIKit
import MetalKit
import CoreImage
import AVKit
import SwiftUI

/// Metal view that draws the engine's latest composited frame, aspect-filled.
/// Also hosts the hidden gestures and the hardware-button listener.
final class PreviewView: UIView, MTKViewDelegate {
    private let mtkView: MTKView
    private let commandQueue: MTLCommandQueue
    private weak var engine: CameraEngine?
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    var onTap: ((CGPoint) -> Void)?
    var onOpenSettings: (() -> Void)?
    var onRearm: (() -> Void)?
    var onHardwareButton: (() -> Void)?

    private var buttonInteraction: AVCaptureEventInteraction?
    private let volumeButtons = VolumeButtons()
    private var buttonsOn = true

    init(engine: CameraEngine) {
        self.engine = engine
        mtkView = MTKView(frame: .zero, device: engine.metalDevice)
        commandQueue = engine.metalDevice.makeCommandQueue()!
        super.init(frame: .zero)
        backgroundColor = .black

        mtkView.framebufferOnly = false
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 30
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.delegate = self
        mtkView.backgroundColor = .black
        mtkView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(mtkView)
        NSLayoutConstraint.activate([
            mtkView.leadingAnchor.constraint(equalTo: leadingAnchor),
            mtkView.trailingAnchor.constraint(equalTo: trailingAnchor),
            mtkView.topAnchor.constraint(equalTo: topAnchor),
            mtkView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])

        // Hidden controls: two-finger long-press = settings, two-finger double-tap = re-arm.
        let settings = UILongPressGestureRecognizer(target: self, action: #selector(handleSettings(_:)))
        settings.numberOfTouchesRequired = 2
        settings.minimumPressDuration = 0.7
        addGestureRecognizer(settings)

        let rearm = UITapGestureRecognizer(target: self, action: #selector(handleRearm))
        rearm.numberOfTouchesRequired = 2
        rearm.numberOfTapsRequired = 2
        addGestureRecognizer(rearm)

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tap.require(toFail: rearm)
        addGestureRecognizer(tap)

        // Volume buttons and the Camera Control button (works while the camera is running).
        let interaction = AVCaptureEventInteraction { [weak self] event in
            if event.phase == .began { self?.onHardwareButton?() }
        }
        addInteraction(interaction)
        buttonInteraction = interaction

        // Volume buttons, heard through the system volume (works whether or not the camera-button
        // API above delivers them; a press heard both ways is only counted once by the model).
        volumeButtons.attach(to: self)
        volumeButtons.onPress = { [weak self] in
            guard let self = self, self.buttonsOn else { return }
            self.onHardwareButton?()
        }
        volumeButtons.start()
    }

    required init?(coder: NSCoder) { fatalError() }

    func setHardwareButtonsEnabled(_ on: Bool) {
        buttonInteraction?.isEnabled = on
        buttonsOn = on
    }

    @objc private func handleSettings(_ g: UILongPressGestureRecognizer) {
        if g.state == .began { onOpenSettings?() }
    }

    @objc private func handleRearm() { onRearm?() }

    @objc private func handleTap(_ g: UITapGestureRecognizer) {
        onTap?(g.location(in: self))
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let engine = engine, let image = engine.latestImage,
              let drawable = view.currentDrawable,
              let cb = commandQueue.makeCommandBuffer() else { return }
        let size = view.drawableSize
        guard size.width > 0, size.height > 0, image.extent.width > 0 else { return }

        let ext = image.extent
        let scale = max(size.width / ext.width, size.height / ext.height)
        let w = ext.width * scale, h = ext.height * scale
        let t = CGAffineTransform(translationX: -ext.minX, y: -ext.minY)
            .concatenating(CGAffineTransform(scaleX: scale, y: scale))
            .concatenating(CGAffineTransform(translationX: (size.width - w) / 2, y: (size.height - h) / 2))
        let bounds = CGRect(origin: .zero, size: size)
        let black = CIImage(color: .black).cropped(to: bounds)
        let fitted = image.transformed(by: t).composited(over: black).cropped(to: bounds)

        engine.ciContext.render(fitted, to: drawable.texture, commandBuffer: cb, bounds: bounds, colorSpace: colorSpace)
        cb.present(drawable)
        cb.commit()
    }
}

/// SwiftUI wrapper.
struct PreviewRepresentable: UIViewRepresentable {
    let model: CameraModel

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView(engine: model.engine)
        v.onTap = { [weak v] p in model.handleTap(at: p, in: v?.bounds.size ?? .zero) }
        v.onOpenSettings = { model.showSettings = true }
        v.onRearm = { model.rearm() }
        v.onHardwareButton = { model.hardwareButtonPressed() }
        v.setHardwareButtonsEnabled(model.settings.buttonsEnabled)
        model.previewView = v
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
