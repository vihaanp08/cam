import AVFoundation
import MediaPlayer
import UIKit

/// Hears the volume up / down buttons by watching the system volume, which works on every iPhone
/// (the camera-button API only reports presses in some setups). After each press the volume is put
/// straight back, and the system volume pop-up is hidden, so pressing changes nothing visible.
final class VolumeButtons {
    var onPress: (() -> Void)?
    private var observation: NSKeyValueObservation?
    private let volumeView = MPVolumeView(frame: CGRect(x: -2000, y: -2000, width: 10, height: 10))
    private var baseline: Float = 0.5
    private var ignoreUntil: CFTimeInterval = 0

    /// The (invisible) volume view must be on screen for the pop-up to stay hidden.
    func attach(to view: UIView) {
        guard volumeView.superview == nil else { return }
        volumeView.alpha = 0.01
        volumeView.isUserInteractionEnabled = false
        view.addSubview(volumeView)
    }

    func start() {
        let session = AVAudioSession.sharedInstance()
        if session.category != .playAndRecord {
            try? session.setCategory(.ambient, options: [.mixWithOthers])
        }
        try? session.setActive(true)
        guard observation == nil else { return }
        baseline = session.outputVolume
        // At full or zero volume one of the buttons wouldn't change anything: start in the middle.
        if baseline > 0.9 || baseline < 0.1 {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.setVolume(0.5) }
        }
        observation = session.observe(\.outputVolume, options: [.new]) { [weak self] _, change in
            guard let v = change.newValue else { return }
            DispatchQueue.main.async { self?.changed(to: v) }
        }
    }

    private func changed(to v: Float) {
        // our own reset of the volume, not a press
        if CACurrentMediaTime() < ignoreUntil { return }
        guard abs(v - baseline) > 0.001 else { return }
        onPress?()
        setVolume(baseline)
    }

    private func setVolume(_ v: Float) {
        guard let slider = volumeView.subviews.compactMap({ $0 as? UISlider }).first else { return }
        baseline = v
        ignoreUntil = CACurrentMediaTime() + 0.25
        slider.setValue(v, animated: false)
        slider.sendActions(for: .valueChanged)
    }
}
