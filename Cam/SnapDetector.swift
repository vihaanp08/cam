import AVFoundation
import QuartzCore

/// Listens to the microphone for a finger snap: a very short, sharp, high-frequency burst
/// that jumps far above the recent background level. Talking and music build up gradually
/// and are mostly low-frequency, so they don't trip it.
final class SnapDetector {
    var onSnap: (() -> Void)?
    var onLevel: ((Float) -> Void)?          // 0...1 meter for the debug view
    var sensitivity: Double = 0.55           // 0 = needs a loud snap, 1 = very touchy

    private let engine = AVAudioEngine()
    private var running = false
    private var floorLevel: Float = 0.002
    private var hpPrevIn: Float = 0
    private var hpPrevOut: Float = 0
    private var lastTrigger: CFTimeInterval = 0
    private var history: [Float] = []        // recent block energies (~250 ms)
    private var observers: [NSObjectProtocol] = []

    func start() {
        guard !running else { return }
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            guard granted else { return }
            DispatchQueue.main.async { self?.startEngine() }
        }
    }

    func stop() {
        guard running else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        running = false
    }

    private func startEngine() {
        guard !running else { return }
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .measurement,
                                    options: [.mixWithOthers, .defaultToSpeaker, .allowBluetoothA2DP])
            try session.setPreferredIOBufferDuration(0.005)
            try session.setActive(true)
        } catch {
            print("Audio session error: \(error)")
        }

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { return }
        let sr = Float(format.sampleRate)
        // First-order high-pass around 2.5 kHz: snaps live up there, voices mostly don't.
        let rc = 1 / (2 * Float.pi * 2500)
        let dt = 1 / sr
        let a = rc / (rc + dt)

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 256, format: format) { [weak self] buf, _ in
            self?.process(buf, alpha: a)
        }
        do {
            try engine.start()
            running = true
        } catch {
            print("Audio engine error: \(error)")
        }

        if observers.isEmpty {
            let nc = NotificationCenter.default
            observers.append(nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
                self?.restart()
            })
            observers.append(nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
                guard let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                      let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
                if type == .ended { self?.restart() }
            })
        }
    }

    private func restart() {
        stop()
        startEngine()
    }

    // A candidate waiting to be confirmed: a snap dies away within a few tens of milliseconds,
    // voices, music, knocks and rustling don't.
    private var pendingRMS: Float = 0
    private var pendingAge: Double = -1        // seconds since the candidate, -1 = none
    private var pendingTail: Float = 0         // loudest block after the first 15 ms

    private func process(_ buf: AVAudioPCMBuffer, alpha a: Float) {
        guard let ch = buf.floatChannelData?[0] else { return }
        let n = Int(buf.frameLength)
        guard n > 0 else { return }
        let blockSeconds = Double(n) / buf.format.sampleRate

        var peak: Float = 0
        var energy: Float = 0, fullEnergy: Float = 0
        var pin = hpPrevIn, pout = hpPrevOut
        for i in 0..<n {
            let x = ch[i]
            let y = a * (pout + x - pin)
            pin = x; pout = y
            let ay = abs(y)
            if ay > peak { peak = ay }
            energy += y * y
            fullEnergy += x * x
        }
        hpPrevIn = pin; hpPrevOut = pout
        let rms = sqrt(energy / Float(n))
        let fullRMS = sqrt(fullEnergy / Float(n))

        // Background: median of the last quarter second.
        let background = history.isEmpty ? floorLevel : history.sorted()[history.count / 2]
        history.append(rms)
        let maxBlocks = max(8, Int(0.25 / blockSeconds))
        if history.count > maxBlocks { history.removeFirst(history.count - maxBlocks) }
        floorLevel = 0.995 * floorLevel + 0.005 * rms

        let s = Float(min(max(sensitivity, 0), 1))
        let jump: Float = 22 - 12 * s            // how far above the background (22x ... 10x)
        let absMin: Float = 0.12 - 0.07 * s      // minimum high-passed peak
        let crest = rms > 0 ? peak / rms : 0     // impulsiveness

        let now = CACurrentMediaTime()
        let level = min(1, peak / max(absMin * 2, 1e-4))
        onLevel?(level)

        // Confirm or reject a waiting candidate.
        if pendingAge >= 0 {
            pendingAge += blockSeconds
            if pendingAge > 0.015 { pendingTail = max(pendingTail, rms) }
            if pendingTail > 0.35 * pendingRMS {
                pendingAge = -1                  // still loud: not a snap
            } else if pendingAge >= 0.06 {
                pendingAge = -1
                lastTrigger = now
                DispatchQueue.main.async { [weak self] in self?.onSnap?() }
            }
            return
        }

        let sharp = peak > absMin && rms > jump * max(background, 0.0005) && crest > 2.4
        // Snaps are mostly high-pitched; knocks, bumps and voices are mostly low.
        let bright = fullRMS > 0 && rms / fullRMS > 0.45
        if sharp, bright, now - lastTrigger > 0.8 {
            pendingRMS = rms
            pendingAge = 0
            pendingTail = 0
        }
    }
}
