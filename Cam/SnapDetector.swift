import AVFoundation
import QuartzCore

/// Listens to the microphone for a finger snap: a very short, sharp, high-frequency burst
/// that jumps far above the recent background level. Talking and music build up gradually
/// and are mostly low-frequency, so they don't trip it.
///
/// The audio is read straight from the microphone a few milliseconds at a time (a sink node on the
/// input). A tap on the input hands the audio over in chunks of up to a tenth of a second, and the
/// reveal waited for the next chunk; if the sink delivers nothing, the tap is used instead.
final class SnapDetector {
    var onSnap: (() -> Void)?
    /// Called on the audio thread the moment a snap is confirmed, before `onSnap` (which waits for the
    /// main thread): for showing the real card without that wait. Keep it short.
    var onSnapNow: (() -> Void)?
    var onLevel: ((Float) -> Void)?          // 0...1 meter for the debug view
    var sensitivity: Double = 0.55           // 0 = needs a loud snap, 1 = very touchy

    private let engine = AVAudioEngine()
    private var sink: AVAudioSinkNode?
    private var usingTap = false
    private var running = false
    private var observers: [NSObjectProtocol] = []
    /// Set from the audio thread when audio arrives (read by the start-up check on the main thread).
    private var heardAudio = false

    // Audio-thread state.
    private var floorLevel: Float = 0.002
    private var hpPrevIn: Float = 0
    private var hpPrevOut: Float = 0
    private var lastTrigger: CFTimeInterval = 0
    /// Recent block levels (a ring, about a quarter of a second), and room to sort a copy of them.
    private var history = [Float](repeating: 0, count: 96)
    private var historyCount = 0
    private var historyNext = 0
    private var sorted = [Float](repeating: 0, count: 96)

    func start() {
        guard !running else { return }
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            guard granted else { return }
            DispatchQueue.main.async { self?.startEngine() }
        }
    }

    func stop() {
        guard running else { return }
        engine.stop()
        engine.inputNode.removeTap(onBus: 0)
        if let s = sink {
            engine.disconnectNodeInput(s)
            engine.detach(s)
            sink = nil
        }
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
        let sampleRate = format.sampleRate
        // First-order high-pass around 2.5 kHz: snaps live up there, voices mostly don't.
        let rc = 1 / (2 * Float.pi * 2500)
        let dt = 1 / Float(sampleRate)
        let a = rc / (rc + dt)

        input.removeTap(onBus: 0)
        if let old = sink {
            engine.disconnectNodeInput(old)
            engine.detach(old)
            sink = nil
        }
        heardAudio = false
        if usingTap || format.commonFormat != .pcmFormatFloat32 {
            usingTap = true
            input.installTap(onBus: 0, bufferSize: 256, format: format) { [weak self] buf, _ in
                guard let ch = buf.floatChannelData?[0], buf.frameLength > 0 else { return }
                self?.process(ch, count: Int(buf.frameLength), step: 1,
                              seconds: Double(buf.frameLength) / sampleRate, alpha: a)
            }
        } else {
            let node = AVAudioSinkNode { [weak self] _, frameCount, list -> OSStatus in
                let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
                guard frameCount > 0, let first = buffers.first, let raw = first.mData else { return noErr }
                // (non-interleaved: one channel per buffer; interleaved: take every n-th sample)
                let step = Int(max(1, first.mNumberChannels))
                self?.process(raw.assumingMemoryBound(to: Float.self), count: Int(frameCount), step: step,
                              seconds: Double(frameCount) / sampleRate, alpha: a)
                return noErr
            }
            engine.attach(node)
            engine.connect(input, to: node, format: format)
            sink = node
        }
        do {
            try engine.start()
            running = true
        } catch {
            print("Audio engine error: \(error)")
            // (the sink would not start: fall back to the tap)
            if !usingTap {
                usingTap = true
                if let s = sink { engine.disconnectNodeInput(s); engine.detach(s); sink = nil }
                startEngine()
            }
            return
        }
        // If the sink hears nothing in the first second, listen with the tap instead.
        if !usingTap {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
                guard let self = self, self.running, !self.usingTap, !self.heardAudio else { return }
                self.usingTap = true
                self.restart()
            }
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
    // voices, music, knocks and rustling don't. The loudest block of the first 10 ms is the snap;
    // what comes after that is its tail.
    private var pendingRMS: Float = 0
    private var pendingAge: Double = -1        // seconds since the candidate, -1 = none
    private var pendingTail: Float = 0         // loudest block after the first 10 ms

    /// Median of the recent block levels (the background).
    private func background() -> Float {
        guard historyCount > 0 else { return floorLevel }
        for k in 0..<historyCount { sorted[k] = history[k] }
        // (insertion sort: a few dozen numbers, no allocation on the audio thread)
        var i = 1
        while i < historyCount {
            let v = sorted[i]
            var j = i - 1
            while j >= 0 && sorted[j] > v {
                sorted[j + 1] = sorted[j]
                j -= 1
            }
            sorted[j + 1] = v
            i += 1
        }
        return sorted[historyCount / 2]
    }

    private func process(_ ch: UnsafePointer<Float>, count n: Int, step: Int, seconds blockSeconds: Double, alpha a: Float) {
        guard n > 0 else { return }
        heardAudio = true

        var peak: Float = 0
        var energy: Float = 0, fullEnergy: Float = 0
        var pin = hpPrevIn, pout = hpPrevOut
        var i = 0
        while i < n {
            let x = ch[i * step]
            let y = a * (pout + x - pin)
            pin = x; pout = y
            let ay = abs(y)
            if ay > peak { peak = ay }
            energy += y * y
            fullEnergy += x * x
            i += 1
        }
        hpPrevIn = pin; hpPrevOut = pout
        let rms = sqrt(energy / Float(n))
        let fullRMS = sqrt(fullEnergy / Float(n))

        // Background: median of the last quarter second.
        let bg = background()
        let window = min(history.count, max(8, Int(0.25 / max(blockSeconds, 1e-4))))
        history[historyNext] = rms
        historyNext = (historyNext + 1) % window
        historyCount = min(historyCount + 1, window)
        floorLevel = 0.995 * floorLevel + 0.005 * rms

        let s = Float(min(max(sensitivity, 0), 1))
        let jump: Float = 22 - 12 * s            // how far above the background (22x ... 10x)
        let absMin: Float = 0.12 - 0.07 * s      // minimum high-passed peak
        let crest = rms > 0 ? peak / rms : 0     // impulsiveness

        let now = CACurrentMediaTime()
        if let cb = onLevel { cb(min(1, peak / max(absMin * 2, 1e-4))) }

        // Confirm or reject a waiting candidate (30 ms after it: the reveal follows the snap that
        // much later, where it waited 60 ms before).
        if pendingAge >= 0 {
            pendingAge += blockSeconds
            if pendingAge <= 0.010 {
                pendingRMS = max(pendingRMS, rms)
            } else {
                pendingTail = max(pendingTail, rms)
            }
            if pendingTail > 0.35 * pendingRMS {
                pendingAge = -1                  // still loud: not a snap
            } else if pendingAge >= 0.03 {
                pendingAge = -1
                lastTrigger = now
                onSnapNow?()
                DispatchQueue.main.async { [weak self] in self?.onSnap?() }
            }
            return
        }

        let sharp = peak > absMin && rms > jump * max(bg, 0.0005) && crest > 2.4
        // Snaps are mostly high-pitched; knocks, bumps and voices are mostly low.
        let bright = fullRMS > 0 && rms / fullRMS > 0.45
        if sharp, bright, now - lastTrigger > 0.8 {
            pendingRMS = rms
            pendingAge = 0
            pendingTail = 0
        }
    }
}
