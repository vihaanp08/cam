import SwiftUI
import PhotosUI

/// Hidden performer settings (two-finger long-press on the preview).
struct SettingsView: View {
    @ObservedObject var model: CameraModel
    @ObservedObject private var s = AppSettings.shared
    @Environment(\.dismiss) private var dismiss
    @State private var pickerItem: PhotosPickerItem?
    @State private var faces: [String] = CardImporter.listFaces()
    @State private var importMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        Text(model.armed ? "Armed: showing the fake card" : "Revealed: showing the real card")
                        Spacer()
                        Circle().fill(model.armed ? Color.green : Color.orange).frame(width: 10, height: 10)
                    }
                    Button(model.armed ? "Reveal now" : "Re-arm") {
                        if model.armed { model.trigger(source: "settings") } else { model.rearm() }
                    }
                }

                Section("Fake card (what they see first)") {
                    Picker("Card", selection: Binding(
                        get: { s.importedFace ?? "" },
                        set: { s.importedFace = $0.isEmpty ? nil : $0 })) {
                        Text("Drawn card").tag("")
                        ForEach(faces, id: \.self) { f in Text(label(for: f)).tag(f) }
                    }
                    if s.importedFace == nil {
                        Picker("Value", selection: $s.rank) {
                            ForEach(1...13, id: \.self) { r in Text(AppSettings.rankNames[r - 1]).tag(r) }
                        }
                        Picker("Suit", selection: $s.suit) {
                            ForEach(0..<4, id: \.self) { i in Text(AppSettings.suitSymbols[i]).tag(i) }
                        }
                        .pickerStyle(.segmented)
                        if s.rank >= 11 {
                            Text("Court cards are drawn generically. For your deck's exact art, import a photo below.")
                                .font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                    if let cg = previewFace() {
                        HStack {
                            Spacer()
                            Image(decorative: cg, scale: 1)
                                .resizable()
                                .aspectRatio(63.0 / 88.0, contentMode: .fit)
                                .frame(height: 150)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .shadow(radius: 3)
                            Spacer()
                        }
                    }
                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("Import a photo of a real card", systemImage: "photo.badge.plus")
                    }
                    if let msg = importMessage {
                        Text(msg).font(.footnote).foregroundStyle(.secondary)
                    }
                    if let f = s.importedFace {
                        Button("Delete this imported card", role: .destructive) {
                            CardImporter.deleteFace(named: f)
                            s.importedFace = nil
                            faces = CardImporter.listFaces()
                        }
                    }
                }

                Section("Triggers") {
                    Toggle("Finger snap", isOn: $s.snapEnabled)
                    if s.snapEnabled {
                        VStack(alignment: .leading) {
                            Text("Snap sensitivity")
                            Slider(value: $s.snapSensitivity, in: 0...1)
                            SnapMeter(level: model.snapLevel)
                        }
                    }
                    Toggle("Volume / Camera Control buttons", isOn: $s.buttonsEnabled)
                    Toggle("Tap on the screen", isOn: $s.tapEnabled)
                    Toggle("Haptic tick on trigger", isOn: $s.haptics)
                    Toggle("Auto re-arm when card is out of view 2 s", isOn: $s.autoRearm)
                }

                Section {
                    Stepper(value: $s.headStart, in: 0...8) {
                        HStack {
                            Text("Head start")
                            Spacer()
                            Text(s.headStart == 0 ? "off" : "\(s.headStart) frames").foregroundStyle(.secondary).monospacedDigit()
                        }
                    }
                } header: {
                    Text("Speed")
                } footer: {
                    Text("While it waits for a card, the picture runs this many frames behind the camera, so the fake is already on the card the instant it comes into view. Once the fake is on, the picture catches up to live. 4 frames is about 0.13 s. Set it to off if the picture ever seems to stop showing the fake.")
                }

                Section("Look") {
                    LabeledSlider(title: "Brightness", value: $s.brightness, range: -0.3...0.3)
                    LabeledSlider(title: "Softness", value: $s.blurScale, range: 0...2)
                    LabeledSlider(title: "Grain", value: $s.noiseScale, range: 0...2)
                    LabeledSlider(title: "Border inset", value: $s.insetScale, range: 0.5...1.6)
                    Button("Reset look") {
                        s.brightness = 0; s.blurScale = 1; s.noiseScale = 1; s.insetScale = 1
                    }
                }

                Section("Camera") {
                    Toggle("Mirror selfie view (like the real camera)", isOn: $s.mirror)
                    Toggle("Grid", isOn: $s.showGrid)
                    Toggle("Only replace card faces", isOn: $s.facesOnly)
                    Toggle("Debug overlay", isOn: $s.debug)
                }

                Section("Hidden controls") {
                    Text("Two-finger long-press: these settings")
                    Text("Two-finger double-tap: re-arm")
                    Text("Snap: reveal")
                    Text("Volume or Camera Control button: press to reveal, press again to re-arm")
                }
                .font(.footnote)
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onChange(of: pickerItem) { _, item in
                guard let item = item else { return }
                importMessage = "Importing…"
                Task { @MainActor in
                    defer { pickerItem = nil }
                    guard let data = try? await item.loadTransferable(type: Data.self),
                          let img = UIImage(data: data) else {
                        importMessage = "Couldn't read that photo."
                        return
                    }
                    do {
                        let name = try CardImporter.importPhoto(img)
                        faces = CardImporter.listFaces()
                        s.importedFace = name
                        importMessage = "Imported. Check the preview: it should be flat, upright and white."
                    } catch {
                        importMessage = "Couldn't find a card. Shoot it flat on a dark surface, straight from above."
                    }
                }
            }
        }
        .onAppear { model.settingsVisible = true }
        .onDisappear { model.settingsVisible = false }
    }

    private func label(for file: String) -> String {
        let idx = (faces.firstIndex(of: file) ?? 0) + 1
        return "Imported card \(idx)"
    }

    private func previewFace() -> CGImage? {
        if let f = s.importedFace { return CardImporter.loadFace(named: f) }
        return CardFaceRenderer.render(rank: s.rank, suit: s.suit)
    }
}

private struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.2f", value)).foregroundStyle(.secondary).monospacedDigit()
            }
            Slider(value: $value, in: range)
        }
    }
}

private struct SnapMeter: View {
    let level: Float
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.25))
                Capsule().fill(level > 0.5 ? Color.green : Color.accentColor)
                    .frame(width: g.size.width * CGFloat(min(1, level)))
            }
        }
        .frame(height: 6)
        .accessibilityHidden(true)
    }
}
