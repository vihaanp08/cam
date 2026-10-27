import SwiftUI
import AVFoundation

/// A camera screen laid out like the built-in iOS Camera app (measured from screenshots on an
/// iPhone 18 Pro, 402 x 874 pt). The live view fills the whole screen; the 3:4 photo area is clear
/// and the bands above and below it are darkened. Everything that controls the trick is hidden
/// behind gestures.
struct CameraScreen: View {
    @ObservedObject var model: CameraModel
    @ObservedObject private var settings = AppSettings.shared

    static let yellow = Color(red: 1.0, green: 0.84, blue: 0.24)
    static let buttonGray = Color(white: 0.12)

    var body: some View {
        GeometryReader { geo in
            let W = geo.size.width
            let H = geo.size.height
            let s = W / 402                          // scale from the measured 402 pt layout
            let clearTop = 128 * s
            let clearH = W * 4 / 3
            let clearBottom = clearTop + clearH

            ZStack(alignment: .topLeading) {
                PreviewRepresentable(model: model)
                    .frame(width: W, height: H)

                // Darkened bands outside the photo area
                VStack(spacing: 0) {
                    Color.black.opacity(0.45).frame(height: clearTop)
                    Color.clear.frame(height: clearH)
                    Color.black.opacity(0.45)
                }
                .allowsHitTesting(false)

                if settings.showGrid {
                    GridLines()
                        .stroke(Color.white.opacity(0.45), lineWidth: 1)
                        .frame(width: W, height: clearH)
                        .offset(y: clearTop)
                        .allowsHitTesting(false)
                }

                if let p = model.focusPoint {
                    FocusSquare().position(p).allowsHitTesting(false)
                }

                if model.flash {
                    Color.black.opacity(0.9).frame(width: W, height: clearH).offset(y: clearTop)
                        .allowsHitTesting(false)
                }

                if settings.debug {
                    DebugHUD(model: model)
                        .offset(x: 8, y: clearTop + 8)
                }

                // Top controls
                topIcons(s: s)
                    .position(x: W / 2 + 2 * s, y: 87.7 * s)

                // Zoom (back) or framing buttons (front), just above the bottom of the photo area
                Group {
                    if model.position == .front {
                        frontButtons(s: s)
                    } else {
                        zoomRow(s: s)
                    }
                }
                .position(x: W / 2, y: clearBottom - 35 * s)

                // Shutter row
                SideButton(size: 48 * s) {
                    Image(systemName: "f.cursive")
                        .font(.system(size: 21 * s, weight: .semibold))
                        .foregroundStyle(.white)
                }
                .position(x: 57.8 * s, y: 719.8 * s)

                ShutterButton(size: 80 * s, video: model.isVideoMode, recording: model.recording) {
                    model.takePhoto()
                }
                .position(x: W / 2, y: 719.8 * s)

                SideButton(size: 48 * s) {
                    DotsIcon(dot: 5.2 * s, gap: 3.4 * s)
                }
                .position(x: 343.8 * s, y: 719.8 * s)

                // Bottom row
                Button { model.openPhotos() } label: {
                    Group {
                        if let img = model.lastPhoto {
                            Image(uiImage: img).resizable().scaledToFill()
                        } else {
                            Color(white: 0.2)
                        }
                    }
                    .frame(width: 48 * s, height: 48 * s)
                    .clipShape(Circle())
                }
                .buttonStyle(PressStyle())
                .position(x: 57.8 * s, y: 815.8 * s)

                ModePill(model: model, s: s)
                    .frame(width: 219.7 * s, height: 47.7 * s)
                    .position(x: 200.85 * s, y: 815.8 * s)

                SideButton(size: 48 * s) {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 21 * s, weight: .semibold))
                        .foregroundStyle(.white)
                } action: {
                    model.flipCamera()
                }
                .position(x: 343.8 * s, y: 815.8 * s)
            }
            .frame(width: W, height: H)
            .onAppear {
                model.viewSize = geo.size
                model.photoRect = CGRect(x: 0, y: clearTop, width: W, height: clearH)
            }
            .onChange(of: geo.size) { _, newSize in
                let ns = newSize.width / 402
                model.viewSize = newSize
                model.photoRect = CGRect(x: 0, y: 128 * ns, width: newSize.width, height: newSize.width * 4 / 3)
            }
        }
        .background(Color.black)
        .ignoresSafeArea()
        .sheet(isPresented: $model.showSettings) {
            SettingsView(model: model)
        }
    }

    // MARK: - Pieces

    @ViewBuilder
    private func topIcons(s: CGFloat) -> some View {
        HStack(spacing: 0) {
            Button {
                model.flashMode = (model.flashMode + 1) % 3
            } label: {
                flashIcon(s: s)
            }
            .frame(width: 43.7 * s)

            Button { model.liveOn.toggle() } label: {
                Image(systemName: model.liveOn ? "livephoto" : "livephoto.slash")
                    .font(.system(size: 21 * s, weight: .semibold))
                    .foregroundStyle(model.liveOn ? Self.yellow : .white)
            }
            .frame(width: 43.7 * s)

            if model.position == .front {
                Button { model.centerStageOn.toggle() } label: {
                    Image(systemName: "person.fill.viewfinder")
                        .font(.system(size: 19 * s, weight: .semibold))
                        .foregroundStyle(model.centerStageOn ? Self.yellow : .white)
                }
                .frame(width: 43.7 * s)
            }

            NightIcon(s: s)
                .frame(width: 43.7 * s)
        }
        .buttonStyle(PressStyle())
    }

    @ViewBuilder
    private func flashIcon(s: CGFloat) -> some View {
        switch model.flashMode {
        case 0:
            ZStack(alignment: .bottomTrailing) {
                Image(systemName: "bolt.fill")
                    .font(.system(size: 19 * s, weight: .semibold))
                Text("A")
                    .font(.system(size: 8 * s, weight: .heavy))
                    .foregroundStyle(.black)
                    .frame(width: 11 * s, height: 11 * s)
                    .background(Circle())
                    .offset(x: 4 * s, y: 3 * s)
            }
            .foregroundStyle(model.position == .front ? Color.white : Self.yellow)
        case 1:
            Image(systemName: "bolt.fill")
                .font(.system(size: 19 * s, weight: .semibold))
                .foregroundStyle(Self.yellow)
        default:
            Image(systemName: "bolt.slash.fill")
                .font(.system(size: 19 * s, weight: .semibold))
                .foregroundStyle(.white)
        }
    }

    private func zoomRow(s: CGFloat) -> some View {
        HStack(spacing: 0) {
            ForEach(Array(model.zoomStops.enumerated()), id: \.offset) { i, stop in
                let selected = i == model.zoomIndex
                Button { model.selectZoom(i) } label: {
                    Text(selected ? "\(stop.label)x" : stop.label)
                        .font(.system(size: 13.5 * s, weight: .semibold))
                        .foregroundStyle(selected ? Self.yellow : .white)
                        .frame(width: 37.7 * s, height: 37.7 * s)
                        .background(Circle().fill(Color.black.opacity(selected ? 0.55 : 0)))
                }
                .buttonStyle(PressStyle())
                .frame(width: 39 * s)
            }
        }
        // The built-in app keeps "1x" centred and lets the other stops hang off either side.
        .offset(x: zoomOffset(s: s))
    }

    private func zoomOffset(s: CGFloat) -> CGFloat {
        guard !model.zoomStops.isEmpty else { return 0 }
        let oneIndex = model.zoomStops.firstIndex { $0.label == "1" } ?? 0
        let n = CGFloat(model.zoomStops.count)
        return (n / 2 - CGFloat(oneIndex) - 0.5) * 39 * s
    }

    private func frontButtons(s: CGFloat) -> some View {
        HStack(spacing: 6.6 * s) {
            RoundGlassButton(size: 37.6 * s) {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
                    .font(.system(size: 14 * s, weight: .semibold))
            }
            RoundGlassButton(size: 37.6 * s) {
                ZStack {
                    Image(systemName: "person.crop.rectangle")
                        .font(.system(size: 15 * s, weight: .semibold))
                    Image(systemName: "arrow.trianglehead.2.clockwise")
                        .font(.system(size: 22 * s, weight: .regular))
                        .opacity(0.9)
                }
            }
        }
        .foregroundStyle(.white)
    }
}

// MARK: - Components

private struct GridLines: Shape {
    func path(in r: CGRect) -> Path {
        var p = Path()
        for i in 1...2 {
            let x = r.minX + r.width * CGFloat(i) / 3
            p.move(to: CGPoint(x: x, y: r.minY)); p.addLine(to: CGPoint(x: x, y: r.maxY))
            let y = r.minY + r.height * CGFloat(i) / 3
            p.move(to: CGPoint(x: r.minX, y: y)); p.addLine(to: CGPoint(x: r.maxX, y: y))
        }
        return p
    }
}

private struct PressStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.92 : 1)
            .opacity(configuration.isPressed ? 0.8 : 1)
    }
}

private struct SideButton<Label: View>: View {
    let size: CGFloat
    @ViewBuilder let label: () -> Label
    var action: () -> Void = {}

    init(size: CGFloat, @ViewBuilder label: @escaping () -> Label, action: @escaping () -> Void = {}) {
        self.size = size
        self.label = label
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            label()
                .frame(width: size, height: size)
                .background(Circle().fill(Color(white: 0.12).opacity(0.92)))
        }
        .buttonStyle(PressStyle())
    }
}

private struct RoundGlassButton<Label: View>: View {
    let size: CGFloat
    @ViewBuilder let label: () -> Label
    var body: some View {
        Button {} label: {
            label()
                .frame(width: size, height: size)
                .background(Circle().fill(Color(white: 0.2).opacity(0.75)))
        }
        .buttonStyle(PressStyle())
    }
}

private struct ShutterButton: View {
    let size: CGFloat
    let video: Bool
    let recording: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(Color(white: 0.16))
                    .overlay(Circle().stroke(Color.white.opacity(0.28), lineWidth: 1))
                    .frame(width: size, height: size)
                if video {
                    RoundedRectangle(cornerRadius: recording ? size * 0.08 : size * 0.4, style: .continuous)
                        .fill(Color(red: 1, green: 0.23, blue: 0.19))
                        .frame(width: recording ? size * 0.42 : size * 0.8, height: recording ? size * 0.42 : size * 0.8)
                } else {
                    Circle().fill(Color.white).frame(width: size * 0.846, height: size * 0.846)
                }
            }
        }
        .buttonStyle(PressStyle())
    }
}

private struct DotsIcon: View {
    let dot: CGFloat
    let gap: CGFloat
    var body: some View {
        VStack(spacing: gap) {
            ForEach(0..<2, id: \.self) { _ in
                HStack(spacing: gap) {
                    ForEach(0..<3, id: \.self) { _ in Circle().fill(Color.white).frame(width: dot, height: dot) }
                }
            }
        }
    }
}

private struct NightIcon: View {
    let s: CGFloat
    var body: some View {
        ZStack(alignment: .topTrailing) {
            Image(systemName: "moon.fill")
                .font(.system(size: 16 * s, weight: .semibold))
                .rotationEffect(.degrees(-15))
            Text("A")
                .font(.system(size: 8.5 * s, weight: .heavy))
                .offset(x: 3 * s, y: -3 * s)
        }
        .foregroundStyle(.white)
    }
}

/// The VIDEO / PHOTO / PORTRAIT selector: the selected mode sits centred in a raised capsule.
private struct ModePill: View {
    @ObservedObject var model: CameraModel
    let s: CGFloat

    private var font: UIFont { .systemFont(ofSize: 16 * s, weight: .semibold) }
    private var kern: CGFloat { 1.1 * s }
    private var spacing: CGFloat { 33 * s }

    private func width(_ t: String) -> CGFloat {
        (t as NSString).size(withAttributes: [.font: font, .kern: kern]).width
    }

    var body: some View {
        GeometryReader { g in
            let widths = model.modes.map(width)
            let total = widths.reduce(0, +) + CGFloat(max(0, widths.count - 1)) * spacing
            let centers: [CGFloat] = widths.indices.map { i in
                widths[..<i].reduce(0, +) + CGFloat(i) * spacing + widths[i] / 2
            }
            // The labels' row is centred in the pill; shift it so the selected mode sits in the middle.
            let offset = total / 2 - centers[min(max(model.modeIndex, 0), centers.count - 1)]

            ZStack {
                Capsule().fill(Color(white: 0.08).opacity(0.95))
                Capsule()
                    .fill(Color(white: 0.13))
                    .overlay(Capsule().stroke(Color.white.opacity(0.13), lineWidth: 1))
                    .frame(width: 90.7 * s, height: 41.7 * s)
                HStack(spacing: spacing) {
                    ForEach(Array(model.modes.enumerated()), id: \.offset) { i, m in
                        Text(m)
                            .font(.system(size: 16 * s, weight: .semibold))
                            .kerning(kern)
                            .foregroundStyle(i == model.modeIndex ? CameraScreen.yellow : Color.white)
                            .lineLimit(1)
                            .fixedSize()
                            .onTapGesture { model.selectMode(i) }
                    }
                }
                .fixedSize()
                .offset(x: offset)
                .animation(.easeOut(duration: 0.2), value: model.modeIndex)
                .frame(width: g.size.width, height: g.size.height)
                .clipped()
                // fade the labels out towards the ends of the pill, like the real one
                .mask(
                    LinearGradient(stops: [
                        .init(color: .clear, location: 0.02),
                        .init(color: .black, location: 0.16),
                        .init(color: .black, location: 0.84),
                        .init(color: .clear, location: 0.98)
                    ], startPoint: .leading, endPoint: .trailing)
                )
            }
            .frame(width: g.size.width, height: g.size.height)
            .clipShape(Capsule())
            .contentShape(Capsule())
            .gesture(DragGesture(minimumDistance: 15).onEnded { v in
                if v.translation.width < -20 { model.selectMode(model.modeIndex + 1) }
                if v.translation.width > 20 { model.selectMode(model.modeIndex - 1) }
            })
        }
    }
}

private struct FocusSquare: View {
    @State private var appear = false
    var body: some View {
        Rectangle()
            .stroke(CameraScreen.yellow, lineWidth: 1.2)
            .frame(width: 76, height: 76)
            .scaleEffect(appear ? 1 : 1.35)
            .onAppear { withAnimation(.easeOut(duration: 0.18)) { appear = true } }
    }
}

private struct DebugHUD: View {
    @ObservedObject var model: CameraModel
    var body: some View {
        let st = model.status
        VStack(alignment: .leading, spacing: 3) {
            Text("\(model.armed ? "ARMED" : "REVEALED")  \(st.tracking)  \(Int(st.fps.rounded())) fps")
            Text("src \(st.source)  paper \(String(format: "%.2f", st.cardness))")
            Text(st.reason)
            HStack(spacing: 6) {
                Text("snap")
                GeometryReader { g in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.white.opacity(0.2))
                        Capsule().fill(model.snapLevel > 0.5 ? Color.green : Color.white)
                            .frame(width: g.size.width * CGFloat(min(1, model.snapLevel)))
                    }
                }
                .frame(width: 90, height: 6)
            }
        }
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(.white)
        .padding(8)
        .background(Color.black.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .allowsHitTesting(false)
    }
}
