import SwiftUI

/// A plain, generic camera screen. Everything that controls the trick is hidden behind gestures.
struct CameraScreen: View {
    @ObservedObject var model: CameraModel
    @ObservedObject private var settings = AppSettings.shared

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let previewH = w / model.previewAspect
            VStack(spacing: 0) {
                topBar
                    .frame(height: max(56, (geo.size.height - previewH) * 0.32))

                ZStack {
                    PreviewRepresentable(model: model)
                    if let p = model.focusPoint { FocusSquare().position(p) }
                    if model.flash { Color.black.opacity(0.85).allowsHitTesting(false) }
                    if settings.debug { DebugHUD(model: model) }
                }
                .frame(width: w, height: previewH)
                .clipped()

                Spacer(minLength: 0)
                bottomBar
                Spacer(minLength: 0)
            }
            .background(Color.black)
        }
        .ignoresSafeArea(edges: .horizontal)
        .background(Color.black.ignoresSafeArea())
        .sheet(isPresented: $model.showSettings) {
            SettingsView(model: model)
        }
    }

    private var flashIcon: String {
        ["bolt.slash.fill", "bolt.badge.automatic.fill", "bolt.fill"][model.flashMode]
    }

    private var topBar: some View {
        HStack {
            Button { model.flashMode = (model.flashMode + 1) % 3 } label: {
                Image(systemName: flashIcon)
                    .font(.system(size: 19, weight: .medium))
                    .foregroundStyle(model.flashMode == 0 ? Color.white : Color.yellow)
                    .frame(width: 44, height: 44)
            }
            Spacer()
            Image(systemName: "chevron.up")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white.opacity(0.9))
            Spacer()
            Image(systemName: "circle.lefthalf.filled")
                .font(.system(size: 19, weight: .medium))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
        }
        .padding(.horizontal, 20)
        .frame(maxHeight: .infinity, alignment: .bottom)
    }

    private var bottomBar: some View {
        VStack(spacing: 18) {
            Text("PHOTO")
                .font(.system(size: 13, weight: .semibold))
                .kerning(1.2)
                .foregroundStyle(.yellow)

            HStack {
                // Last photo
                Group {
                    if let img = model.lastPhoto {
                        Image(uiImage: img).resizable().scaledToFill()
                    } else {
                        Color(white: 0.16)
                    }
                }
                .frame(width: 50, height: 50)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))

                Spacer()

                // Shutter
                Button { model.takePhoto() } label: {
                    ZStack {
                        Circle().stroke(Color.white, lineWidth: 4).frame(width: 76, height: 76)
                        Circle().fill(Color.white).frame(width: 63, height: 63)
                    }
                }
                .buttonStyle(ShutterStyle())

                Spacer()

                // Flip camera
                Button { model.flipCamera() } label: {
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 21, weight: .medium))
                        .foregroundStyle(.white)
                        .frame(width: 50, height: 50)
                        .background(Circle().fill(Color(white: 0.16)))
                }
            }
            .padding(.horizontal, 30)
        }
    }
}

private struct ShutterStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label.scaleEffect(configuration.isPressed ? 0.92 : 1)
    }
}

private struct FocusSquare: View {
    @State private var appear = false
    var body: some View {
        Rectangle()
            .stroke(Color.yellow, lineWidth: 1.2)
            .frame(width: 76, height: 76)
            .scaleEffect(appear ? 1 : 1.35)
            .onAppear { withAnimation(.easeOut(duration: 0.18)) { appear = true } }
            .allowsHitTesting(false)
    }
}

private struct DebugHUD: View {
    @ObservedObject var model: CameraModel
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(model.armed ? "ARMED" : "REVEALED")  \(model.trackingStatus)")
            Text(String(format: "paper %.2f", model.cardness))
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
        .background(Color.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(10)
        .allowsHitTesting(false)
    }
}
