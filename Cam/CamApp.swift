import SwiftUI

@main
struct CamApp: App {
    @StateObject private var model = CameraModel()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            CameraScreen(model: model)
                .statusBarHidden(true)
                .preferredColorScheme(.dark)
                .onAppear { model.start() }
                .onChange(of: phase) { _, newPhase in
                    switch newPhase {
                    case .active: model.start()
                    case .background: model.stop()
                    default: break
                    }
                }
        }
    }
}
