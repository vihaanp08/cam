import Foundation
import Combine

/// Everything the performer can tweak. Persisted in UserDefaults.
/// The UI binds to this; the video pipeline reads an immutable `RenderConfig` snapshot.
final class AppSettings: ObservableObject {
    static let shared = AppSettings()
    private let d = UserDefaults.standard

    // Fake card shown before the snap
    @Published var rank: Int { didSet { d.set(rank, forKey: "rank") } }            // 1 = A ... 13 = K
    @Published var suit: Int { didSet { d.set(suit, forKey: "suit") } }            // 0 spades, 1 hearts, 2 clubs, 3 diamonds
    @Published var importedFace: String? { didSet { d.set(importedFace, forKey: "importedFace") } } // file name in Documents/faces

    // Triggers
    @Published var snapEnabled: Bool { didSet { d.set(snapEnabled, forKey: "snapEnabled") } }
    @Published var snapSensitivity: Double { didSet { d.set(snapSensitivity, forKey: "snapSensitivity") } } // 0...1
    @Published var buttonsEnabled: Bool { didSet { d.set(buttonsEnabled, forKey: "buttonsEnabled") } }
    @Published var tapEnabled: Bool { didSet { d.set(tapEnabled, forKey: "tapEnabled") } }

    // Behaviour
    @Published var mirror: Bool { didSet { d.set(mirror, forKey: "mirror") } }
    @Published var haptics: Bool { didSet { d.set(haptics, forKey: "haptics") } }
    @Published var autoRearm: Bool { didSet { d.set(autoRearm, forKey: "autoRearm") } }
    @Published var facesOnly: Bool { didSet { d.set(facesOnly, forKey: "facesOnly") } }
    @Published var debug: Bool { didSet { d.set(debug, forKey: "debug") } }
    @Published var showGrid: Bool { didSet { d.set(showGrid, forKey: "showGrid") } }

    // Look tuning
    @Published var brightness: Double { didSet { d.set(brightness, forKey: "brightness") } }   // -0.3...0.3
    @Published var blurScale: Double { didSet { d.set(blurScale, forKey: "blurScale") } }      // 0...2
    @Published var noiseScale: Double { didSet { d.set(noiseScale, forKey: "noiseScale") } }   // 0...2
    @Published var insetScale: Double { didSet { d.set(insetScale, forKey: "insetScale") } }   // 0.5...1.5

    // Speed: how many frames the picture runs behind the camera, so the fake is already in place
    // the moment the card appears (0 = live, 4 = about 0.13 s)
    @Published var headStart: Int { didSet { d.set(headStart, forKey: "headStart") } }

    private init() {
        d.register(defaults: [
            "rank": 9, "suit": 1,
            "snapEnabled": true, "snapSensitivity": 0.55,
            "buttonsEnabled": true, "tapEnabled": false,
            "mirror": true, "showGrid": true, "haptics": true, "autoRearm": false,
            "facesOnly": true, "debug": false,
            "brightness": 0.0, "blurScale": 1.0, "noiseScale": 1.0, "insetScale": 1.0,
            "headStart": 4
        ])
        rank = d.integer(forKey: "rank")
        suit = d.integer(forKey: "suit")
        importedFace = d.string(forKey: "importedFace")
        snapEnabled = d.bool(forKey: "snapEnabled")
        snapSensitivity = d.double(forKey: "snapSensitivity")
        buttonsEnabled = d.bool(forKey: "buttonsEnabled")
        tapEnabled = d.bool(forKey: "tapEnabled")
        mirror = d.bool(forKey: "mirror")
        haptics = d.bool(forKey: "haptics")
        autoRearm = d.bool(forKey: "autoRearm")
        facesOnly = d.bool(forKey: "facesOnly")
        debug = d.bool(forKey: "debug")
        showGrid = d.bool(forKey: "showGrid")
        brightness = d.double(forKey: "brightness")
        blurScale = d.double(forKey: "blurScale")
        noiseScale = d.double(forKey: "noiseScale")
        insetScale = d.double(forKey: "insetScale")
        headStart = d.integer(forKey: "headStart")
    }

    var renderConfig: RenderConfig {
        RenderConfig(mirror: mirror, facesOnly: facesOnly, debug: debug,
                     brightness: brightness, blurScale: blurScale,
                     noiseScale: noiseScale, insetScale: insetScale, headStartFrames: headStart)
    }

    static let rankNames = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    static let suitNames = ["Spades", "Hearts", "Clubs", "Diamonds"]
    static let suitSymbols = ["♠", "♥", "♣", "♦"]
}

/// Immutable snapshot handed to the video queue.
struct RenderConfig {
    var mirror = false
    var facesOnly = true
    var debug = false
    var brightness = 0.0
    var blurScale = 1.0
    var noiseScale = 1.0
    var insetScale = 1.0
    var headStartFrames = 4
    /// Run live while the fake is on the card; only fall behind while waiting for the card.
    var adaptiveHeadStart = true
}
