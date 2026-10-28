import Foundation

/// Tunable numbers gathered in one place, so the test harness can try other values against the
/// same footage. The app always runs with these defaults.
enum Tuning {
    // Smoothing of the card's outline: a One Euro filter on each corner coordinate.
    /// Hz. Lower = steadier while the card is held still.
    static var smoothMinCutoff = 1.5
    /// Higher = less lag while the card moves. (0.2 measured best against 0.12: less lag in swipes
    /// and tilts, and no more jitter when still)
    static var smoothBeta = 0.2
    /// Hz. How quickly the filter notices the card has started (or stopped) moving. (at 1 Hz it took
    /// several frames to notice a fast move, and the fake trailed the card by up to 15 px meanwhile)
    static var smoothDCutoff = 4.0

    // Starting the fake on a card
    /// Frames in a row a newly found card must be tracked before the fake is painted on it. Only
    /// while the picture runs behind the camera (the head start), which hides the wait: the fake is
    /// then painted back onto the earlier frames too, so it still appears with the card.
    static var confirmFrames = 2
}
