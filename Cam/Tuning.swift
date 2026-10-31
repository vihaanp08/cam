import Foundation

/// Tunable numbers gathered in one place, so the test harness can try other values against the
/// same footage. The app always runs with these defaults.
enum Tuning {
    // Smoothing of the card's outline: a One Euro filter on each corner coordinate.
    /// Hz. Lower = steadier while the card is held still.
    static var smoothMinCutoff = 1.5
    /// Higher = less lag while the card moves. (0.5 measured best of 0.12 to 0.5: less lag in swipes
    /// and tilts, and less jitter when still too)
    static var smoothBeta = 0.5
    /// Hz. How quickly the filter notices the card has started (or stopped) moving. (at 1 Hz it took
    /// several frames to notice a fast move, and the fake trailed the card by up to 15 px meanwhile)
    static var smoothDCutoff = 4.0

    // Starting the fake on a card
    /// Frames in a row a newly found card must be tracked before the fake is painted on it. Only
    /// while the picture runs behind the camera (the head start), which hides the wait: the fake is
    /// then painted back onto the earlier frames too, so it still appears with the card.
    static var confirmFrames = 2
    /// A card coming into view is taken by where its corner index is, long before most of it is in
    /// the picture (CardTracker `completions`). (a switch, to compare in the test harness)
    static var entryByIndex = true
    /// A side with the card's paper and printing carrying on past it (a court card's printed frame
    /// taken for its edge) is moved out to the edge, and such outlines lose to the whole card.
    static var printingGuard = true
    /// With the card's picture known, a card held so close that it runs off two edges of the picture
    /// is found from its one corner in view (CardTracker `cornerFits`).
    static var cornerSearch = true
    /// A card going out of the picture too fast to follow carries on the way it was going, for the
    /// few frames it takes to leave, while that lands on its own printing (FramePipeline).
    static var exitCarryOn = true
    /// With the card's picture known and nothing found matching it, the picture itself is looked for
    /// anywhere in the frame, at any size (CardTracker `pictureSearch`).
    static var pictureSearch = true
    /// Colours on and round the card are read under the light measured on the card itself (a lamp
    /// can light it orange while the room looks white), and its border is judged against its own
    /// colour (CardTracker `PaperReader.gain`, Compositor).
    static var cardWhiteBalance = true
    /// The fake's light follows a sudden change in the real light (switched on or off, exposure, colour)
    /// on the same frame, instead of fading over to it (Compositor).
    static var fastLight = true
    /// The light across the card is read along its four edges and carried across between them, so a
    /// shadow falling across the card falls across the fake too (Compositor).
    static var edgeLight = true
    /// A card's border read the right distance in from each side at a steep angle (4% of its longest
    /// side in can land on the printing when it is narrow one way) (PlanarTracker.cardScore).
    static var evenInset = false
    /// The followed card's border judged even along each side against its own neighbours, so a
    /// shadow or a lamp to one side doesn't count against it (PlanarTracker.cardScore).
    static var evenLocalTrack = true
    /// The same for candidates found by detection.
    static var evenLocalDetect = false
    /// The card's picture read as its printing over the paper around it, so a shadow over part of the
    /// card doesn't stop it matching (CardTracker.samplePicture).
    static var shadowMatch = true
    /// Printing past one end of a side (an index cut off at a steep angle, a thumb over the middle)
    /// moves that side out to the card's edge (CardTracker.printingPast).
    static var bandPast = false
    /// The card's picture can be taken at a steep angle, with a little more leeway on its
    /// proportions while both corner indices show clearly (CardTracker).
    static var tiltLearn = true
    /// A shadow with a straight edge across the card is found where its edge crosses the border and
    /// laid over the fake with a sharp edge (Compositor.findShadow).
    static var shadowLine = true
}
