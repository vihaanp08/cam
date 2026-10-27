# Cam

A plain-looking camera app with a secret: when a playing card is held up to it, the screen
shows a different card. On a snap, the real card appears.

## Install on your iPhone (no Mac needed)

1. Every push to `main` builds the app on GitHub's cloud Macs. Open the **Actions** tab,
   click the latest green run, and download **Cam-ipa** (unzip it to get `Cam.ipa`).
2. On your computer, open **Sideloadly**, plug in your iPhone, drag in `Cam.ipa`,
   enter your Apple ID and hit **Start**.
3. On the iPhone:
   - Settings > General > VPN & Device Management > trust your Apple ID
   - Settings > Privacy & Security > Developer Mode > on (the phone restarts)
4. With a free Apple ID the app stops opening after 7 days. Re-run step 2 to refresh it.

## Hidden controls

| Action | What it does |
| --- | --- |
| Finger snap | Reveal (fake card disappears, real card shows) |
| Volume buttons / Camera Control | Silent backup reveal |
| Two-finger double-tap | Re-arm (fake card comes back) |
| Two-finger long-press | Settings |

## How it works

- Vision finds the card's four corners every frame and tracks it; a One Euro filter removes jitter.
- Both cards share the same white border, so only the area inside the border is replaced.
  Fingertips on the edges sit on the border and stay real.
- Light is measured from the real card's own border each frame (a smooth bilinear fit),
  and the fake face is relit with it. Blur and sensor grain are matched too.
- Fingers that reach past the border are detected and cut out of the overlay.
- On launch and on camera flip, the view stays covered until the card is locked,
  so the real face never flashes.

## Performing tips

- Hold the card by the **middle of the long edges**, not the corners.
- Get the card in front of the selfie camera **before** opening the app or flipping the camera.
- A plain, darker background behind the card tracks best.
- Turn on **Debug overlay** in settings to see tracking (green dots) and finger cut-outs (red).
