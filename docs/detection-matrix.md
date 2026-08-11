# Detection matrix

> Generated from [`tests/detection_matrix.toml`](../tests/detection_matrix.toml) by `scripts/gen_detection_matrix_doc.py` - do not edit by hand.

This is the complete, honest statement of what overlap detects. Detection tiers:

- **designed to detect** - verified by a hard-failing test on generated fixtures
- **best-effort** - sometimes caught; the test records but does not require it
- **not detected** - beyond the current technique; asserted in tests so a capability improvement forces a documentation update

| manipulation | example tested | status | test row |
|---|---|---|---|
| metadata strip / rename | Byte-identical file with a different name and stripped metadata | **designed to detect** | `identical-rename` |
| re-encode | Transcode to H.265 at visibly lower quality | **designed to detect** | `reencode-h265-crf30` |
| container swap | Repackaged mp4 -> mkv without re-encoding | **designed to detect** | `container-swap-mkv` |
| trimming | A 12-second cut from the middle of a corpus file | **designed to detect** | `trim-12s` |
| trimming | A cut at an arbitrary sub-second point (sampling grids misaligned) | **designed to detect** | `trim-arbitrary-phase` |
| splicing | 15 s of corpus footage spliced with unrelated footage | **designed to detect** | `splice-15s-plus-other` |
| reassembly (one master) | Two non-adjacent pieces of one master concatenated and sold as a new session | **designed to detect** | `merge-two-same` |
| concatenation (two owned files) | Pieces of two different owned masters concatenated into one offer | **designed to detect** | `merge-two-masters` |
| concatenation (owned + new) | Owned footage concatenated with genuinely new footage; only the owned half should count | **designed to detect** | `merge-owned-plus-new` |
| speed change (slowdown) | Slowed to half speed - the billable-hours inflation trick | **designed to detect** | `speed-0.5x` |
| speed change (speedup) | Sped up to double speed | **designed to detect** | `speed-2x` |
| horizontal flip | Mirrored left-right | **designed to detect** | `hflip` |
| crop (slight) | Center crop keeping 95% of each dimension, upscaled back | **designed to detect** | `crop-5pct` |
| crop (moderate) | Center crop keeping 85%; needs --preset balanced | **designed to detect** | `crop-15pct` |
| crop (heavy) | Center crop keeping 70%; needs --preset balanced | **designed to detect** | `crop-30pct` |
| crop (beyond the ladder) | Center crop keeping 60% - past the deepest rung of the deepest ladder | not detected | `crop-40pct` |
| edge crop (bottom strip) | Bottom 15% removed (overlay strip trimmed); needs --crop-edges bottom | **designed to detect** | `crop-bottom-15` |
| edge crop (top strip) | Top 10% removed; lands between the 6% and 12% rungs | **designed to detect** | `crop-top-10` |
| edge crop (thin bottom bar) | Bottom 8% removed - a trimmed HUD strip, caught by the default bottom rung | **designed to detect** | `crop-bottom-8-default` |
| crop (slight zoom) | Center crop keeping 92% - caught by the default centred rung | **designed to detect** | `crop-centre-8-default` |
| edge crop, variants disabled | Bottom 15% removed - deeper than the default bottom rung reaches | not detected | `crop-bottom-default-config` |
| letterbox / aspect change | 15% black bars top and bottom | **designed to detect** | `letterbox-15pct` |
| watermark / overlay | Corner text overlay at 60% opacity | **designed to detect** | `watermark-corner` |
| color grading | Saturation +40%, gamma 1.15, brightness and contrast shifts | **designed to detect** | `colorgrade` |
| frame-rate resample | Resampled from 24 fps to 15 fps | **designed to detect** | `fps-resample-15` |
| cross-format laundering | Footage re-wrapped as an MCAP camera topic (or extracted from one) | **designed to detect** | `launder-mcap` |
| false-positive control | Entirely different footage must not match | not detected | `unrelated-footage` |
| false-positive control (same room) | Same environment and camera rig, a different part of the scene - must not match | not detected | `same-scene-different-view` |
| short clips (documented floor) | Clips shorter than the 10 s evidence floor are not detected | not detected | `clip-below-floor` |

See [claim-hygiene.md](claim-hygiene.md) for the wording rules that keep every documented claim tied to a row of this matrix.
