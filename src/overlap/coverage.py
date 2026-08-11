"""What a given index configuration can and cannot detect, and what it costs.

Detection coverage is a *configuration* property, not a fixed property of the
tool: indexing more crop geometries catches more manipulation and costs
proportionally more time and space. Leaving that trade-off implicit means a
user inherits whatever the author guessed, and only discovers the gap when a
manipulated file slips through - so the choice is named, printed, and priced
here instead.

Every claim below is tied to rows of the executable detection matrix
(tests/detection_matrix.toml), which fails if the claim stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass

from overlap.hashing.prep import build_crop_variants

# Manipulations covered by the plain frame hash, whatever the crop settings.
# Fingerprints are computed on decoded pixels, so container- and encoding-level
# changes never survive to the comparison.
ALWAYS_DETECTED = (
    "re-encoding and container swaps",
    "metadata stripping and renaming",
    "trims and splits",
    "concatenation and reassembly from multiple sources",
    "speed changes, with the ratio reported",
    "frame-rate resampling",
    "horizontal mirroring",
    "letterboxing and aspect-ratio changes",
    "colour grading and small overlays",
    "cross-format laundering (MCAP or ROS bag to mp4)",
)

NEVER_DETECTED = (
    "footage re-shot of the same scene",
    "crops deeper than the deepest indexed geometry",
    "off-centre crops that are neither centred nor edge-aligned",
    "rotation, and vertical flips",
    "matched stretches shorter than the evidence floor (10 s)",
)


@dataclass(frozen=True)
class Preset:
    name: str
    crop_ladder: str
    crop_edges: str
    adds: str
    note: str


PRESETS: dict[str, Preset] = {
    "fast": Preset(
        name="fast",
        crop_ladder="",
        crop_edges="",
        adds="nothing beyond the list above",
        note="cheapest. Right for very large public archives where the realistic "
        "threat is lazy resale, and for a first pass you intend to deepen later.",
    ),
    "mild": Preset(
        name="mild",
        crop_ladder="0.94",
        crop_edges="bottom:0.06",
        adds="centred zoom-crops to ~11% and bottom strips to ~9%",
        note="calibrated to what a reseller can crop and still sell the footage. "
        "Egocentric hand data stops being usable if the hands leave the frame or "
        "the pixels are zoomed to mush, so realistic laundering is a slight zoom "
        "or a thin bottom bar - not a deep crop. Half the cost of balanced. "
        "Measured: the plain hash alone covers only 5% centred and 3% one-sided, "
        "which is thinner than a typical HUD strip, so one rung each way earns "
        "its keep where three more do not.",
    ),
    "balanced": Preset(
        name="balanced",
        crop_ladder="0.94,0.88,0.82,0.76,0.70",
        crop_edges="",
        adds="centred zoom-crops to ~30%",
        note="the default. Catches reframing, which is the common crop.",
    ),
    "thorough": Preset(
        name="thorough",
        crop_ladder="0.94,0.88,0.82,0.76,0.70",
        crop_edges="bottom,top,left,right",
        adds="centred zoom-crops plus one-sided edge strips on all four sides",
        note="for corpora worth disguising: your own footage, or a dataset a "
        "reseller has motive to launder carefully. Costs ~4x balanced.",
    ),
}


def codes_per_frame(crop_ladder: str, crop_edges: str) -> int:
    """Hashes stored per sampled frame (each geometry, upright and mirrored)."""
    return (1 + len(build_crop_variants(crop_ladder, crop_edges))) * 2


def describe(crop_ladder: str, crop_edges: str, sample_fps: float) -> dict[str, object]:
    """Coverage and cost of a configuration, for printing or --json output."""
    codes = codes_per_frame(crop_ladder, crop_edges)
    variants = build_crop_variants(crop_ladder, crop_edges)
    centred = [v for v in variants if v.side == "center"]
    edges = sorted({v.side for v in variants if v.side != "center"})
    mb_per_hour = codes * 32 * sample_fps * 3600 / 1e6

    extra = []
    if centred:
        deepest = max(v.pct for v in centred)
        extra.append(f"centred crops to ~{deepest:g}%")
    if edges:
        deepest = max(v.pct for v in variants if v.side != "center")
        extra.append(f"{'/'.join(edges)} edge strips to ~{deepest:g}%")

    gaps = list(NEVER_DETECTED)
    if not centred:
        gaps.insert(0, "cropped copies of any kind (no crop geometries indexed)")
    elif not edges:
        gaps.insert(0, "one-sided edge crops, e.g. a trimmed overlay strip")

    return {
        "codes_per_frame": codes,
        "geometries": 1 + len(variants),
        "sample_fps": sample_fps,
        "index_mb_per_hour": round(mb_per_hour, 1),
        "also_detects": extra,
        "not_detected": gaps,
        "preset": _matching_preset(crop_ladder, crop_edges),
    }


def _matching_preset(crop_ladder: str, crop_edges: str) -> str | None:
    for preset in PRESETS.values():
        if build_crop_variants(preset.crop_ladder, preset.crop_edges) == build_crop_variants(
            crop_ladder, crop_edges
        ):
            return preset.name
    return None


def summary_lines(crop_ladder: str, crop_edges: str, sample_fps: float) -> list[str]:
    """Short, honest lines for the console."""
    d = describe(crop_ladder, crop_edges, sample_fps)
    also = "; ".join(d["also_detects"]) or "nothing beyond the baseline"  # type: ignore[arg-type]
    lines = [
        f"coverage: baseline (re-encode, trim, splice, speed, flip, laundering) + {also}",
        f"blind to: {d['not_detected'][0]}",  # type: ignore[index]
        f"cost: {d['codes_per_frame']} hashes/frame at {sample_fps:g} fps "
        f"= {d['index_mb_per_hour']} MB of index per hour of footage",
    ]
    return lines
