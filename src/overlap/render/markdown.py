"""Markdown report renderer (for pasting into issues/email)."""

from __future__ import annotations

from typing import Any


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# overlap comparison report",
        "",
        f"**{s['overlap_pct']}% of offered footage matches this corpus** - "
        f"{s['matched_hours']} of {s['offered_hours']} offered hours, "
        f"{s['files_with_overlap']} of {s['files_offered']} files.",
        "",
        f"- exact (byte-identical): {s['by_tier_hours']['exact']} h",
        f"- strong: {s['by_tier_hours']['strong']} h",
        f"- probable: {s['by_tier_hours']['probable']} h",
        f"- speed-adjusted underlying footage: {s['speed_adjusted_matched_hours']} h",
    ]
    flags = s["flags"]
    if flags["slowdown_files"]:
        lines.append(
            f"- **{flags['slowdown_files']} file(s) slowed down (billable-hours inflation)**"
        )
    if flags["spliced_files"]:
        lines.append(f"- {flags['spliced_files']} spliced file(s)")
    if flags["flipped_files"]:
        lines.append(f"- {flags['flipped_files']} mirrored file(s)")
    if flags.get("cropped_files"):
        lines.append(f"- {flags['cropped_files']} cropped file(s)")
    if flags.get("weak_only_files"):
        lines.append(
            f"- **{flags['weak_only_files']} file(s) carry weak evidence not counted "
            f"above** (re-run with `--tier weak` to include it)"
        )

    matched = [f for f in report["files"] if f["overlap_pct"] > 0]
    if matched:
        lines += ["", "| offered file | duration s | overlap % | signals |", "|---|---:|---:|---|"]
        for f in matched:
            sigs = []
            for m in f["matches"]:
                if m["speed_ratio"] >= 1.1:
                    sigs.append(f"slowed ~{m['speed_ratio']:.1f}x")
                if m["mirrored"]:
                    sigs.append("h-flip")
                if m.get("crop_pct"):
                    sigs.append(f"cropped {m.get('crop_geometry', '')}".strip())
            if f["sha256_exact"]:
                sigs.append("byte-identical")
            lines.append(
                f"| {f['relpath']} | {f['duration_s']} | {f['overlap_pct']} | "
                f"{', '.join(sorted(set(sigs)))} |"
            )
    return "\n".join(lines) + "\n"
