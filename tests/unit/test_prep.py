from __future__ import annotations

import numpy as np

from overlap.hashing import BorderCrop, apply_crop, detect_border_crop, to_gray


def inner_content(h: int = 360, w: int = 480) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(20, 236, size=(h, w)).astype(np.uint8)


def letterboxed(content: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h, w = content.shape
    canvas = np.zeros((h + top + bottom, w + left + right), dtype=np.uint8)
    canvas[top : top + h, left : left + w] = content
    return canvas


def test_to_gray_from_bgr() -> None:
    bgr = np.zeros((10, 12, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # pure red
    gray = to_gray(bgr)
    assert gray.shape == (10, 12)
    assert gray.dtype == np.uint8
    assert 70 <= int(gray[0, 0]) <= 82  # BT.601: 0.299 * 255 ~= 76


def test_letterbox_bars_detected() -> None:
    content = inner_content()
    frames = [letterboxed(content, 60, 60, 0, 0) for _ in range(10)]
    crop = detect_border_crop(frames)
    assert abs(crop.top - 60) <= 2
    assert abs(crop.bottom - 60) <= 2
    assert crop.left == 0
    assert crop.right == 0


def test_pillarbox_bars_detected_and_crop_recovers_content() -> None:
    content = inner_content()
    frames = [letterboxed(content, 0, 0, 80, 80) for _ in range(10)]
    crop = detect_border_crop(frames)
    recovered = apply_crop(frames[0], crop)
    assert abs(recovered.shape[1] - content.shape[1]) <= 4
    assert recovered.shape[0] == content.shape[0]


def test_transient_dark_scene_does_not_trigger_crop() -> None:
    content = inner_content()
    frames: list[np.ndarray] = [content] * 8 + [np.zeros_like(content)] * 2
    crop = detect_border_crop(frames)
    assert crop.is_noop()


def test_full_black_frames_clamped_to_max_crop() -> None:
    frames = [np.zeros((400, 400), dtype=np.uint8)] * 5
    crop = detect_border_crop(frames)
    assert crop.top <= 160  # 40% of 400
    assert crop.left <= 160


def test_apply_noop_crop_returns_same_array() -> None:
    g = inner_content()
    assert apply_crop(g, BorderCrop()) is g


def test_border_crop_str_roundtrip() -> None:
    crop = BorderCrop(1, 2, 3, 4)
    assert BorderCrop.from_str(crop.as_str()) == crop


def test_empty_probe_set_is_noop() -> None:
    assert detect_border_crop([]).is_noop()
