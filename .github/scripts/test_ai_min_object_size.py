"""Tests for the per-camera AI minimum object size filter (ai_min_object_area).

Covers the pure engine helpers (box area maths + filtering) and the backend
schema validator that sanitizes the stored value.
"""
import pytest

import schemas
from engine.ai_filters import box_area_percent, filter_results_by_min_area


def _det(label, box, conf=0.8):
    return {"label": label, "score": conf, "confidence": conf, "box": box}


# Normalized boxes are [ymin, xmin, ymax, xmax]
FULL_FRAME = [0.0, 0.0, 1.0, 1.0]            # 100% of the frame
BIG_PERSON = [0.2, 0.4, 0.8, 0.6]            # 0.6 * 0.2 = 12%
TINY_BLOB = [0.50, 0.50, 0.52, 0.51]         # 0.02 * 0.01 = 0.02%


# --- box_area_percent -------------------------------------------------------

def test_box_area_percent_basic():
    assert box_area_percent(FULL_FRAME) == pytest.approx(100.0)
    assert box_area_percent(BIG_PERSON) == pytest.approx(12.0)
    assert box_area_percent(TINY_BLOB) == pytest.approx(0.02)


def test_box_area_percent_clamps_out_of_frame_coordinates():
    # Detectors can emit slightly negative / >1 values; the area is clamped to the frame.
    assert box_area_percent([-0.5, -0.5, 1.5, 1.5]) == pytest.approx(100.0)


def test_box_area_percent_inverted_or_malformed_boxes_are_zero():
    assert box_area_percent([0.8, 0.8, 0.2, 0.2]) == 0.0  # max < min
    assert box_area_percent(None) == 0.0
    assert box_area_percent([0.1, 0.2]) == 0.0
    assert box_area_percent(["a", "b", "c", "d"]) == 0.0


# --- filter_results_by_min_area ---------------------------------------------

@pytest.mark.parametrize("disabled_value", [0, 0.0, None, -1, "", "abc"])
def test_filter_disabled_keeps_everything(disabled_value):
    results = [_det("person", TINY_BLOB), _det("person", BIG_PERSON)]
    kept, dropped = filter_results_by_min_area(results, disabled_value)
    assert kept == results
    assert dropped == []


def test_filter_drops_small_and_keeps_large():
    small = _det("person", TINY_BLOB)
    large = _det("person", BIG_PERSON)
    kept, dropped = filter_results_by_min_area([small, large], 0.5)
    assert kept == [large]
    assert dropped == [small]


def test_filter_boundary_is_inclusive():
    # 0.5 * 0.2 = 10% exactly -> a threshold of 10 keeps the detection
    box = [0.0, 0.0, 0.5, 0.2]
    det = _det("person", box)
    kept, dropped = filter_results_by_min_area([det], box_area_percent(box))
    assert kept == [det]
    assert dropped == []


def test_filter_applies_to_every_label():
    results = [_det("person", TINY_BLOB), _det("dog", TINY_BLOB), _det("car", BIG_PERSON)]
    kept, dropped = filter_results_by_min_area(results, 1.0)
    assert [r["label"] for r in kept] == ["car"]
    assert [r["label"] for r in dropped] == ["person", "dog"]


def test_filter_drops_results_without_a_valid_box_when_enabled():
    broken = {"label": "person", "confidence": 0.9}
    kept, dropped = filter_results_by_min_area([broken], 0.5)
    assert kept == []
    assert dropped == [broken]


def test_filter_empty_input():
    assert filter_results_by_min_area([], 1.0) == ([], [])
    assert filter_results_by_min_area(None, 1.0) == ([], [])


def test_filter_does_not_mutate_input():
    results = [_det("person", TINY_BLOB), _det("person", BIG_PERSON)]
    snapshot = list(results)
    filter_results_by_min_area(results, 1.0)
    assert results == snapshot


# --- backend schema validator ------------------------------------------------

def _camera(**extra):
    return schemas.CameraCreate(name="Test", rtsp_url="rtsp://example.com/stream", **extra)


def test_schema_default_is_disabled():
    assert _camera().ai_min_object_area == 0.0


@pytest.mark.parametrize("raw, expected", [
    (0.5, 0.5),
    ("1.5", 1.5),
    (0, 0.0),
    (-3, 0.0),            # clamped to the lower bound
    (999, 50.0),          # clamped to the upper bound
    (None, 0.0),
    ("not-a-number", 0.0),
    (float("nan"), 0.0),
])
def test_schema_sanitizes_min_object_area(raw, expected):
    assert _camera(ai_min_object_area=raw).ai_min_object_area == pytest.approx(expected)


# --- CameraThread wiring ------------------------------------------------------

def test_camera_thread_filter_reads_config_key():
    """The thread method must read the same config key the backend pushes to the engine."""
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    import types
    import camera_thread

    small = _det("person", TINY_BLOB)
    large = _det("person", BIG_PERSON)

    enabled = types.SimpleNamespace(config={"ai_min_object_area": 1.0})
    assert camera_thread.CameraThread._filter_ai_results_by_size(enabled, [small, large]) == [large]

    # Key missing (e.g. config pushed by an older backend) -> filter is off
    missing = types.SimpleNamespace(config={})
    assert camera_thread.CameraThread._filter_ai_results_by_size(missing, [small, large]) == [small, large]
