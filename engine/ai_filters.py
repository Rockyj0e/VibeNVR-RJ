"""Pure helpers to post-filter AI detections (no cv2/av dependencies, easy to unit test).

NOTE: deliberately NOT placed in utils.py. CI runs with PYTHONPATH=.:backend:engine,
where `utils` resolves to backend/utils.py and would shadow engine/utils.py.
"""


def box_area_percent(box) -> float:
    """Area of a normalized [ymin, xmin, ymax, xmax] box as a percentage (0-100) of the frame.

    Coordinates are clamped to the 0-1 range first (detectors may emit values slightly
    outside the frame). Malformed boxes are reported as 0.0.
    """
    try:
        ymin, xmin, ymax, xmax = (min(1.0, max(0.0, float(c))) for c in box)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin) * 100.0


def filter_results_by_min_area(results, min_area_pct):
    """Split AI detections into (kept, dropped) using a minimum box area (% of frame).

    A value that is None, non-numeric or <= 0 disables the filter (everything is kept).
    """
    if not results:
        return [], []
    try:
        min_area = float(min_area_pct or 0.0)
    except (TypeError, ValueError):
        min_area = 0.0
    if min_area <= 0.0:
        return list(results), []

    kept, dropped = [], []
    for res in results:
        if box_area_percent(res.get('box')) >= min_area:
            kept.append(res)
        else:
            dropped.append(res)
    return kept, dropped
