import pytest
from PIL import Image

from vlm.cache.overlay import build_overlay_pair

BBOXES = [
    [0.10, 0.10, 0.20, 0.20],   # person 0
    [0.70, 0.70, 0.80, 0.80],   # person 1
]
LABELS = {0: "Person A", 1: "Person B"}


def _blank(size=(200, 200)):
    return Image.new("RGB", size, "black")


def _color_count(img, rgb, tol=0):
    return sum(
        1 for pixel in img.getdata()
        if all(abs(a - b) <= tol for a, b in zip(pixel, rgb))
    )


def test_gaze_vecs_without_known_task_raises():
    with pytest.raises(ValueError, match="gaze_vecs requires a known task"):
        build_overlay_pair(
            _blank(), 0, 1, BBOXES, LABELS,
            task=None, gaze_vecs=[(1.0, 0.0), (0.0, 1.0)],
        )


def test_lah_arrow_added_only_for_source_person():
    gaze_vecs = [(1.0, 0.0), (0.0, 1.0)]
    box_only = build_overlay_pair(_blank(), 0, 1, BBOXES, LABELS)
    with_arrow = build_overlay_pair(
        _blank(), 0, 1, BBOXES, LABELS, task="lah", gaze_vecs=gaze_vecs
    )
    red_box_only = _color_count(box_only, (255, 0, 0))
    red_with_arrow = _color_count(with_arrow, (255, 0, 0))
    blue_box_only = _color_count(box_only, (0, 0, 255))
    blue_with_arrow = _color_count(with_arrow, (0, 0, 255))
    assert red_with_arrow > red_box_only, "LAH must draw an arrow for the source (A)"
    assert blue_with_arrow == blue_box_only, "LAH must NOT draw an arrow for the target (B)"


@pytest.mark.parametrize("task", ["laeo", "sa"])
def test_laeo_and_sa_arrows_added_for_both_people(task):
    gaze_vecs = [(1.0, 0.0), (0.0, 1.0)]
    box_only = build_overlay_pair(_blank(), 0, 1, BBOXES, LABELS)
    out = build_overlay_pair(
        _blank(), 0, 1, BBOXES, LABELS, task=task, gaze_vecs=gaze_vecs
    )
    assert _color_count(out, (255, 0, 0)) > _color_count(box_only, (255, 0, 0))
    assert _color_count(out, (0, 0, 255)) > _color_count(box_only, (0, 0, 255))


def _center_y(bbox_norm, H=200):
    y1, y2 = bbox_norm[1] * H, bbox_norm[3] * H
    return (y1 + y2) / 2


def test_arrow_direction_matches_gaze_vec_with_no_sign_flip():
    """gaze_vecs=(1,0) (pure +x, y-down convention) must draw the arrow to the RIGHT
    of the head-bbox center with no y-flip -- the arrow uses the SAME convention as
    gaze_point/gaze_vecs directly (verified empirically against the real graph cache;
    see evidence.py's _direction_bin docstring). Only the human-facing text label
    negates dy, never the visual arrow."""
    W = H = 200
    bboxes = [[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.02, 0.02]]
    out = build_overlay_pair(
        _blank((W, H)), 0, 1, bboxes, LABELS,
        task="lah", gaze_vecs=[(1.0, 0.0), (0.0, 0.0)],
    )
    # bbox is [0.4,0.4,0.6,0.6] -> pixel box [80,80,120,120]; its own outline occupies
    # x in [80,84) and [116,120). Only sample at/below the box's center line (not above
    # it): the Person A/B label sits directly above the box, close enough to the center
    # for a box this size that scanning upward would catch the label instead.
    cx, cy = W // 2, _center_y(bboxes[0], H)
    right_far = out.crop((cx + 25, cy, cx + 40, cy + 4))   # x=125..140, past the box
    left_far = out.crop((cx - 60, cy, cx - 45, cy + 4))    # x=40..55, past the box
    assert _color_count(right_far, (255, 0, 0)) > 0
    assert _color_count(left_far, (255, 0, 0)) == 0


def test_arrow_length_is_constant_across_bbox_sizes():
    """Arrow length is a fixed fraction of the image's shorter side, not the person's
    bbox size, so it never doubles as an unintended distance/confidence cue."""
    W = H = 200
    small_bbox = [[0.48, 0.48, 0.52, 0.52], [0.0, 0.0, 0.02, 0.02]]
    large_bbox = [[0.30, 0.30, 0.70, 0.70], [0.0, 0.0, 0.02, 0.02]]
    out_small = build_overlay_pair(
        _blank((W, H)), 0, 1, small_bbox, LABELS, task="lah", gaze_vecs=[(1.0, 0.0), (0.0, 0.0)],
    )
    out_large = build_overlay_pair(
        _blank((W, H)), 0, 1, large_bbox, LABELS, task="lah", gaze_vecs=[(1.0, 0.0), (0.0, 0.0)],
    )
    # Both boxes are centered at cx=100 and the arrow's fixed length (13% of the
    # shorter image side = 26px) never depends on bbox size, so both tips land at
    # x=126 regardless of box size. Sample narrow windows just before and just past
    # that tip -- clear of each box's own border (small: x=104, large: x=140) and of
    # the Person A/B label (drawn above the box, close to the center line for the
    # small box) -- so the check isolates the arrow itself rather than incidentally
    # hitting the label or border. The "beyond" window (x=129..133) sits strictly
    # between the expected tip (x<=126) and the large box's own right border (x>=137).
    cy_small, cy_large = _center_y(small_bbox[0], H), _center_y(large_bbox[0], H)
    near_tip_small = out_small.crop((122, cy_small, 130, cy_small + 4))
    near_tip_large = out_large.crop((122, cy_large, 130, cy_large + 4))
    beyond_small = out_small.crop((129, cy_small, 133, cy_small + 4))
    beyond_large = out_large.crop((129, cy_large, 133, cy_large + 4))
    assert _color_count(near_tip_small, (255, 0, 0)) > 0
    assert _color_count(near_tip_large, (255, 0, 0)) > 0
    assert _color_count(beyond_small, (255, 0, 0)) == 0
    assert _color_count(beyond_large, (255, 0, 0)) == 0
