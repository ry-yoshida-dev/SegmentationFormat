import cv2
import numpy as np
import pytest
from masks import rectangle_mask

from segmentation_format import MaskPolygonizer, Polygon, RunLengthEncoding
from segmentation_format.array_types import BoolArray, UInt8Array


def test_rectangle_traces_pixel_edges() -> None:
    polygons: tuple[Polygon, ...] = MaskPolygonizer().polygons(rectangle_mask(20, 10, 2, 3, 7, 9))
    assert len(polygons) == 1
    assert sorted(polygons[0].to_point_list()) == [[2.0, 3.0], [2.0, 9.0], [7.0, 3.0], [7.0, 9.0]]
    assert polygons[0].area == pytest.approx(30.0)


def test_regions_are_sorted_largest_first() -> None:
    mask: BoolArray = rectangle_mask(30, 30, 0, 0, 3, 3) | rectangle_mask(30, 30, 10, 10, 30, 30)
    polygons: tuple[Polygon, ...] = MaskPolygonizer().polygons(mask)
    assert [polygon.area for polygon in polygons] == [400.0, 9.0]


@pytest.mark.parametrize(
    ("x_min", "y_min", "x_max", "y_max"), [(4, 5, 5, 6), (2, 5, 8, 6), (4, 1, 5, 9)], ids=["pixel", "row", "column"]
)
def test_thin_region_keeps_its_pixel_area(x_min: int, y_min: int, x_max: int, y_max: int) -> None:
    polygons: tuple[Polygon, ...] = MaskPolygonizer().polygons(rectangle_mask(10, 10, x_min, y_min, x_max, y_max))
    assert len(polygons) == 1
    assert polygons[0].area == pytest.approx((x_max - x_min) * (y_max - y_min))
    assert polygons[0].points.shape == (4, 2)


def test_concave_region_area_matches_pixel_count() -> None:
    mask: BoolArray = rectangle_mask(10, 10, 0, 0, 6, 2) | rectangle_mask(10, 10, 0, 0, 2, 7)
    polygons: tuple[Polygon, ...] = MaskPolygonizer().polygons(mask)
    assert len(polygons) == 1
    assert polygons[0].points.shape == (6, 2)
    assert polygons[0].area == pytest.approx(float(mask.sum()))


def test_corner_touching_pixels_form_one_pinched_region() -> None:
    mask: BoolArray = rectangle_mask(4, 4, 0, 0, 1, 1) | rectangle_mask(4, 4, 1, 1, 2, 2)
    polygons: tuple[Polygon, ...] = MaskPolygonizer().polygons(mask)
    assert len(polygons) == 1
    assert polygons[0].area == pytest.approx(2.0)


def test_empty_mask_has_no_polygon() -> None:
    assert MaskPolygonizer().polygons(np.zeros((4, 4), dtype=np.bool_)) == ()


def test_simplification_reduces_vertices() -> None:
    disk: UInt8Array = np.zeros((101, 101), dtype=np.uint8)
    cv2.circle(disk, (50, 50), 40, 1, thickness=-1)
    exact: Polygon = MaskPolygonizer().polygons(disk.astype(np.bool_))[0]
    simplified: Polygon = MaskPolygonizer(simplification_tolerance=2.0).polygons(disk.astype(np.bool_))[0]
    assert simplified.points.shape[0] < exact.points.shape[0]
    assert simplified.area == pytest.approx(exact.area, rel=0.05)


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MaskPolygonizer(simplification_tolerance=-1.0)


def test_polygon_needs_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        Polygon(points=np.zeros((2, 2), dtype=np.float64))


def test_run_length_encoding_is_column_major() -> None:
    mask: BoolArray = np.array([[False, True], [False, True]])
    assert RunLengthEncoding.from_mask(mask).counts == (2, 2)
    assert RunLengthEncoding.from_mask(~mask).counts == (0, 2, 2)


def test_run_length_encoding_round_trips() -> None:
    mask: BoolArray = np.random.default_rng(0).random((17, 23)) > 0.5
    assert np.array_equal(RunLengthEncoding.from_mask(mask).to_mask(), mask)


def test_run_length_encoding_rejects_wrong_total() -> None:
    with pytest.raises(ValueError, match="sum to"):
        RunLengthEncoding(counts=(1, 2), height=2, width=2)
