from shapely.geometry import Point, MultiPolygon, Polygon
from shapely import Geometry
from typing import List


def multiple_objects_union(objects: List[Geometry]) -> Geometry:
    """
    Computes the union of multiple Shapely geometry objects.

    :param objects: List of Shapely geometry objects (e.g., Polygon, MultiPolygon).
    :return: A single Shapely geometry object representing the union of all input objects.
    """

    num_objects = len(objects)
    if num_objects == 0:
        objects_union = Point()  # Return an empty Point if no objects are provided
    elif num_objects == 1:
        objects_union = objects[0]
    elif num_objects == 2:
        objects_union = objects[0].union(objects[1])
    else:
        objects_union = objects[0].union(objects[1])
        for i in range(2, num_objects):
            objects_union = objects_union.union(objects[i])
    return objects_union


def split_multipolygon(objects: MultiPolygon) -> List[Polygon]:
    """
    Splits MultiPolygon objects in a list into individual Polygon objects.

    :param objects: MultiPolygon object containing multiple polygons.
    :return: List of Polygon objects, with each MultiPolygon split into individual Polygons.
    """

    polygons = []
    for obj in objects.geoms:
        if isinstance(obj, MultiPolygon):
            polygons.extend(list(obj.geoms))
        else:
            polygons.append(obj)
    return polygons


def is_patch_in_segmentation(patch: Polygon, mask: Polygon, area_threshold: float = 0.5) -> bool:
    """
    Determines if a patch Polygon is sufficiently covered by a mask Polygon, based on an area threshold.

    :param patch: Polygon object representing the patch area.
    :param mask: Polygon object representing the segmentation mask.
    :param area_threshold: Threshold for the ratio of intersection area to patch area to determine inclusion.
                           Must be between 0.0 and 1.0. Default is 0.5.
    :return: True if the intersection area over patch area ratio exceeds the threshold, False otherwise.
    """

    assert 0.0 <= area_threshold <= 1.0, f"Area threshold between 0 and 1 expected, got: {area_threshold}"
    return patch.intersection(mask).area / patch.area > area_threshold
