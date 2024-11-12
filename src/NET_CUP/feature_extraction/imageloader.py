"""
This module handles the extraction of image patches from whole-slide images (WSIs) based on annotation data in GeoJSON 
format. It reads the annotations, filters regions by category, and extracts patches from these regions. The `ImageLoader` 
class allows for random or grid-based sampling of patches, with optional filtering of patches containing excessive white 
pixels. 
"""

# Local dependencies
from NET_CUP.utils.shapely_utils import multiple_objects_union, split_multipolygon, is_patch_in_segmentation

# Other dependencies
import random
import math
import PIL.Image
from openslide import OpenSlide
import json
from shapely.geometry import Point, shape, MultiPolygon, Polygon
import numpy as np
from typing import Optional, List, Tuple


def img_white_pixel_filter(img: PIL.Image.Image, threshold: float) -> bool:
    """
    Checks if a given image has a proportion of white pixels greater than a specified threshold.

    :param img: PIL image to check.
    :param threshold: Proportion threshold for white pixels.
    :return: True if the white pixel proportion exceeds the threshold; otherwise, False.
    """
    img = img.convert('L')
    img_arr = np.array(img)
    white_pixels_count = np.count_nonzero(img_arr[img_arr > 240])
    white_pixels_rel = white_pixels_count / img_arr.size
    return white_pixels_rel > threshold


def patch_is_not_empty(tile: PIL.Image.Image, threshold_white: int = 20) -> bool:
    """
    Determines if a patch is not mostly white by checking the median of each RGB channel.

    :param tile: The patch image.
    :param threshold_white: The threshold for whiteness; lower values mean less whiteness allowed.
    :return: True if the patch is not mostly white; otherwise, False.
    """
    histogram = tile.histogram()

    # Take the median of each RGB channel. Alpha channel is not of interest.
    # If roughly each chanel median is below a threshold, i.e close to 0 till color value around 250 (white reference) then tile mostly white.
    whiteness_check = [0, 0, 0]
    for channel_id in (0, 1, 2):
        whiteness_check[channel_id] = np.median(
            histogram[256 * channel_id: 256 * (channel_id + 1)][100:200]
        )

    if all(c <= threshold_white for c in whiteness_check):
        # exclude tile
        return False

    # keep tile
    return True


def read_geojson_file(geojson_file_path: str, all_categories: bool,
                      categories_to_include: Optional[List[str]],
                      categories_to_exclude: Optional[List[str]]) -> List[Polygon]:
    """
    Reads a GeoJSON file and returns annotation geometries as shapely Polygon objects.

    :param geojson_file_path: Path to the GeoJSON file containing segmentation data.
    :param all_categories: If True, include all categories of annotations.
    :param categories_to_include: List of categories to include. 
    :param categories_to_exclude: List of categories to exclude.
    :return: List of annotation objects as shapely Polygons.
    """

    with open(geojson_file_path) as f:
        js = json.load(f)

    classification_missing = False

    if not all_categories:
        try:
            if isinstance(js, list):
                for obj in js:
                    category = obj['properties']['classification']['name']
            if isinstance(js, dict):
                for obj in js['features']:
                    category = obj['properties']['classification']['name']
        except KeyError as e:
            if isinstance(js, list) and len(js) == 1:
                print(
                    f'Annotation object without class, Slide {geojson_file_path}')
                classification_missing = True
            if isinstance(js, dict) and len(js['features']) == 1:
                print(
                    f'Annotation object without class, Slide {geojson_file_path}')
                classification_missing = True
            else:
                raise Exception(
                    f'Multiple annotation objects without class, Slide {geojson_file_path}')

    if all_categories or classification_missing:
        if isinstance(js, list):  # Annotations in QuPath not exported as feature collection
            objects = [shape(obj['geometry']) for obj in js]
        elif isinstance(js, dict):  # Annotations in QuPath exported as feature collection
            objects = [shape(obj['geometry']) for obj in js['features']]
        else:
            raise ValueError('GeoJSON file not in correct format')

        objects = split_multipolygon(objects)
    else:
        objects_to_include = []
        objects_to_exclude = []

        if isinstance(js, list):  # Annotations in QuPath not exported as feature collection
            for obj in js:
                category = obj['properties']['classification']['name']
                if category in categories_to_include:
                    objects_to_include.append(shape(obj['geometry']))
                elif category in categories_to_exclude:
                    objects_to_exclude.append(shape(obj['geometry']))
        elif isinstance(js, dict):  # Annotations in QuPath exported as feature collection
            for obj in js['features']:
                category = obj['properties']['classification']['name']
                if category in categories_to_include:
                    objects_to_include.append(shape(obj['geometry']))
                elif category in categories_to_exclude:
                    objects_to_exclude.append(shape(obj['geometry']))
        else:
            raise ValueError('GeoJSON file not in correct format')

        objects_to_include_union = multiple_objects_union(objects_to_include)
        objects_to_exclude_union = multiple_objects_union(objects_to_exclude)

        difference = objects_to_include_union.difference(
            objects_to_exclude_union)
        if isinstance(difference, MultiPolygon):
            objects = split_multipolygon(difference)
        else:
            objects = [difference]
    return objects


def get_grid_patch_coordinates(annotation_objects: List[Polygon], size_x: int, size_y: int) -> List[Tuple[int, int]]:
    """
    Generates coordinates along a gridfor patches within given annotation objects.

    :param annotation_objects: List of segmentation objects.
    :param size_x: Patch size in x-directon.
    :param size_y: Patch size in y-direction.
    :return: List of coordinates (x, y) of upper left patch vertex points that are inside the annotation objects
    """
    filtered_patches = []
    for object in annotation_objects:
        bounds = object.bounds
        for x in range(math.ceil(bounds[0]), math.ceil(bounds[2]), size_x):
            for y in range(math.ceil(bounds[1]), math.ceil(bounds[3]), size_y):
                patch_candidate = Polygon(
                    [(x, y), (x + size_x, y), (x + size_x, y + size_y), (x, y + size_y)])
                if is_patch_in_segmentation(patch_candidate, object):
                    filtered_patches.append((x, y))
    return filtered_patches


class ImageLoader:
    def __init__(self,
                 geojson_file_path: str,
                 openslide_file_path: str,
                 size_x: int,
                 size_y: int,
                 all_categories: bool = False,
                 categories_to_include: Optional[List[str]] = None,
                 categories_to_exclude: Optional[List[str]] = None,
                 level: int = 0,
                 white_pixels_threshold: float = 0.8):
        """
        Initializes an ImageLoader instance for loading patches from whole-slide images (WSIs).

        :param geojson_file_path: Segmentations mask saved as GeoJSON data_preparation (exported out of QuPath)
        :param openslide_file_path: Path to a WSI in any format that openslide is able to read
        :param size_x: Width of patches that ImageLoader extracts
        :param size_y: Height of patches that ImageLoader extracts
        :param all_categories: True if patches should be extracted from all classes of regions,
                               False if patches should be extracted from the difference between regions of classes
                               from classes_to_include and classes_to_exclude
        :param categories_to_include: List of all annotation classes (strings) to include
        :param categories_to_exclude: List of all annotation classes (strings) to exclude

        Explanation of ImageLoader's instance variables:
        - self.offset_x and self.offset_y: QuPath and therefore exported annotation objects as GeoJSON files use a
            different coordinate system than openslide. QuPath's coordinate system's origin is set as
            (self.offset_x, self.offset_y) in relation to openslide's
        - self.shapely_objects: To prevent accessing the GeoJSON file over and over again the spacial information
            about the annotation objects are stored in this instance variable
        - self.shapely_boundaries: list with boundaries saved as tuple (minx, miny, maxx, maxy) for every annotation
            object
        """
        if categories_to_include == None:
            categories_to_include = ['Tumor']
        if categories_to_exclude == None:
            categories_to_exclude = ['Edding', 'Artefakt groß']
        self.shapely_objects = read_geojson_file(geojson_file_path, all_categories, categories_to_include,
                                                 categories_to_exclude)
        self.shapely_boundaries = self._get_shapely_boundaries()
        self.openslide_img = OpenSlide(openslide_file_path)

        if openslide_file_path.endswith('.svs'):
            self.offset_x = 0
            self.offset_y = 0
        elif openslide_file_path.endswith('.mrxs'):
            self.offset_x = int(
                self.openslide_img.properties["openslide.bounds-x"])
            self.offset_y = int(
                self.openslide_img.properties["openslide.bounds-y"])

        self.size_x = size_x
        self.size_y = size_y
        self.all_categories = all_categories
        self.categories_to_include = categories_to_include
        self.categories_to_exclude = categories_to_exclude
        self.level = level
        self.grid_patch_coordinates = get_grid_patch_coordinates(self.shapely_objects, self.size_x * (2 ** self.level),
                                                                 self.size_y * (2 ** self.level))
        self.white_pixels_threshold = white_pixels_threshold

    def _get_shapely_boundaries(self) -> List[tuple[float, float, float, float]]:
        """
        :return: list with boundaries saved as tuple (minx, miny, maxx, maxy) for every annotation object
        """
        boundaries = []
        if self.shapely_objects is None:
            raise ValueError('No annotations loaded')
        else:
            for shape_object in self.shapely_objects:
                boundaries.append(shape_object.bounds)
        return boundaries

    def get_random_patches(self, n: int, border_patches: bool = True) -> List[tuple[int, int, int, int, PIL.Image.Image]]:
        """
        Selects random patches from annotation objects in the WSI.

        :param n: number of patches
        :param border_patches: 0 if patches whose vertices aren't completely inside the annotation object should not be
            extracted
        :return: random patches of size (self.size_x, self.size_y) inside the annotation objects
        as a list of tuples (x_position, y_position, x_size, y_size, img represented as an array).
        The x,y positions are given in a coordinate system with (self.offset_x, self.offset_y) set as (0,0).
        """

        areas = []
        for shape_object in self.shapely_objects:
            areas.append(shape_object.area)
        areas_proportion = [int(area / sum(areas) * 10000000)
                            for area in areas]

        patch_counter = 0
        random_patches = []

        while True:
            """
            An annotation object from which a patch is going to be extracted is selected first with a probability 
            proportional to its size
            """
            shape_object_selector = random.randint(0, 10000000)
            area_counter = 0
            selected_shape = len(areas_proportion)
            for i, val in enumerate(areas_proportion):
                area_counter += val
                if shape_object_selector <= area_counter:
                    selected_shape = i
                    break

            """Randomly generates points inside of the selected annotation object's boundary box until
            one point is inside the annotation object"""
            while True:
                rand_x = random.randint(int(self.shapely_boundaries[selected_shape][0]),
                                        int(self.shapely_boundaries[selected_shape][2]))
                rand_y = random.randint(int(self.shapely_boundaries[selected_shape][1]),
                                        int(self.shapely_boundaries[selected_shape][3]))
                if border_patches:
                    # If the annotation object contains the middle point of a patch
                    if self.shapely_objects[selected_shape].contains(Point(rand_x, rand_y)):
                        x = rand_x - int(self.size_x / 2)
                        y = rand_y - int(self.size_y / 2)
                        img = self.openslide_img.read_region(
                            (x + self.offset_x, y + self.offset_y),
                            self.level,
                            (self.size_x, self.size_y))
                        if img_white_pixel_filter(img, self.white_pixels_threshold):
                            break
                        random_patches.append(
                            (x, y, self.size_x, self.size_y, img))
                        patch_counter += 1
                        break
                else:
                    # If the annotation objects contains the complete patch (i.e. all vertices)
                    if (self.shapely_objects[selected_shape].contains(Point(rand_x, rand_y)) and
                            self.shapely_objects[selected_shape].contains(Point(rand_x + self.size_x, rand_y)) and
                            self.shapely_objects[selected_shape].contains(Point(rand_x, rand_y + self.size_y)) and
                            self.shapely_objects[selected_shape].contains(
                                Point(rand_x + self.size_x, rand_y + self.size_y))):
                        img = self.get_patch_at_position(rand_x, rand_y)
                        if img_white_pixel_filter(img, self.white_pixels_threshold):
                            break
                        random_patches.append(
                            (rand_x, rand_y, self.size_x, self.size_y, img))
                        patch_counter += 1
                        break
            if patch_counter == n:
                break
        return random_patches

    def get_patch_at_position(self, x: int, y: int) -> PIL.Image.Image:
        """
        Retrieves a patch at a specified position in the WSI.

        :param x: X-coordinate of the patch.
        :param y: Y-coordinate of the patch.
        :return: Patch as a PIL Image.
        """
        img = self.openslide_img.read_region(
            (x + self.offset_x, y + self.offset_y),
            self.level,
            (self.size_x, self.size_y))
        return img

    def __iter__(self):
        """
        Initializes an iterator to return non-empty patches along a grid within annotation objects.
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns the next non-empty patch in the grid iteration.
        """
        try:
            while True:
                patch = self.get_patch_at_position(self.grid_patch_coordinates[self.idx][0],
                                                   self.grid_patch_coordinates[self.idx][1])
                self.idx += 1
                if patch_is_not_empty(patch):
                    break
        except IndexError:
            raise StopIteration
        return patch
