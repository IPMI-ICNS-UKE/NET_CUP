"""
This module extracts and saves feature vectors for patches from whole-slide images (WSIs) using specified feature 
extractors. It processes WSIs and their corresponding GeoJSON annotation files, extracting patches based on defined 
categories and conditions, such as patch size, resolution level, and inclusion/exclusion of border patches. The extracted 
features are saved as numpy arrays, with metadata embedded in the filenames for later analysis or model training.
"""

# Local dependencies
from NET_CUP.feature_extraction import feature_extractor
from NET_CUP.feature_extraction.imageloader import ImageLoader
import NET_CUP.datasources_config as datasources_config

# Other dependencies
import os
from tqdm import tqdm
import numpy as np
from typing import Optional, List


def extract_features(input_directory: str,
                     file_ending: str,
                     extractor: feature_extractor.FeatureExtractor,
                     num_patches: int,
                     size_x: int,
                     size_y: int,
                     level: int,
                     new_slides: bool = False,
                     categories_to_include: Optional[List[str]] = None,
                     categories_to_exclude: Optional[List[str]] = None,
                     border_patches: bool = True):
    """
    Extracts and saves feature vectors for patches from whole-slide images (WSIs) using a specified feature extractor.

    :param input_directory: Directory containing WSI files and .geojson annotations.
    :param file_ending: File extension of the WSI files (e.g., '.svs' or '.mrxs').
    :param extractor: Feature extractor to use for generating feature vectors (e.g., ResNet-based extractors).
    :param num_patches: Number of patches to extract from each slide.
    :param size_x: Width of each patch at the specified level.
    :param size_y: Height of each patch at the specified level.
    :param level: Resolution level of the WSI from which to extract patches.
    :param new_slides: If True, only extract features for slides that don't already have a feature directory.
    :param categories_to_include: List of categories to include for patch extraction.
    :param categories_to_exclude: List of categories to exclude from patch extraction.
    :param border_patches: If True, includes patches on the border of the slide; otherwise, excludes them.
    """
    # Gather slide names from geojson files in the input directory
    slides = []
    for file in os.listdir(input_directory):
        if file.endswith('.geojson'):
            slides.append(file.removesuffix('.geojson'))

    # Loop through each slide and extract patches
    patch = 0
    for j, slide in enumerate(tqdm(slides)):
        # Create a directory for feature vectors if it doesn't exist
        feature_vector_folder = slide + '_' + repr(extractor)
        try:
            os.mkdir(os.path.join(input_directory, feature_vector_folder))
        except OSError:
            if new_slides:
                continue
            else:
                print("Creation of the directory failed, it may already exist")

        # Paths for the slide image and geojson annotation file
        openslide_path = os.path.join(input_directory, slide + file_ending)
        geojson_path = os.path.join(input_directory, slide + '.geojson')

        try:
            imgloader = ImageLoader(geojson_path, openslide_path, size_x, size_y,
                                    all_categories=False,
                                    categories_to_include=categories_to_include,
                                    categories_to_exclude=categories_to_exclude,
                                    level=level)

            os.chdir(os.path.join(input_directory, feature_vector_folder))

            # Extract and save feature vectors for the specified number of patches
            for i in range(num_patches):
                patch = i

                patch = imgloader.get_random_patches(
                    1, border_patches=border_patches)
                feature_vector = extractor.get_vector(patch[0][4])

                # Determine the patch dimensions at level 0 (highest resolution)
                if file_ending == '.svs':
                    size_x_level_0 = size_x * (4 ** level)
                    size_y_level_0 = size_y * (4 ** level)
                elif file_ending == '.mrxs':
                    size_x_level_0 = size_x * (2 ** level)
                    size_y_level_0 = size_y * (2 ** level)

                # Save the feature vector with metadata in the file name
                np.save(
                    os.getcwd() + "/" +
                    f"x={patch[0][0]} y={patch[0][1]} w={size_x_level_0} h={size_y_level_0} excluded_categories={str(categories_to_exclude)} border_patches={border_patches}",
                    feature_vector)
        except Exception as e:
            raise Exception(
                f'Did not work for {slide}. Patch {patch + 1}, Error:{str(e)}')


if __name__ == '__main__':
    standard_settings = {
        '.mrxs': [256, 256, 4],
        '.svs': [512, 512, 1]
    }
    feature_extractors = [feature_extractor.ImageNetResnetFeatureExtractor(
    ), feature_extractor.MtdpResnetFeatureExtractor(), feature_extractor.RetCCLResnetFeatureExtractor()]
    number_of_patches = 100

    for extractor in feature_extractors:
        extract_features(datasources_config.EXTERNAL_DATASET_DIR,
                         '.svs', extractor, number_of_patches, *standard_settings['.svs'])
        extract_features(datasources_config.UKE_DATASET_DIR,
                         '.mrxs', extractor, number_of_patches, *standard_settings['.mrxs'])

