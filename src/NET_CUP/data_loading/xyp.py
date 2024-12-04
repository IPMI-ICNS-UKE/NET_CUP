"""
This module provides functions to generate feature matrices (X) and label arrays (y) for model training,
using pre-extracted feature vectors for image patches stored in a specified directory (feature_dir).
Additionally, it returns the path (p) to each saved feature vector, with information about the patch
coordinates in the whole-slide image (WSI) embedded in the file name. This information allows for
manual retrieval and visual inspection of individual patches if needed.
"""

# Base dependencies
import os
import copy
from typing import Tuple, Dict, List
import random
# Other dependencies
import numpy as np
# Local dependencies
from NET_CUP.data_loading import data_tree
from NET_CUP.data_loading.feature_type import FeatureType
from NET_CUP.utils import general_utils


def get_Xyp(patient: data_tree.Patient,
            patches_per_patient: int,
            feature_type: FeatureType,
            patch_size: int,
            border_patches: bool,
            feature_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates feature vectors (X), labels (y), and file paths (p) for each patch feature vector of a given patient.

    :param patient: Patient node containing patient-specific data and structure.
    :param patches_per_patient: Number of patches to retrieve per patient.
    :param feature_type: Type of feature to retrieve (e.g., IMAGENET, MTDP, RETCCL).
    :param patch_size: Size of the patch to retrieve.
    :param border_patches: If True, includes patches on the border.
    :param feature_dir: Directory path containing extracted features.
    :return: Tuple containing the feature matrix (X), labels (y), and list of patch file paths.
    """
    category = patient.origin.value
    patches = []
    patch_counter = 0
    # Label array for each patch
    y = np.full((patches_per_patient), category, dtype=int)

    # Split patches_per_patient across each E-number child of the patient node (i.e. different biopsy or resection specimens)
    e_number_distribution = general_utils.split_integer(
        patches_per_patient, len(patient.children))

    for i, enumber in enumerate(patient.children):  # Loop through each E-number
        # Split the patches for this E-number across individual slides produced from the specimen
        slide_distribution = general_utils.split_integer(
            e_number_distribution[i], len(enumber.children))
        for j, slide in enumerate(enumber.children):  # Loop through each slide
            try:
                # List all feature files for this slide and feature type
                features_complete = os.listdir(os.path.join(
                    feature_dir, repr(slide) + '_' + repr(feature_type)))
                
                random.shuffle(features_complete)

                # Filter feature files based on the desired patch size and border_patches setting
                features_preselected = []
                for feature_vector in features_complete:
                    if ((f'w={patch_size}' in feature_vector) and
                            (f'border_patches={border_patches}' in feature_vector)):
                        features_preselected.append(feature_vector)

                # Select the required number of patches for this slide based on the distribution
                features_selected = features_preselected[:slide_distribution[j]]

                # Load each selected feature and add to the X matrix and patches list
                for k, feature in enumerate(features_selected):
                    feature_vector_path = os.path.join(feature_dir, repr(
                        slide) + '_' + repr(feature_type), feature)
                    patches.append(feature_vector_path)
                    if ((i == 0) and (j == 0) and (k == 0)):
                        X_0 = np.array(
                            [np.load(feature_vector_path)])
                        X = np.empty((patches_per_patient, X_0.shape[1]))
                        X[patch_counter, :] = X_0
                        patch_counter += 1
                    else:
                        feature_vector = np.load(feature_vector_path)
                        X[patch_counter, :] = feature_vector
                        patch_counter += 1

            except Exception as e:
                raise Exception(e, slide)

    return X, y, patches


def get_patient_Xyp_dict(patients: List[data_tree.Patient],
                         patches_per_patient: int,
                         feature_type: FeatureType,
                         patch_size: int,
                         border_patches: bool,
                         feature_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Generates a dictionary containing feature vectors, labels, and paths to the saved feature vectors for each patient.

    :param patients: List of Patient nodes.
    :param patches_per_patient: Number of patches per patient.
    :param feature_type: Feature type to retrieve.
    :param patch_size: Size of each patch.
    :param border_patches: Whether to include border patches.
    :param feature_dir: Directory path containing features.
    :return: Dictionary with patient IDs as keys and (X, y, patches) tuples as values.
    """
    patient_Xyp_dict = {}
    for i, patient in enumerate(patients):
        X, y, patches = get_Xyp(patient, patches_per_patient,
                                feature_type, patch_size, border_patches, feature_dir)
        patient_Xyp_dict[repr(patient)] = (X, y, patches)

    return patient_Xyp_dict


def get_patch_level_Xyp_complete(patients: list[data_tree.Patient],
                                 patches_per_patient: int,
                                 feature_type: FeatureType,
                                 patch_size: int,
                                 border_patches: bool,
                                 feature_dir: str):
    """
    Aggregates feature vectors (X), labels (y), and paths for feature vectors for patches (p) from multiple patients.

    :param patients: List of Patient nodes.
    :param patches_per_patient: Number of patches per patient.
    :param feature_type: Type of feature.
    :param patch_size: Patch size.
    :param border_patches: Include border patches if True.
    :param feature_dir: Directory with features.
    :return: Concatenated X, y, and patches from all patients.
    """
    Xyp_dict = get_patient_Xyp_dict(
        patients, patches_per_patient, feature_type, patch_size, border_patches, feature_dir)

    X_0 = Xyp_dict[repr(patients[0])][0]
    y_0 = Xyp_dict[repr(patients[0])][1]

    X_complete = np.empty((len(patients) * X_0.shape[0], X_0.shape[1]))
    y_complete = np.empty((len(patients) * y_0.shape[0]), dtype=int)
    patches_complete = copy.copy(Xyp_dict[repr(patients[0])][2])

    X_complete[:X_0.shape[0], :] = X_0
    y_complete[:y_0.shape[0]] = y_0

    for i in range(1, len(patients)):
        X_complete[i * X_0.shape[0]
            :(i + 1) * X_0.shape[0], :] = Xyp_dict[repr(patients[i])][0]
        y_complete[i * y_0.shape[0]
            :(i + 1) * y_0.shape[0]] = Xyp_dict[repr(patients[i])][1]
        patches_complete.extend(Xyp_dict[repr(patients[i])][2])

    return X_complete, y_complete, patches_complete


def get_patient_level_y(patients: List[data_tree.Patient]) -> np.ndarray:
    """
    Generates a label array at the patient level.

    :param patients: List of Patient nodes.
    :return: Array of labels representing patient origins.
    """
    y = np.ndarray(shape=(1), dtype=int)
    y[0] = patients[0].origin.value
    for i in range(1, len(patients)):
        y = np.append(y, patients[i].origin.value)
    return y
