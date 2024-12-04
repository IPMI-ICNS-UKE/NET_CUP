"""
This module contains functions and classes for managing and manipulating a hierarchical data tree structure, 
representing data for patients, their E-numbers, associated slides, and staining types. 

The data tree is built using the `anytree` library, where each node represents a different level in the hierarchy:
- `DataTree`: The root node that represents the entire data structure.
- `Patient`: Represents a patient with information like origin, birthdate, sex, initial diagnosis, and more.
- `ENumber`: Represents a unique E-number. Each biopsy or resection specimen is given an
    unique E-number under which multiple slides can be produced.
- `Slide`: Represents a slide associated with an E-number

Functions provided in this module:
- Tree manipulation: Functions for building the tree, adding/removing nodes, and filtering the data.
- Stain and feature filtering: Functions to drop slides based on their stain type or if they lack extracted features.
- Train-test split: A function to split the data (patients) into training and test sets for machine learning purposes.
"""


# Base dependencies
from __future__ import annotations
import os
from aenum import Enum
from typing import List, Tuple
# Other dependencies
import sklearn.model_selection
from anytree import NodeMixin
from anytree.search import findall
import pandas
from datetime import datetime
# Local dependencies
from NET_CUP.data_loading.feature_type import FeatureType
from NET_CUP.utils.general_utils import convert_e_number_format


class DataTree(NodeMixin):
    """
    A class representing the root of a data tree structure. 
    This should only be used as a root node and cannot have a parent.
    """

    def __init__(self, name: str = 'root', parent=None, children: Patient = None):
        super(DataTree, self).__init__()
        self.name = name
        self.parent = parent
        if self.parent != None:
            raise ValueError('Only use this class as root!')
        if children:
            self.children = children

    def __repr__(self):
        return self.name


class Patient(NodeMixin):
    """
    A class representing a patient node in the data tree structure.
    """

    def __init__(self, name: str, origin: Origin, external: str, parent: DataTree,
                 children: ENumber = None):
        super(Patient, self).__init__()
        self.name = name
        self.parent = parent

        self.origin = origin
        self.external = bool(external)
        if children:
            self.children = children

    def __repr__(self):
        return self.name

    def __str__(self):
        return f'ID: {self.name} Origin: {self.origin} Birthdate: {self.birthdate} Sex: {self.sex} CUP: {self.cup}' \
               f'Initial diagnosis: {self.init_diagn}'


class ENumber(NodeMixin):
    """
    A class representing an E-number node. Each biopsy or resection specimen is given an
    unique E-number under which multiple slides can be produced.
    """

    def __init__(self, name: str, biopsy: str, parent: Patient, children: Slide = None):
        super(ENumber, self).__init__()
        self.name = convert_e_number_format(name, seperations=False)
        self.biopsy = bool(biopsy)
        self.parent = parent
        if children:
            self.children = children

    def __repr__(self):
        return self.name


class Slide(NodeMixin):
    """
    A class representing a slide node, associated with a specific E-number.
    """

    def __init__(self, name: str, stain: Stain, parent: ENumber, children=None):
        super(Slide, self).__init__()
        self.name = name
        self.stain = stain
        self.parent = parent
        if children:
            self.children = children

    def __repr__(self):
        return f'{self.parent}P{self.name}'


class Stain(Enum):
    """Enum representing different types of stains for slides."""
    HE = 0
    SYN = 1
    SSTR2A = 2
    SCHNELL = 3
    STUFE = 4


class Origin(Enum):
    """Enum representing different origins of liver metastases."""
    PANCREAS = 0
    SI = 1
    OTHER = 2
    CUP = 3

    @staticmethod
    def get_origin(origin: str) -> Origin:
        """
        Converts a string origin to an Origin enum value.
        :param origin: A string representing the origin of the liver metastasis.
        :return: An Origin enum value.
        """
        origins_dict = {
            'Pancreas': Origin.PANCREAS,
            'Duodenum': Origin.SI,
            'Jejunum': Origin.SI,
            'Ileum': Origin.SI,
            'Kidney': Origin.OTHER,
            'Stomach': Origin.OTHER,
            'Rectum': Origin.OTHER,
            'Lung': Origin.OTHER,
            '?': Origin.CUP
        }
        return origins_dict[origin]


def create_tree(patients_path: str, enumbers_path: str) -> DataTree:
    """
    Creates a tree structure from patient and E-number data files.
    :param patients_path: Path to the CSV file containing patient data.
    :param enumbers_path: Path to the CSV file containing E-number data.
    :return: Root node of the constructed data tree.
    """
    data = DataTree()

    # Append patient nodes to tree
    df_patients = pandas.read_csv(patients_path)
    patient_nodes = []
    for _, row in df_patients.iterrows():
        patient_nodes.append(
            Patient(str(row['ID']), Origin.get_origin(row['ORIGIN']), row['EXTERNAL'], parent=data))

    # Append E-number nodes and associated slide nodes to tree
    df_enumbers = pandas.read_csv(enumbers_path)
    df_enumbers = df_enumbers.astype({'ID': str})

    for patient in patient_nodes:
        for _, row in df_enumbers[df_enumbers['ID'] == repr(patient)].iterrows():
            enumber = ENumber(row['ENUMBER'], row['BIOPSY'], parent=patient)
            for slide in row['HE'].split(', '):
                if slide != "x":
                    Slide(slide, Stain['HE'], parent=enumber)
    return data


def get_patients(data: DataTree, filter: str = 'pi') -> List[Patient]:
    """
    Retrieves patients based on specified filter criteria.
    :param data: Root of the data tree.
    :param filter: Filter criteria, which determines the origins to include.
    :return: List of Patient nodes that match the filter.
    """
    origin_map = {
        'pi': [Origin.PANCREAS, Origin.SI],
        'p': [Origin.PANCREAS],
        'i': [Origin.SI],
        'o': [Origin.OTHER],
        'pio': [Origin.PANCREAS, Origin.SI, Origin.OTHER],
        'po': [Origin.PANCREAS, Origin.OTHER],
        'all': [Origin.PANCREAS, Origin.SI, Origin.OTHER, Origin.CUP],
        'cup': [Origin.CUP],
    }
    origins = origin_map.get(filter, [])
    patients = findall(data, filter_=lambda node: isinstance(
        node, Patient) and node.origin in origins)
    return list(patients)


def update_tree(data: DataTree) -> None:
    """
    Removes Patient and ENumber nodes that do not have any slide nodes as descendants inplace.
    :param data: Root of the data tree.
    :return: None, modifies the tree in place.
    """

    def patient_no_slide_filter(node: NodeMixin) -> bool:
        # True if node is of type patient and does not have any slide nodes as descendants
        no_slides = False
        if isinstance(node, Patient):
            if len(node.children) == 0:
                no_slides = True
            else:
                no_slides = True
                for child in node.children:
                    if len(child.children) != 0:
                        no_slides = False
        return no_slides

    def enumber_no_slide_filter(node: NodeMixin) -> bool:
        return isinstance(node, ENumber) and len(node.children) == 0

    nodes_to_remove = findall(
        data, filter_=lambda node: enumber_no_slide_filter(node))
    for node in nodes_to_remove:
        node.parent = None
        del node

    nodes_to_remove = findall(
        data, filter_=lambda node: patient_no_slide_filter(node))
    for node in nodes_to_remove:
        node.parent = None
        del node


def drop_slides_of_stain(data: DataTree, stain_to_delete: Stain) -> None:
    """
    Removes all slides of a specified stain type from the tree inplace.
    :param data: Root of the data tree.
    :param stain_to_delete: Stain type to delete.
    :return: None, modifies the tree in place.
    """
    nodes_to_remove = findall(data, filter_=lambda node: (
        isinstance(node, Slide) and node.stain == stain_to_delete))
    for node in nodes_to_remove:
        node.parent = None
        del node
    update_tree(data)


def drop_slides_except_one_stain(data: DataTree, stain_to_remain: Stain) -> None:
    """
    Removes all slides that are not of a specified stain type from the tree inplace.
    :param data: Root of the data tree.
    :param stain_to_remain: Stain type to keep.
    :return: None, modifies the tree in place.
    """
    nodes_to_remove = findall(data, filter_=lambda node: (
        isinstance(node, Slide) and node.stain != stain_to_remain))
    for node in nodes_to_remove:
        node.parent = None
        del node
    update_tree(data)


def drop_slides_without_available_segmentations(data: DataTree, segmentation_dir: str) -> None:
    """
    Removes slides from the tree that do not have available segmentation files in the specified directory.
    :param data: Root of the data tree.
    :param segmentation_dir: Directory containing segmentation files.
    :return: None, modifies the tree in place.
    """
    drop_slides_of_stain(data, Stain.SSTR2A)
    drop_slides_of_stain(data, Stain.SYN)
    available_segmentations = []
    for file in os.listdir(segmentation_dir):
        if file.endswith('geojson'):
            available_segmentations.append(file)
    nodes_to_remove = findall(data, filter_=lambda node: isinstance(node, Slide)
                              and not ((repr(node) + '.geojson') in available_segmentations))
    for node in nodes_to_remove:
        node.parent = None
        del node
    update_tree(data)


def drop_slides_without_extracted_features(data: DataTree, feature_type: FeatureType, feature_dir: str) -> None:
    """
    Removes all slide nodes without any features of the specified type (feature_type) extracted in the given directory (feature_dir),
    and updates the data tree accordingly.

    :param data: The root node of the data tree (DataTree object).
    :param feature_type: The type of feature to look for (FeatureType object).
    :param feature_dir: The directory where the extracted features are stored.
    :return: None, modifies the tree in place.
    """
    drop_slides_of_stain(data, Stain.SSTR2A)
    drop_slides_of_stain(data, Stain.SYN)
    available_features = []

    for file in os.listdir(feature_dir):
        if file.endswith(repr(feature_type)):
            available_features.append(file)

    nodes_to_remove = findall(data, filter_=lambda node: isinstance(node, Slide)
                              and not (
        (repr(node) + '_' + repr(feature_type)) in available_features))

    for node in nodes_to_remove:
        node.parent = None
        del node

    update_tree(data)


def train_test_split(data: DataTree,
                     train_size: float,
                     test_size: float) -> Tuple[List[Patient], List[Patient]]:
    """
    Splits the data into training and testing sets based on the specified train and test sizes.

    :param data: The root node of the data tree (DataTree object).
    :param train_size: The proportion of the data to be used for training (float between 0 and 1).
    :param test_size: The proportion of the data to be used for testing (float between 0 and 1).
    :return: A tuple containing the training and testing Patient nodes (List[Patient]).
    """
    if not train_size + test_size == 1:
        raise ValueError('Train or test size wrong!')

    patients = findall(data, filter_=lambda node: isinstance(node, Patient))
    y = []
    for patient in patients:
        y.append(patient.origin.value)

    train_patients, test_patients, y_train, y_test = sklearn.model_selection.train_test_split(patients,
                                                                                              train_size=train_size,
                                                                                              stratify=y)
    return train_patients, test_patients
