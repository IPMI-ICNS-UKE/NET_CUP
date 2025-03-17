"""
    This module trains classifiers on extracted features and exports the trained models together with the fitted PCA as ONNX files. 
    For each combination of feature type and classifier, a separate ONNX file is generated.
"""

# Local dependencies
from NET_CUP.data_loading.feature_type import FeatureType
from NET_CUP.data_loading import data_tree, xyp
import NET_CUP.datasources_config as datasources_config

# Other dependencies
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx
import numpy as np

# Settings
classifiers = [SVC(kernel='rbf'), SVC(kernel='linear'), LogisticRegression()]
pca = PCA(0.95)
patches_per_patient = 100
patch_size = 4096
border_patches = True

for feature_type in FeatureType:
    data = data_tree.create_tree(datasources_config.PATIENTS_PATH,
                                 datasources_config.ENUMBER_PATH)
    data_tree.drop_slides_without_extracted_features(
        data, feature_type, datasources_config.UKE_DATASET_DIR)
    patients = data_tree.get_patients(data, 'pi')

    for i, classifier in enumerate(classifiers):
        X_train_patch_level, y_train_patch_level, _ = xyp.get_patch_level_Xyp_complete(
        patients, patches_per_patient, feature_type, patch_size, border_patches, datasources_config.UKE_DATASET_DIR)

        ereg = Pipeline([
            ('pca', pca),
            ('classifier', classifier)
        ])
        ereg.fit(X_train_patch_level, y_train_patch_level)

        onx = to_onnx(ereg, X_train_patch_level[:1],  options={'zipmap': False})
        with open(f"/Users/jiaxilue/Desktop/NET_CUP/models/{repr(feature_type)}_{repr(classifier)}.onnx", "wb") as f:
            f.write(onx.SerializeToString())
