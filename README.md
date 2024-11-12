## About The Project

This repository contains the code for the publication titled "Metastatic neuroendocrine tumors with unknown site of the primary: a machine-learning based approach to a complex scenario".

## Installation

1. Clone the repository.
   ```sh
   git clone https://github.com/IPMI-ICNS-UKE/NET_CUP
   ```
2. Install the package and necessary dependencies.
   ```sh
   pip install -e .
   pip install -r requirements.txt
   ```
3. Download data:
   - Download the required CSV files and feature vectors from LINK and place them in the [data/](data/) folder.

### Additional Steps for Generating New Feature Vectors

To respect data privacy, only feature vectors generated from random patches of whole-slide images (WSIs) are publicly accessible. The original WSIs can be provided upon reasonable request.

1. Download ResNet weights:
   - Download the MTDP ResNet weights from [MTDP repository](https://github.com/waliens/multitask-dipath/issues/1).
   - Download the RetCCL ResNet weights from [RetCCL repository](https://github.com/Xiyue-Wang/RetCCL?tab=readme-ov-file) under the "Pre-trained models for histopathological image tasks" section.
   - Rename the downloaded files to mtdp.pth and retccl.pth and place them into [weights/](weights/).
2. Copy WSIs along with their .geojson segmentation files into [data/external_dataset/](data/external_dataset/) and [data/uke_dataset/](data/uke_dataset/).
3. Install OpenSlide: https://openslide.org/download/.

## Usage

### Reproducing results

To reproduce the results from the publication, run the Jupyter Notebooks available in [src/NET_CUP/experiments](src/NET_CUP/experiments).
Within each notebook, you can select different pretrained ResNet backbones and classifiers. Configure these settings in the notebook's "Settings" section by changing the classifier variable or setting the feature_type variable to one of the following options as described in the publication:

- FeatureType.IMAGENET
- FeatureType.RETCCL
- FeatureType.MTDP

### Generating new feature vectors

After acquiring access to the original WSIs and following the [Installation](#installation) steps, you can generate new feature vectors:

```sh
python src/NET_CUP/feature_extraction/extract_features.py
```

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->

## Contact

Rüdiger Schmitz - r.schmitz@uke.de

Project Link: [https://github.com/IPMI-ICNS-UKE/NET_CUP](https://github.com/IPMI-ICNS-UKE/NET_CUP)
