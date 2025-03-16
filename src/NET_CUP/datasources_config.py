from pathlib import Path

config_file_path = Path(__file__).parent.parent.parent

PATIENTS_PATH = config_file_path / 'data' / 'net_patients.csv'
ENUMBER_PATH = config_file_path / 'data' / 'net_enumbers.csv'

UKE_DATASET_DIR = config_file_path / 'data' / 'uke_dataset'
EXTERNAL_DATASET_DIR = config_file_path / 'data' / 'external_dataset'

############# Only necessary for feature extraction ##############
RETCCL_WEIGHTS_PATH = config_file_path / 'weights' / 'retccl.pth'
MTDP_WEIGHTS_PATH = config_file_path / 'weights' / 'mtdp.pth'