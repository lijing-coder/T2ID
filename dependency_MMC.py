"""MMC-AMD dataset configuration and label definitions.

Keep this module focused on values that are actually used by the MMC data
pipeline.  Paths can be overridden with environment variables so the code is not
bound to a single machine-specific directory layout.
"""

import os

DEFAULT_DATA_ROOT = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd"
DATA_ROOT = os.getenv("MMC_DATA_ROOT", DEFAULT_DATA_ROOT)
META_DIR = os.getenv("MMC_META_DIR", os.path.join(DATA_ROOT, "meta"))

train_index_path = os.getenv("MMC_TRAIN_INDEX_PATH", os.path.join(META_DIR, "train_set.csv"))
val_index_path = os.getenv("MMC_VAL_INDEX_PATH", os.path.join(META_DIR, "test_set.csv"))
test_index_path = os.getenv("MMC_TEST_INDEX_PATH", os.path.join(META_DIR, "test_set.csv"))
img_info_path = os.getenv("MMC_IMG_INFO_PATH", os.path.join(META_DIR, "meta.csv"))
source_dir = os.getenv("MMC_IMAGE_DIR", os.path.join(DATA_ROOT, "ImageData"))

INDEX_COLUMN = "indexes"
FUNDUS_IMAGE_COLUMN = "CFP"
OCT_IMAGE_COLUMN = "OCT"
LABEL_COLUMN = "MMC_label"
MMC_LABEL_LIST = ("wetAMD", "dryAMD", "PCV", "Normal")
