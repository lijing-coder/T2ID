# Paths for MMC-AMD dataset
test_index_path = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd/meta/test_set.csv"
train_index_path = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd/meta/train_set.csv"
val_index_path = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd/meta/test_set.csv"
img_info_path = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd/meta/meta.csv"
source_dir = "/19962387/lijing/mmif-scence-class/image_fusion_moe/data/mmc-amd/ImageData/"

# MMC-AMD classes
MMC_CLASSES = ["wetAMD", "dryAMD", "PCV", "Normal"]
MMC_LABEL_TO_INDEX = {name: i for i, name in enumerate(MMC_CLASSES)}
num_MMC_label = len(MMC_CLASSES)
num_classes = num_MMC_label
