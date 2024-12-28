import os
import sys

from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.85)
model.eval()
preprocess = weights.transforms()

input_directory = sys.argv[1]
backup_folder = sys.argv[2]


def detect(cls_folder, filename, folder_size):
    img_path = os.path.join(input_directory, cls_folder, filename)

    img = Image.open(img_path).convert("RGB")
    batch = [preprocess(img)]

    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    if len([x for x in labels if x == 'bird']) == 0:
        if folder_size >= 200:
            os.remove(img_path)
        else:
            backup_cls_folder = os.path.join(backup_folder, cls_folder)
            if not os.path.exists(backup_cls_folder):
                os.makedirs(backup_cls_folder)
            os.rename(img_path, os.path.join(backup_folder, cls_folder, filename))



def traverse_dataset():
    for cls_folder in os.listdir(input_directory):
        cls_folder_path = os.path.join(input_directory, cls_folder)
        folder_size = len(os.listdir(cls_folder_path))
        for file in os.listdir(cls_folder_path):
            detect(cls_folder, file, folder_size)


if __name__ == '__main__':
    traverse_dataset()
