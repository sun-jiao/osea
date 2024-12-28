import argparse
import csv
import math
import os
import sqlite3

import torch
from PIL import Image, ImageDraw
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detect_transforms = transforms.Compose([
    transforms.ToTensor(),
])

classify_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def parse_args():
    parser = argparse.ArgumentParser(description='OSEA Command-Line Tool for Bird Species Detection and Identification')
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing bird images')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder for annotated images')
    parser.add_argument('--location', required=False, help='Geographical coordinates (latitude,longitude) for species filtering')
    parser.add_argument('--classification_model', required=False, help='Name of classification model, must match the model name in Pytorch identically')
    parser.add_argument('--model_path', required=True, help='Path to the model .pth file')
    parser.add_argument('--label_path', required=False, help='Path to the label file')
    parser.add_argument('--class_number', type=int, default=11000, required=False, help='Class number to identify species')
    parser.add_argument('--detection_model', required=False, help='Name of detection model, must match the model name in Pytorch identically')
    parser.add_argument('--detection_model_path', required=False, help='Path to the model .pth file')
    parser.add_argument('--detection_class_number', type=int, required=False, help='Class number to identify species')
    parser.add_argument('--db_path', required=False, help='Path to the distribution database, is absent, distribution filter will not be applied')
    parser.add_argument('--clasification_threshold', type=float, default=0.85, help='Confidence threshold for classification')
    parser.add_argument('--detection_threshold', type=float, default=0.85, help='Confidence threshold for detection')
    parser.add_argument('--detection_target', type=int, help='Target class for detection')
    return parser.parse_args()


def load_detection_model(model_name: str, num_classes: int, model_path: str, threshold: float):
    if not model_name:
        model_name = 'fasterrcnn_resnet50_fpn_v2'

    m = getattr(models.detection, model_name)
    if num_classes and model_path:
        model = m(num_classes=num_classes, score_thresh=threshold, box_score_thresh=threshold)
    elif num_classes:
        model = m(num_classes=num_classes, score_thresh=threshold, box_score_thresh=threshold, pretrained=True)
    elif model_path:
        model = m(score_thresh=threshold, box_score_thresh=threshold)
    else:
        model = m(pretrained=True, score_thresh=threshold, box_score_thresh=threshold)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_classification_model(model_name: str, num_classes: int, model_path: str):
    if not model_name:
        model_name = 'resnet34'

    m = getattr(models, model_name)
    if num_classes and model_path:
        model = m(num_classes=num_classes)
    elif num_classes:
        model = m(num_classes=num_classes, pretrained=True)
    elif model_path:
        model = m()
    else:
        model = m(pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return detect_transforms(image).unsqueeze(0), image


def detect_objects(detection_model, image_tensor, detection_target):
    if not detection_target:
        detection_target = 16

    with torch.no_grad():
        predictions = detection_model(image_tensor)
    keep = [i for i, label in enumerate(predictions[0]['labels']) if label == detection_target]
    boxes = [predictions[0]['boxes'][i].tolist() for i in keep]
    scores = [predictions[0]['scores'][i].item() for i in keep]
    return boxes, scores


def read_label_list(file_path):
    if not file_path:
        file_path = './labels.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]


# crop and classification
def classify_objects(classification_model, image, boxes, label_map, species_list):
    results = []

    for box in boxes:
        cropped_img = image.crop(box)
        input_tensor = classify_transforms(cropped_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classification_model(input_tensor)[0]
            filtered = get_filtered_predictions(logits, species_list)
            probabilities = softmax(filtered)
            (top_class_id, top_prob) = probabilities[0]
            results.append((box, label_map[top_class_id], top_prob))
    return results

def save_results(image, results, output_folder, threshold, image_name):
    folder = os.path.join(output_folder, "image_output")
    os.makedirs(folder, exist_ok=True)

    out_file = os.path.join(output_folder, "result.csv")

    with open(out_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)

        for box, label, confidence in results:
            if confidence >= threshold:
                draw = ImageDraw.Draw(image)
                draw.rectangle(box, outline="red", width=2)
                draw.text((box[0], box[1]), f"{label} ({confidence:.2f})", fill="red")

                writer.writerow([image_name, label, confidence])
                image.save(os.path.join(folder, image_name))



def softmax(tuples):
    # `torch.nn.functional.softmax` requires the input to be `Tensor`, so I implemented it myself
    values = [t[1] for t in tuples]
    exp_values = [math.exp(v) for v in values]
    sum_exp_values = sum(exp_values)
    softmax_values = [ev / sum_exp_values for ev in exp_values]
    updated_tuples = [(t[0], softmax_values[i]) for i, t in enumerate(tuples)]
    updated_tuples.sort(key=lambda t: t[1], reverse=True)

    return updated_tuples


def get_filtered_predictions(predictions: list[float], species_list: list[int]) -> list[tuple[int, float]]:
    original = {index: value for index, value in enumerate(predictions)}

    if species_list:
        filtered_predictions = [(key, value) for key, value in original.items() if key in species_list]
    else:
        filtered_predictions = [(key, value) for key, value in original.items()]

    return filtered_predictions


class DistributionDB:
    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()


    def get_list(self, lat, lng) -> list:
        self.cur.execute(f'''
SELECT m.cls
FROM distributions AS d
LEFT OUTER JOIN places AS p
  ON p.worldid = d.worldid
LEFT OUTER JOIN sp_cls_map AS m
  ON d.species = m.species
WHERE p.south <= {lat}
  AND p.north >= {lat}
  AND p.east >= {lng}
  AND p.west <= {lng}
GROUP BY d.species, m.cls;
''')

        return [row[0] for row in self.cur]

    def close(self):
        self.cur.close()
        self.con.close()


def main():
    args = parse_args()
    classification_model = load_classification_model(args.classification_model, args.class_number, args.model_path)
    detection_model = load_detection_model(args.detection_model, args.detection_class_number, args.detection_model_path, args.detection_threshold)

    db, species_list = None, None
    if args.db_path and args.location:
        lat, lng = map(float, args.location.split(','))
        db = DistributionDB(args.db_path)
        species_list = db.get_list(lat, lng)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for image_name in os.listdir(args.input_folder):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            continue

        image_path = os.path.join(args.input_folder, image_name)
        image_tensor, original_image = preprocess_image(image_path)
        boxes, scores = detect_objects(detection_model, image_tensor, args.detection_target)
        label_map = read_label_list(args.label_path)

        results = classify_objects(classification_model, original_image, boxes, label_map, species_list)

        save_results(original_image, results, args.output_folder, args.clasification_threshold, image_name)

    if db:
        db.close()



if __name__ == '__main__':
    main()
