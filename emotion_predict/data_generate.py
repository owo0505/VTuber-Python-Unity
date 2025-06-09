from torchvision import datasets, transforms
from facial_landmark import FaceMeshDetector
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import cv2
import sys
import csv


def normalize_landmarks_by_face_extent(landmarks):
    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max_x - min_x
    range_y = max_y - min_y

    # Use the larger of the two as scale factor
    scale = max(range_x, range_y)

    if scale == 0:  # Avoid divide-by-zero
        scale = 1e-6
    normalized = [((x - min_x) / scale, (y - min_y) / scale) for (x, y) in landmarks]
    return normalized

if __name__ == "__main__":

    input_root = Path("test")          # e.g., /happy/img0.png

    detector = FaceMeshDetector()
    class_to_idx = {d.name: idx for idx, d in enumerate(input_root.iterdir()) if d.is_dir()}
    print(class_to_idx)

    counter = 0
    with open("emotion_test.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"f{i}" for i in range(956)] + ["label"]
        writer.writerow(header)


        for class_dir in input_root.iterdir():
            if not class_dir.is_dir():
                continue

            label = class_to_idx[class_dir.name]
            print(f"processing: {class_dir.name}")
            for img_path in class_dir.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                #print(img_path)
                #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_facemesh, faces = detector.findFaceMesh(img)
                #print(img_path)
                if len(faces):
                    face = faces[0]
                    landmarks = normalize_landmarks_by_face_extent(face)

                    flat = [coord for xy in landmarks for coord in xy]
                    if len(flat) != 956:  # ensure correct shape
                        continue
                    writer.writerow(flat + [label])
                    counter += 1
                    if (counter % 100 == 0):
                        print(f"processed: {counter} images")

                # if results.detections:
                #     for i, detection in enumerate(results.detections):
                        
                #         cv2.imwrite(str(out_path), face)