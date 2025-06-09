import cv2
import mediapipe as mp
from data_generate import normalize_landmarks_by_face_extent
import csv
import time
from facial_landmark import FaceMeshDetector

detector = FaceMeshDetector()
cap = cv2.VideoCapture(0)

output_csv = "custom_emotion_data.csv"
emotion_keys = {'a': 0, 'h': 1, 'n': 2, 's': 3}

def normalize_landmarks(landmarks):
    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0: scale = 1e-6
    return [((x - min_x) / scale, (y - min_y) / scale) for (x, y) in landmarks]

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [f'f{i}' for i in range(956)] + ['label']
    writer.writerow(header)

    print("[INFO] Press A (angry), H (happy), N (Neutral), S (Sad) to label a sample. Q to quit.")
    cnt = [0] * 4
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_facemesh, faces = detector.findFaceMesh(frame)

        if not len(faces):
            continue

        cv2.imshow("Webcam (Press a/h/n/s/q)", frame)
        key = cv2.waitKey(1) & 0xFF
        char = chr(key).lower()

        if char in emotion_keys:
            face = faces[0]
            landmarks = normalize_landmarks_by_face_extent(face)

            flat = [coord for xy in landmarks for coord in xy]
            if len(flat) != 956:  # ensure correct shape
                continue
            label = emotion_keys[char]
            writer.writerow(flat + [label])
            cnt[label] += 1
            counter += 1
            if (counter % 100 == 0):
                print(f"Total: {counter}, labels: {cnt}")
            #print(f"[+] Saved sample as {label} at {time.strftime('%X')}")

        if char == 'q':
            break
        
cap.release()
cv2.destroyAllWindows()