import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
label_recover = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
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
def align_landmarks(landmarks):
    """
    landmarks: list of 478 (x, y) tuples in image coordinates
    returns: aligned landmarks list
    """
    # Example: use left eye corner (33) and right eye corner (263)
    left_eye = np.array(landmarks[468])
    right_eye = np.array(landmarks[473])
    if (right_eye[0] < left_eye[0]): 
        left_eye, right_eye = right_eye, left_eye

    # Compute the angle of the eye line
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.arctan2(dy, dx)

    # Rotate all landmarks to align eyes horizontally
    center = np.mean([left_eye, right_eye], axis=0)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    def rotate(pt):
        shifted = np.array(pt) - center
        rotated = [
            shifted[0] * cos_a - shifted[1] * sin_a,
            shifted[0] * sin_a + shifted[1] * cos_a
        ]
        return rotated

    aligned = [rotate(pt) for pt in landmarks]
    return aligned
def align_row(row):
    coords = [(row[f"f{2*i}"], row[f"f{2*i+1}"]) for i in range(478)]
    aligned = align_landmarks(coords)
    flat = [coord for pt in aligned for coord in pt]
    return pd.Series(flat + [row["label"]])

def important_indices():
    LEFT_EYEBROW = [55, 65, 52, 53, 46, 70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [285, 295, 282, 283, 276, 336, 296, 334, 293, 300]
    LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
    RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
    OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
    INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    return LEFT_EYEBROW + RIGHT_EYEBROW + LEFT_EYE + RIGHT_EYE + OUTER_LIPS + INNER_LIPS
def preprocess(landmarks):
    landmarks = normalize_landmarks_by_face_extent(landmarks)
    landmarks = align_landmarks(landmarks)
    imp_indices = important_indices()
    new_landmarks = [(landmarks[i][0], landmarks[i][1]) for i in imp_indices]
    return new_landmarks
class EmotionNet(nn.Module):
    def __init__(self, input_size=len(important_indices())*2, num_classes=7):  # adjust num_classes if needed
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def select_columns(X):
    indices = important_indices()
    cols = []
    for i in indices:
        cols.append(f'f{2*i}')     # x_i
        cols.append(f'f{2*i + 1}') # y_i
    print(len(cols))
    print(X[cols].shape)
    return X[cols]

class Stable_Emotion_Predictor():
    def __init__(self, model_path, window_size=10):
        self.model = EmotionNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.cnt = [0] * 7
        self.window = []
        self.window_size = window_size
    def feed(self, frame):
        flat = [coord for pt in preprocess(frame) for coord in pt]
        input_tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)    
            pred = torch.argmax(probs, dim=1).item()
            #prob_values = probs.squeeze().tolist()
        self.cnt[pred] += 1
        self.window.append(pred)
        if len(self.window) > self.window_size:
            self.cnt[self.window[0]] -= 1
            self.window = self.window[1:]
    def predict(self):
        if len(self.window) < self.window_size:
            return "neutral"
        return label_recover[np.argmax(self.cnt)]
