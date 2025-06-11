import cv2
import numpy as np
import random

def flatten_frame(frame, size=32):
    resized = cv2.resize(frame, (size, size))
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rounded = (np.round(rgb)).astype(np.uint8)
    return rounded.flatten().tolist()

def frames_similar(flat1, flat2, threshold=40): #
    diff = max(abs(a - b) for a, b in zip(flat1, flat2))
    return diff < threshold

def max_landmark_distance(lm1, lm2, img_width, img_height):
    if len(lm1) != len(lm2):
        raise ValueError(f"different shape: {len(lm1)}, {len(lm2)}")

    lm1 = np.array(lm1) / [img_width, img_height]
    lm2 = np.array(lm2) / [img_width, img_height]

    distances = np.linalg.norm(lm1 - lm2, axis=1)
    return np.max(distances)
def mean_landmark_distance(lm1, lm2, img_width, img_height):
    if len(lm1) != len(lm2):
        raise ValueError("different shape")

    lm1 = np.array(lm1) / [img_width, img_height]
    lm2 = np.array(lm2) / [img_width, img_height]

    distances = np.linalg.norm(lm1 - lm2, axis=1)
    return np.mean(distances)


class FrameCache:
    class block:
        def __init__(self, key, img, landmark):
            self.key = key
            self.img = img
            self.landmark = landmark
            pass
    def __init__(self, cachesize = 1):
        self.blocks = list(None for i in range(cachesize))
        self.cachesize = cachesize
    def query(self, key):
        for block in self.blocks:
            if (block != None) and frames_similar(key, block.key):
                return block.img, block.landmark
        return [], []
    def feed(self, key, img, landmark):
        new_block = FrameCache.block(key, img, landmark)
        self.blocks[random.randint(0, self.cachesize - 1)] = new_block