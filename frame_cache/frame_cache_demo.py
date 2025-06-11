"""
FaceMeshDetector is copied from facial_landmark.py
"""

import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.faces = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = self.drawing_spec,
                        connection_drawing_spec = self.drawing_spec)

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])


                self.faces.append(face)

        return img, self.faces


import os, time
def get_cpu_time():
    return os.times().user + os.times().system


from frame_cache import *
import traceback
def main():

    detector = FaceMeshDetector()

    cap = cv2.VideoCapture('detector_test2.mp4')
    counter = 0

    start_cpu = get_cpu_time()
    start_time = time.time()
    cache = FrameCache()
    hit = 0
    dis_sum, dis_sum2 = 0, 0

    try:
        while cap.isOpened():
            counter = counter + 1
            success, img = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                break

            key = flatten_frame(img)

            result_img, result_landmark = cache.query(key)


            if len(result_img) > 0:
                hit += 1

                # To print the accuracy drop (aka distance), uncomment the following codes
                # The CPU usage would be affected since it will run the original model prediction for comparison 
                # Must test CPU usage and Distances seperate on separate runs

                # height, width = img.shape[:2]
                # _faces = result_landmark
                # img, faces = detector.findFaceMesh(img)
                
                # dis_sum += max_landmark_distance(faces, _faces, width, height)
                # dis_sum2 += mean_landmark_distance(faces, _faces, width, height)

            else:
                result_img, result_faces = detector.findFaceMesh(img)
                if len(result_faces) == 0:
                    continue
                cache.feed(key, result_img, result_faces) #

            cv2.imshow('MediaPipe FaceMesh', result_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("error0:", e)
        traceback.print_exc()

    cap.release()
    #out.release()
    end_cpu = get_cpu_time()
    end_time = time.time()
    
    elapsed = end_time - start_time
    cpu_usage_percent = 100 * (end_cpu - start_cpu) / elapsed
    print(f"total {counter} frames")
    print(f"FPS: {counter / elapsed:.4f}")
    print(f"Estimated CPU usage: {cpu_usage_percent:.2f}%")
    print(f"Hit rate: {hit / counter}")
    print(f"Avg Max Distance: {dis_sum / counter}")
    print(f"Avg Mean Distance: {dis_sum2 / counter}")


if __name__ == "__main__":
    # demo code
    main()
