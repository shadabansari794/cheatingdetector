from mtcnn import MTCNN
import cv2
import numpy as np

try:
    detector = MTCNN()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = detector.detect_faces(img)
    print("MTCNN Initialized and Ran Successfully")
except Exception as e:
    print(f"MTCNN Failed: {e}")
