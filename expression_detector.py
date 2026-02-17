import cv2
from deepface import DeepFace
import numpy as np

class ExpressionDetector:
    def __init__(self):
        pass

    def get_expression(self, frame):
        try:
            # Analyze the frame for emotion
            # actions=['emotion'] speeds it up slightly
            # enforce_detection=False prevents crash if face not found immediately
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            if result:
               # Result is a list in recent versions
               return result[0]['dominant_emotion']
            return "Neutral"
        except Exception as e:
            # print(f"Error in expression detection: {e}")
            return "Neutral"
