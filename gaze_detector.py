import cv2
import numpy as np

class GazeDetector:
    def __init__(self):
        pass

    def get_eye_gaze(self, eye_roi):
        # Gray -> Threshold -> Center of Mass
        try:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            # Adaptive threshold? Or simple? 
            # Simple works if reliable lighting.
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Largest contour should be pupil
                max_cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    # cY = int(M["m01"] / M["m00"])
                    
                    # Normalized X (0 to 1) 
                    norm_x = cX / eye_roi.shape[1]
                    return norm_x
        except:
            pass
        return 0.5 # Center

    def get_gaze_direction(self, frame, keypoints):
        if not keypoints:
            return "Unknown", 0.5
            
        le = keypoints['left_eye']
        re = keypoints['right_eye']
        
        # Determine ROI size (based on distance between eyes)
        dist = np.linalg.norm(np.array(le) - np.array(re))
        eye_w = int(dist * 0.3)
        eye_h = int(eye_w * 0.6)
        
        # Crop Left Eye
        le_x, le_y = le
        le_roi = frame[max(0, le_y-eye_h//2):le_y+eye_h//2, max(0, le_x-eye_w//2):le_x+eye_w//2]
        
        # Crop Right Eye
        re_x, re_y = re
        re_roi = frame[max(0, re_y-eye_h//2):re_y+eye_h//2, max(0, re_x-eye_w//2):re_x+eye_w//2]
        
        l_gaze = self.get_eye_gaze(le_roi)
        r_gaze = self.get_eye_gaze(re_roi)
        
        avg_gaze = (l_gaze + r_gaze) / 2
        
        # Thresholds (Balanced)
        # Wide dead-zone: 0.35 to 0.65 is "Center"
        if avg_gaze < 0.35: 
            return "Right", avg_gaze
        elif avg_gaze > 0.65: 
            return "Left", avg_gaze
            
        return "Center", avg_gaze

