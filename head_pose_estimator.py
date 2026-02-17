import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        pass

    def get_head_pose(self, keypoints):
        # keypoints: {'left_eye': (x,y), 'right_eye': (x,y), 'nose': (x,y), ...}
        if not keypoints:
            return "Unknown", 0, 0, 0

        le = np.array(keypoints['left_eye'])
        re = np.array(keypoints['right_eye'])
        n = np.array(keypoints['nose'])
        mj = np.array(keypoints['mouth_left'])
        mr = np.array(keypoints['mouth_right'])
        
        # New Yaw Logic: Nose X relative to Eye Midpoint X
        # Normalized by Eye Distance
        
        eye_mid = (le + re) / 2
        eye_dist = np.linalg.norm(re - le)
        
        if eye_dist == 0: return "Unknown", 0, 0, 0
        
        # Yaw Ratio: (NoseX - MidX) / Dist
        # If Nose is to the right of mid (Image Right) -> Turns Positive
        # In Flipped Image (Mirror): Looking Right -> Nose moves Right -> Positive
        
        yaw_ratio = (n[0] - eye_mid[0]) / eye_dist 
        
        # Pitch: Nose Y relative to Eye Mid Y (normalized by face height approx)
        mouth_mid = (mj + mr) / 2
        face_height = np.linalg.norm(eye_mid - mouth_mid)
        pitch_ratio = (n[1] - eye_mid[1]) / face_height # Smaller = Nose higher (Up)
        
        text = "Forward"
        
        # Thresholds (Linear Projection Method)
        # Center is roughly 0.
        # Looking Right -> Positive > 0.1?
        # Looking Left -> Negative < -0.1?
        
        # Note: In Mirror View:
        # Looking Right -> Nose moves Right -> Positive Ratio
        # Looking Left -> Nose moves Left -> Negative Ratio
        
        THRESHOLD_YAW = 0.1
        
        if yaw_ratio > THRESHOLD_YAW:
             text = "Looking Right"
        elif yaw_ratio < -THRESHOLD_YAW:
             text = "Looking Left"
        
        # Pitch
        # Normal is around 0.3-0.4?
        # Looking Up -> Nose closer to eyes -> Ratio decreases (< 0.2)
        # Looking Down -> Nose further -> Ratio increases (> 0.6)
        
        if pitch_ratio < 0.25:
             text = "Looking Up"
        elif pitch_ratio > 0.55: # Re-tuned
             text = "Looking Down"

        return text, pitch_ratio, yaw_ratio, 0

