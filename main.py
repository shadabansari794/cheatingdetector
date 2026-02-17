import cv2
import time
import threading
from mtcnn import MTCNN
from gaze_detector import GazeDetector
from head_pose_estimator import HeadPoseEstimator
from expression_detector import ExpressionDetector
from object_detector import ObjectDetector

def main():
    # Initialize Detectors
    # Use MTCNN instead of MediaPipe (Python 3.13 fix)
    detector = MTCNN()
    
    gaze_detector = GazeDetector()
    head_pose_estimator = HeadPoseEstimator()
    object_detector = ObjectDetector() # YOLOv4-Tiny
    
    # Load Cheating Meme Image
    meme_path = "cheating_meme.jpg" # User needs to save the image as this name
    meme_img = cv2.imread(meme_path)
    if meme_img is None:
        print(f"Warning: {meme_path} not found. using text overlay instead.")
    else:
        # Resize meme to a fixed width (e.g., 350px) to fit webcam feed
        target_width = 350
        aspect_ratio = meme_img.shape[0] / meme_img.shape[1]
        target_height = int(target_width * aspect_ratio)
        meme_img = cv2.resize(meme_img, (target_width, target_height))
    
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Helper for threaded emotion detection
    emotion_result = "Neutral"
    frame_count = 0
    
    def update_emotion(start_frame):
        nonlocal emotion_result
        try:
            # Using DeepFace lightweight backend
            # Note: DeepFace is heavy, so we run it less frequently
            result = DeepFace.analyze(start_frame, actions=['emotion'], enforce_detection=False, silent=True)
            if result:
                 emotion_result = result[0]['dominant_emotion']
        except:
             pass

    from deepface import DeepFace # Import here to avoid lag on startup if possible
    
    print("Starting Cheating Detection System (MTCNN + YOLOv4-Tiny)...")
    print("Press 'q' to quit.")
    
    # Create Full Screen Window
    cv2.namedWindow('Cheating Detection System', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Cheating Detection System', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # State variables for optimization
    phone_detected = False
    phone_boxes = []
    
    # Calibration
    neutral_yaw = 0
    neutral_pitch = 0
    calibrated = False
    
    # Smoothing variables (Exponential Moving Average)
    alpha = 0.4 # Heavier smoothing for stability
    smooth_yaw = 0
    smooth_pitch = 0
    smooth_gaze = 0.5
    
    # Debounce variables
    suspicious_buffer = 0
    SUSPICIOUS_THRESHOLD_FRAMES = 8 # ~2 seconds at 4 FPS
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # 0. Mobile Phone Detection (YOLOv4-Tiny)
        # Run every 2 frames for better responsiveness
        if frame_count % 2 == 0:
             phone_detected, phone_boxes = object_detector.detect_phone(image)
        
        # MTCNN Detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_image)
        
        keypoints = None
        face_box = None
        
        if faces:
            # Take the first face (largest confidence)
            face = faces[0]
            face_box = face['box']
            keypoints = face['keypoints']
            
            # Draw Face Box
            x, y, w_box, h_box = face_box
            cv2.rectangle(image, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
            
            # Draw Keypoints
            for key, point in keypoints.items():
                cv2.circle(image, point, 2, (0, 255, 255), 2)
        
        # 1. Head Pose (Smoothed)
        pass_text, pitch, yaw, roll = head_pose_estimator.get_head_pose(keypoints)
        
        # Initialize smooth values on first frame
        if frame_count == 0:
            smooth_yaw = yaw
            smooth_pitch = pitch
            
        smooth_yaw = alpha * yaw + (1 - alpha) * smooth_yaw
        smooth_pitch = alpha * pitch + (1 - alpha) * smooth_pitch
        
        # Calibration Delta
        delta_yaw = smooth_yaw - neutral_yaw
        delta_pitch = smooth_pitch - neutral_pitch
        
        # Re-evaluate thresholds on DELTA values
        head_text = "Forward"
        THRESHOLD_YAW = 0.25 # Balanced: needs a clear turn
        
        if delta_yaw > THRESHOLD_YAW: head_text = "Looking Right"
        elif delta_yaw < -THRESHOLD_YAW: head_text = "Looking Left"
        
        # Pitch Delta
        if delta_pitch < -0.3: head_text = "Looking Up"
        elif delta_pitch > 0.3: head_text = "Looking Down" 
        
        # Debug Output on Screen
        cv2.putText(image, f"Y:{smooth_yaw:.2f} P:{smooth_pitch:.2f} | D_Y:{delta_yaw:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if not calibrated:
             cv2.putText(image, "Press 'c' to Calibrate Center", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # 2. Gaze Direction (Smoothed)
        pass_gaze, gaze_ratio = gaze_detector.get_gaze_direction(image, keypoints)
        
        # Handle case where gaze is 0 (no eyes found) - skip smoothing
        if gaze_ratio != 0:
             smooth_gaze = alpha * gaze_ratio + (1 - alpha) * smooth_gaze
             
        gaze_text = "Center"
        if smooth_gaze < 0.35: gaze_text = "Right"
        elif smooth_gaze > 0.65: gaze_text = "Left"
        
        # 3. Expression (Every 30 frames to save FPS)
        if frame_count % 30 == 0:
             threading.Thread(target=update_emotion, args=(image.copy(),)).start()
        
        frame_count += 1
        
        # --- LOGIC & ALERTS ---
        cheat_status = "Safe"
        color = (0, 255, 0) # Green
        is_currently_suspicious = False
        
        # Check suspicious conditions
        if phone_detected:
             is_currently_suspicious = True
             cheat_status = "Mobile Detected"
             
        elif "Left" in head_text or "Right" in head_text or "Up" in head_text:
             is_currently_suspicious = True
             cheat_status = "Looking Away"
             
        elif "Left" in gaze_text or "Right" in gaze_text:
             is_currently_suspicious = True
             cheat_status = "Eye Averaging"
        
        # Debounce Logic
        if is_currently_suspicious:
            suspicious_buffer += 1
        else:
            suspicious_buffer = max(0, suspicious_buffer - 4) # Fast recovery when looking straight
            
        # Trigger Alert only if buffer exceeds threshold
        display_status = "Safe"
        display_color = (0, 255, 0)
        
        if suspicious_buffer > SUSPICIOUS_THRESHOLD_FRAMES:
             display_status = "Suspicious: " + cheat_status
             display_color = (0, 0, 255) # Red
             
             # Overlay Meme
             if meme_img is not None:
                try:
                    x_offset = (w - meme_img.shape[1]) // 2
                    y_offset = h - meme_img.shape[0] - 50
                    if y_offset > 0 and x_offset > 0:
                        image[y_offset:y_offset+meme_img.shape[0], x_offset:x_offset+meme_img.shape[1]] = meme_img
                except:
                    pass
        
        if phone_detected: # Always draw box if detected, even if buffer building up
            for (px, py, pw, ph) in phone_boxes:
                cv2.rectangle(image, (px, py), (px+pw, py+ph), (0, 0, 255), 3)
                cv2.putText(image, "Mobile", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        if emotion_result in ["fear", "sad"]:
             display_status += " (Nervous?)"
             
        # --- VISUALIZATION ---
        
        # Dashboard UI
        cv2.rectangle(image, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(image, f"Status: {display_status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
        
        # Stats Box
        cv2.putText(image, f"Head: {head_text} ({delta_yaw:.2f})", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Gaze: {gaze_text} ({smooth_gaze:.2f})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Buffer: {suspicious_buffer}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Cheating Detection System', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
             neutral_yaw = smooth_yaw
             neutral_pitch = smooth_pitch
             calibrated = True
             print(f"Calibrated! Neutral Yaw: {neutral_yaw}, Pitch: {neutral_pitch}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
