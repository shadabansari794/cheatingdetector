import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, weights_path="yolov4-tiny.weights", cfg_path="yolov4-tiny.cfg", names_path="coco.names"):
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load names
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.CONFIDENCE_THRESHOLD = 0.45
        self.NMS_THRESHOLD = 0.4
        self.TARGET_CLASS = "cell phone"

    def detect_phone(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.CONFIDENCE_THRESHOLD:
                    # Check if it's a phone
                    if self.classes[class_id] == self.TARGET_CLASS:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        
        detected = False
        final_boxes = []
        
        if len(indices) > 0:
            detected = True
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                
        return detected, final_boxes

