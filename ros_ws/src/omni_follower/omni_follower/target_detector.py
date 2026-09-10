import cv2
import numpy as np
from ultralytics import YOLOWorld

CAMERA_INDEX = 6
REAL_WIDTH_M = 0.25         
FOV_H_DEG = 69.0          
CONFIDENCE_THRESHOLD = 0.70 



model = YOLOWorld('yolov8s-worldv2.pt')


target_classes = ["transparent bucket", "clear bucket"]
model.set_classes(target_classes)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

FOV_H_RAD = np.deg2rad(FOV_H_DEG)
print("Video stream started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    fx = (w / 2.0) / np.tan(FOV_H_RAD / 2.0)

    results = model.track(frame, conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False)

    best_target = None
    min_dist = 999.0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            box_w = x2 - x1
            box_h = y2 - y1

          
            if box_w < 30 or box_h < 30:
                continue

            
            aspect_ratio = box_h / box_w
            if not (0.5 < aspect_ratio < 1.8):
                continue

          
            dist_x = (fx * REAL_WIDTH_M) / box_w

    
            if dist_x < min_dist:
                min_dist = dist_x
                cx = (x1 + x2) / 2.0
                dx_pixel = cx - (w / 2.0)
                theta = (dx_pixel / (w / 2.0)) * (FOV_H_RAD / 2.0)
                dist_y = -dist_x * np.tan(theta)
                best_target = (x1, y1, x2, y2, conf, dist_x, dist_y)

  
    if best_target is not None:
        x1, y1, x2, y2, conf, dist_x, dist_y = best_target
        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, (0, 0, 255), -1)

        info_text = f"BUCKET ({conf:.2f}) | {dist_x:.2f}m | Side: {dist_y:.2f}m"
        cv2.putText(frame, info_text, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
    cv2.imshow("RealSense RGB Bucket Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
