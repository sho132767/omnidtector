from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLOWorld

REAL_cloth_l = 0.300  
REAL_cloth_s = 0.200  
persentage = 0.10  
BOMP_score = 0.90 
SHAPE_score = 0.80
corner = 4 
ditect_frame = 5  
W, H = 640, 640  


model = YOLOWorld('yolov8s-world.pt')
target_classes = ['towel', 'cloth', 'rag']
model.set_classes(target_classes)

CAMERA_ID = 4

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)


cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 30.0)


actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



def get_cloth_mask(color_roi):
  if color_roi is None or color_roi.size == 0:
    return np.zeros((10, 10), dtype=np.uint8)

  gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
  _, color_mask = cv2.threshold(
      gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
  )

  kernel = np.ones((7, 7), np.uint8)
  combined_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
  combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(
      combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if contours:
    main_cnt = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(combined_mask)
    cv2.drawContours(clean_mask, [main_cnt], -1, 255, thickness=cv2.FILLED)
    return clean_mask

  return combined_mask


def analyze_shape(mask_u8, box_w, box_h):
  ng_reasons = []
  debug = {}

  contours, _ = cv2.findContours(
      mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return ['NoContour'], debug

  cnt = max(contours, key=cv2.contourArea)
  area = cv2.contourArea(cnt)
  if area < 300:
    return ['TooSmallContour'], debug

  hull = cv2.convexHull(cnt)
  hull_area = cv2.contourArea(hull)
  convexity = area / hull_area if hull_area > 0 else 0
  debug['convexity'] = convexity

  if convexity < BOMP_score:
    ng_reasons.append('Folded(Dent)')


  rect = cv2.minAreaRect(cnt)
  (rw, rh) = rect[1]
  rect_area = rw * rh if rw > 0 and rh > 0 else 1
  rectangularity = area / rect_area if rect_area > 0 else 0
  debug['rectangularity'] = rectangularity
  debug['min_area_rect'] = rect

  if rectangularity < SHAPE_score:
    ng_reasons.append('NotRectangular')

 
  peri = cv2.arcLength(cnt, True)
  approx = cv2.approxPolyDP(
      cnt, 0.04 * peri, True
  ) 
  n_corners = len(approx)
  debug['n_corners'] = n_corners
  debug['approx'] = approx

  if n_corners > corner + 2:
    ng_reasons.append('CornerFolded')


  if rw > 0 and rh > 0:
    eff_long, eff_short = max(rw, rh), min(rw, rh)
    measured_aspect = eff_long / eff_short
    real_aspect = max(REAL_cloth_l, REAL_cloth_s) / min(
        REAL_cloth_l, REAL_cloth_s
    )

    debug['measured_aspect'] = measured_aspect
    debug['real_aspect'] = real_aspect

  
    aspect_tolerance = 0.20
    if abs(measured_aspect - real_aspect) > (real_aspect * aspect_tolerance):
      ng_reasons.append('BadAspect(Ratio)')

  debug['contour'] = cnt
  return ng_reasons, debug


status_history = defaultdict(lambda: deque(maxlen=ditect_frame))


def stabilize_status(track_id, raw_status):
  status_history[track_id].append(raw_status)
  hist = status_history[track_id]

  if len(hist) < ditect_frame:
    return raw_status

  ng_list = [s for s in hist if s.startswith('NG')]
  if len(ng_list) > ditect_frame // 2:
    return max(set(ng_list), key=ng_list.count)
  else:
    return 'OK (Straight)'


try:
  while True:
    ret, frame = cap.read()
    current_h, current_w = frame.shape[:2]
    CX, CY = current_w / 2.0, current_h / 2.0

   
    results = model.track(frame, conf=persentage, persist=True, verbose=False)

    best_target = None
    max_conf = -1.0

    for result in results:
      if result.boxes is None:
        continue

      for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        detected_name = (
            target_classes[cls_id] if cls_id < len(target_classes) else 'cloth'
        )
        track_id = int(box.id[0]) if box.id is not None else -1

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(current_w, x2), min(current_h, y2)
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 35 or box_h < 35:
          continue

        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)

        color_roi = frame[y1:y2, x1:x2]
        if color_roi.size == 0:
          continue

     
        mask_u8 = get_cloth_mask(color_roi)
        ng_reasons = []
        shape_ng, shape_debug = analyze_shape(mask_u8, box_w, box_h)
        ng_reasons.extend(shape_ng)

       
        raw_status = (
            f"NG ({'/'.join(ng_reasons)})" if ng_reasons else 'OK (Straight)'
        )
        status = (
            stabilize_status(track_id, raw_status)
            if track_id >= 0
            else raw_status
        )
        color = (0, 0, 255) if status.startswith('NG') else (0, 255, 0)

        
        if conf > max_conf:
          max_conf = conf
          best_target = (
              x1,
              y1,
              x2,
              y2,
              cx,
              cy,
              conf,
              status,
              color,
              detected_name,
              shape_debug,
          )

    if best_target is not None:
      (
          x1,
          y1,
          x2,
          y2,
          cx,
          cy,
          conf,
          status,
          color,
          detected_name,
          shape_debug,
      ) = best_target

      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

      if 'contour' in shape_debug:
        shifted_cnt = shape_debug['contour'] + np.array([x1, y1])
        cv2.drawContours(frame, [shifted_cnt], -1, (255, 200, 0), 2)

      info_text = (
          f'[{status}] {detected_name}({conf:.2f}) Pos:({cx},{cy})'
      )
      cv2.putText(
          frame,
          info_text,
          (x1, max(20, y1 - 10)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.55,
          color,
          2,
      )

      if 'convexity' in shape_debug:
        dbg_text = (
            f"Conv:{shape_debug['convexity']:.2f} "
            f"Rect:{shape_debug.get('rectangularity', 0):.2f} "
            f"Aspect:{shape_debug.get('measured_aspect', 0):.2f}"
        )
        cv2.putText(
            frame,
            dbg_text,
            (x1, min(current_h - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    cv2.line(frame, (int(CX), 0), (int(CX), current_h), (255, 0, 0), 1)

    cv2.imshow('Cloth Straight/Folded Tracker (Ubuntu)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

finally:
  cap.release()
  cv2.destroyAllWindows()