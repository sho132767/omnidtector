import cv2
import numpy as np
import pyrealsense2 as rs
from collections import deque, defaultdict
from ultralytics import YOLOWorld

REAL_cloth_l = 0.300#m
REAL_cloth_s = 0.200   
persentage = 0.20 
BOMP_score = 0.88        
SHAPE_score = 0.78
corner = 4              
camera_far=0.500
MAX_THICKNESS = 12.0    
SIZE_RATIO = 0.80      
ditect_frame = 3 
W, H = 894, 894             

model = YOLOWorld('yolov8s-world.pt')
target_classes = ["towel", "cloth", "rag"]
model.set_classes(target_classes)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

profile = pipeline.start(config)


align = rs.align(rs.stream.color)
color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_profile.get_intrinsics()
FX, FY = intr.fx, intr.fy
CX, CY = intr.ppx, intr.ppy


depth_sensor = profile.get_device().first_depth_sensor()
DEPTH_SCALE = depth_sensor.get_depth_scale()

print(f"[INFO] カメラパラメータ: FX={FX:.1f}, FY={FY:.1f}, DepthScale={DEPTH_SCALE}")


def estimate_local_floor(depth_image, x1, y1, x2, y2, pad=15):
   
    y_min, y_max = max(0, y1 - pad), min(H, y2 + pad)
    x_min, x_max = max(0, x1 - pad), min(W, x2 + pad)

    outer_roi = depth_image[y_min:y_max, x_min:x_max]
    floor_mask = np.ones(outer_roi.shape, dtype=bool)
    in_y1, in_y2 = y1 - y_min, y2 - y_min
    in_x1, in_x2 = x1 - x_min, x2 - x_min
    floor_mask[in_y1:in_y2, in_x1:in_x2] = False

   
    valid_floor_depths = outer_roi[floor_mask & (outer_roi > 0)]

    if len(valid_floor_depths) > 20:
    
        return float(np.median(valid_floor_depths)) * DEPTH_SCALE
    return None



def get_cloth_mask(color_roi, depth_roi, local_floor_m):
    roi_h, roi_w = color_roi.shape[:2]
    depth_m = depth_roi.astype(np.float32) * DEPTH_SCALE
    if local_floor_m is not None:
        depth_mask = (depth_m > 0.05) & (depth_m < (local_floor_m - 0.005))
    else:
        depth_mask = np.zeros((roi_h, roi_w), dtype=bool)
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    _, color_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_mask = np.where(depth_mask, 255, color_mask).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask

def analyze_shape(mask_u8, box_w, box_h):

    ng_reasons = []
    debug = {}

  
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ["NoContour"], debug


    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 300: 
        return ["TooSmallContour"], debug
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    debug["convexity"] = convexity

    if convexity < BOMP_score:
        ng_reasons.append("Folded(Dent)") # 凹み・三角形折れ


    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    rect_area = rw * rh
    rectangularity = area / rect_area if rect_area > 0 else 0
    debug["rectangularity"] = rectangularity
    debug["min_area_rect"] = rect

    if rectangularity < SHAPE_score:
        ng_reasons.append("NotRectangular")


    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    n_corners = len(approx)
    debug["n_corners"] = n_corners
    debug["approx"] = approx

    if n_corners > corner:
        ng_reasons.append("CornerFolded")

    debug["contour"] = cnt
    return ng_reasons, debug

status_history = defaultdict(lambda: deque(maxlen=ditect_frame))

def stabilize_status(track_id, raw_status):
    
    status_history[track_id].append(raw_status)
    hist = status_history[track_id]

    if len(hist) < ditect_frame:
        return raw_status

    ng_list = [s for s in hist if s.startswith("NG")]
    if len(ng_list) > ditect_frame // 2:
      
        return max(set(ng_list), key=ng_list.count)
    else:
        return "OK (Straight)"




try:
    while True:
      
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

       
        results = model.track(frame, conf=persentage, persist=True, verbose=False)

        best_target = None
        min_dist = 999.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detected_name = target_classes[cls_id] if cls_id < len(target_classes) else "cloth"
                track_id = int(box.id[0]) if box.id is not None else -1

                
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                box_w = x2 - x1
                box_h = y2 - y1

                if box_w < 35 or box_h < 35:
                    continue

                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

              
                roi_depth = depth_image[y1:y2, x1:x2]
                valid_depths = roi_depth[roi_depth > 0]
                if len(valid_depths) < 30:
                    continue

            
                dist_forward = float(np.median(valid_depths)) * DEPTH_SCALE
                if dist_forward < 0.2 or dist_forward > 2.5:
                    continue

                
                local_floor_m = estimate_local_floor(depth_image, x1, y1, x2, y2)
                
                color_roi = frame[y1:y2, x1:x2]
                mask_u8 = get_cloth_mask(color_roi, roi_depth, local_floor_m)

                ng_reasons = []

                
                shape_ng, shape_debug = analyze_shape(mask_u8, box_w, box_h)
                ng_reasons.extend(shape_ng)

              
                exp_long_px  = (REAL_cloth_l * FX) / dist_forward
                exp_short_px = (REAL_cloth_s * FY) / dist_forward

                
                if "min_area_rect" in shape_debug:
                    (rw, rh) = shape_debug["min_area_rect"][1]
                    eff_long, eff_short = max(rw, rh), min(rw, rh)
                else:
                    eff_long, eff_short = max(box_w, box_h), min(box_w, box_h)

                
                if (eff_long < exp_long_px * SIZE_RATIO) or (eff_short < exp_short_px * SIZE_RATIO):
                    ng_reasons.append("TooSmall(Folded)")

               
                masked_depth_vals = roi_depth[(mask_u8 > 0) & (roi_depth > 0)]
                if len(masked_depth_vals) > 40:
                    depth_std_mm = float(np.std(masked_depth_vals.astype(np.float32) * DEPTH_SCALE)) * 1000.0
                    if depth_std_mm > MAX_THICKNESS:
                     ng_reasons.append("Wrinkles")

                raw_status = f"NG ({'/'.join(ng_reasons)})" if ng_reasons else "OK (Straight)"
                status = stabilize_status(track_id, raw_status) if track_id >= 0 else raw_status
                color = (0, 0, 255) if status.startswith("NG") else (0, 255, 0)

                
                if dist_forward < min_dist:
                    min_dist = dist_forward
                   
                    dist_side = ((cx - CX) / FX) * dist_forward
                    best_target = (x1, y1, x2, y2, cx, cy, conf, dist_forward, dist_side,
                                   status, color, detected_name, shape_debug)

        if best_target is not None:
            (x1, y1, x2, y2, cx, cy, conf, dist_forward, dist_side,
             status, color, detected_name, shape_debug) = best_target

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

         
            if "contour" in shape_debug:
                shifted_cnt = shape_debug["contour"] + np.array([x1, y1])
                cv2.drawContours(frame, [shifted_cnt], -1, (255, 200, 0), 2)

           
            info_text = f"[{status}] {detected_name}({conf:.2f}) Dist:{dist_forward:.2f}m Side:{dist_side:.2f}m"
            cv2.putText(frame, info_text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

         
            if "convexity" in shape_debug:
                dbg_text = (f"Conv:{shape_debug['convexity']:.2f} "
                            f"Rect:{shape_debug.get('rectangularity', 0):.2f} "
                            f"Corners:{shape_debug.get('n_corners', 0)}")
                cv2.putText(frame, dbg_text, (x1, min(H - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    
        cv2.line(frame, (int(CX), 0), (int(CX), H), (255, 0, 0), 1)

        cv2.imshow("Cloth Straight/Folded Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
  