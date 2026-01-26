import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# Movement Smoothing Variables
prev_x, prev_y = 0, 0
smoothing = 7 
margin = 120  # Frame margin for easier corner reaching

# State Toggles
is_dragging = False
right_clicked = False

# ==========================================
# MEDIAPIPE TASKS SETUP (2026 Standards)
# ==========================================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)
timestamp = 0

def draw_skeleton(frame, landmarks, w, h, color=(0, 255, 0)):
    """Manually draws the hand skeleton without mp.solutions"""
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),   # Index
        (9, 10), (10, 11), (11, 12),      # Middle
        (13, 14), (14, 15), (15, 16),     # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)         # Palm
    ]
    for start_idx, end_idx in connections:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, color, 2)

    for lm in landmarks:
        pt = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Process Frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        
        # 1. Landmark Extraction
        ttip = landmarks[4]   # Thumb
        itip = landmarks[8]   # Index
        mtip = landmarks[12]  # Middle
        ptip = landmarks[20]  # Pinky
        
        itip_p = (int(itip.x * w), int(itip.y * h))
        ttip_p = (int(ttip.x * w), int(ttip.y * h))
        mtip_p = (int(mtip.x * w), int(mtip.y * h))
        ptip_p = (int(ptip.x * w), int(ptip.y * h))

        # 2. Movement Logic (Smoothing + Scaling)
        screen_x = np.interp(ptip_p[0], (margin, w - margin), (0, screen_w))
        screen_y = np.interp(ptip_p[1], (margin, h - margin), (0, screen_h))
        
        curr_x = prev_x + (screen_x - prev_x) / smoothing
        curr_y = prev_y + (screen_y - prev_y) / smoothing
        
        pyautogui.moveTo(curr_x, curr_y, _pause=False)
        prev_x, prev_y = curr_x, curr_y

        # 3. Gesture Distances
        left_dist = np.hypot(itip_p[0] - ttip_p[0], itip_p[1] - ttip_p[1])
        right_dist = np.hypot(mtip_p[0] - ttip_p[0], mtip_p[1] - ttip_p[1])

        # 4. Action: Left Click / Drag
        if left_dist < 30:
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging = True
            cv2.putText(frame, "HOLD/DRAG", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(frame, itip_p, 10, (0, 255, 0), cv2.FILLED)
            skeleton_color = (0, 255, 0)
        else:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            skeleton_color = (255, 255, 255)

        # 5. Action: Right Click
        if right_dist < 30:
            if not right_clicked:
                pyautogui.rightClick()
                right_clicked = True
            cv2.putText(frame, "RIGHT CLICK", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.circle(frame, mtip_p, 10, (0, 255, 255), cv2.FILLED)

        else:
            right_clicked = False

        # 6. Draw Visuals
        draw_skeleton(frame, landmarks, w, h, skeleton_color)
        # cv2.line(frame, ttip_p, itip_p, (0, 255, 0), 2)
        cv2.circle(frame, ptip_p, 9, (255, 100, 25), -1) # Pinky cursor

    # UI Overlay
    cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (255, 255, 255), 1)
    cv2.imshow('Air Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()