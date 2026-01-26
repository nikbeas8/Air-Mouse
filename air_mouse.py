import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup Screen & Smoothing
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothing = 5  # Increase for smoother, decrease for faster movement
margin = 100   # Camera frame margin to make reaching corners easier

# 2. Setup Task (Ensure hand_landmarker.task is in folder)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

right_clicked = False

is_dragging = False

scroll_mode = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:


        for landmarks in result.hand_landmarks:
        
            # 1. Define the connection pairs (the "bones")
            # Structure: (starting_landmark_index, ending_landmark_index)
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index Finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle Finger
                (9, 10), (10, 11), (11, 12),
                # Ring Finger
                (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm / Knuckle connections
                (5, 9), (9, 13), (13, 17)
            ]

            # 2. Draw the Lines
            for connection in connections:
                start_idx, end_idx = connection
                
                # Get landmarks
                start_lm = landmarks[start_idx]
                end_lm = landmarks[end_idx]
                
                # Convert normalized (0-1) to pixel coordinates
                pt1 = (int(start_lm.x * w), int(start_lm.y * h))
                pt2 = (int(end_lm.x * w), int(end_lm.y * h))
                
                # Draw the line (B, G, R) -> Green color here
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

            # 3. Draw the Joints (Optional, but looks better)
            for lm in landmarks:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, pt, 5, (255, 255, 255), -1)
                cv2.circle(frame, pt, 4, (0, 0, 255), -1) # Red dots for joints



        landmarks = result.hand_landmarks[0]
        # Index Tip is landmark 8
        ptip = landmarks[20]
        
        # 3. Map Coordinates with Margin
        # We scale the coordinates so the 'active area' is smaller than the full frame
        raw_x = int(ptip.x * w)
        raw_y = int(ptip.y * h)
        
        # Linear Interpolation (Mapping)
        # Using a margin helps you reach screen edges without hand leaving camera view
        screen_x = np.interp(raw_x, (margin, w - margin), (0, screen_w))
        screen_y = np.interp(raw_y, (margin, h - margin), (0, screen_h))

        # 4. Apply Smoothing Logic
        curr_x = prev_x + (screen_x - prev_x) / smoothing
        curr_y = prev_y + (screen_y - prev_y) / smoothing
        
        # Move the actual mouse cursor
        pyautogui.moveTo(curr_x, curr_y, _pause=False)
        
        prev_x, prev_y = curr_x, curr_y

        # Index Tip is landmarks[8], Thumb Tip is landmarks[4]
        itip = landmarks[8]
        ttip = landmarks[4]
        mtip = landmarks[12]

        # Get pixel coordinates for all three
        itip_x, itip_y = int(itip.x * w), int(itip.y * h)
        ttip_x, ttip_y = int(ttip.x * w), int(ttip.y * h)
        mtip_x, mtip_y = int(mtip.x * w), int(mtip.y * h)

        # Calculate distance between thumb and index
        left_dist = np.hypot(itip_x - ttip_x, itip_y - ttip_y)
        right_dist = np.hypot(mtip_x - ttip_x, mtip_y - ttip_y)

        # Visual feedback: Draw a line between fingers
        cv2.line(frame, (itip_x, itip_y), (ttip_x, ttip_y), (0, 255, 0), 2)
        cv2.circle(frame, (raw_x, raw_y), 9, (255, 120, 0), cv2.FILLED)
        # Trigger Click
        # if distance < 20:
        #     if not clicked:
        #         pyautogui.click()
        #         clicked = True
        #         print("Click Registered!")
        #     # Visual feedback when held
        #     cv2.circle(frame, (itip_x, itip_y), 15, (0, 0, 255), cv2.FILLED)
        # else:
        #     clicked = False # Reset once fingers are apart
        #     # Normal movement color
        #     cv2.circle(frame, (itip_x, itip_y), 10, (255, 0, 255), cv2.FILLED)

        # Left click / hold and drag
        if left_dist < 20:
            if not is_dragging:
                pyautogui.mouseDown()
                is_dragging = True
                print("Mouse Down - Dragging")
            
            # Visual feedback: Change color to show "Holding"
            cv2.circle(frame, (itip_x, itip_y), 11, (0, 0, 0), cv2.FILLED) 
        else:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
                print("Mouse Up - Dropped")

        # Right click
        if right_dist < 20:
            if not right_clicked:
                pyautogui.rightClick()
                right_clicked = True
                print("Right Click!")
            # Visual feedback: Yellow circle for Right Click
            cv2.circle(frame, (mtip_x, mtip_y), 11, (0, 255, 255), cv2.FILLED)
        else:
            right_clicked = False

       

        # --- Inside landmarks loop ---
        # 1. Calculate distance between Index(8) and Middle(12) tips
        scroll_dist = np.hypot(itip_x - mtip_x, itip_y - mtip_y)

        # 2. Check if they are close enough to "lock" into scroll mode
        # (Distance should be small, but they shouldn't be pinching the thumb)
        if scroll_dist < 40 and left_dist > 50 and right_dist > 50:
            scroll_mode = True
            
            # Calculate how much the hand moved vertically
            # Using 'prev_y' from our smoothing logic
            diff_y = prev_y - curr_y 
            
            # if abs(diff_y) > 1: # Threshold to prevent accidental tiny scrolls
            pyautogui.scroll(int(diff_y * 12)) # Multiply by 2 for faster scrolling
                
            # Visual feedback for Scroll Mode
            cv2.putText(frame, "SCROLL MODE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.line(frame, (itip_x, itip_y), (mtip_x, mtip_y), (255, 255, 0), 5)
        else:
            scroll_mode = False


    cv2.imshow('Air Mouse Movement', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()