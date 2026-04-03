# 🖱️ Air Mouse
**Controlling your PC cursor using hand gestures.**

This project uses Computer Vision and Hand Tracking to allow users to control their mouse movement and clicks without touching the physical device. It leverages **MediaPipe** for hand landmark detection and **PyAutoGUI** for controlling system inputs.
**Demo video at the very bottom**

---

### 🚀 Features
* **Cursor Movement:** Move your hand to move the cursor on the screen.
* **Clicking:** Use specific finger gestures to perform clicks.
* **Real-time Performance:** Low-latency tracking using a standard webcam.

---

### 🖐️ How to Use (Gestures)
| Action | Gesture |
| :--- | :--- |
| **Move Cursor** | Move your hand with fingers visible to the camera. |
| **Left Click** | Bring your **Index Finger** and **Thumb** close together (Pinch). |
| **Right Click** | Bring your **Middle Finger** and **Thumb** close together. |

---

### 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nikbeas8/Air-Mouse.git](https://github.com/nikbeas8/Air-Mouse.git)
   cd Air-Mouse

2. **Install Dependencies:**
   ```bash
   pip install opencv-python mediapipe pyautogui
   
3. **Check Environment:**
   ```bash
   python check_env.py

---

### 💻 How to Run
  **To start the air mouse application, execute the main script**:
  ```bash
  python final.py
```
---

### 📂 Project Structure
* **`final.py`**: The main application script.
* **`hand_landmarker.task`**: Pre-trained MediaPipe model.
* **`check_env.py`**: Environment verification script.
* **`webcam_check.py`**: Webcam access utility.


### 📺 Demo Video
<video src="https://github.com/user-attachments/assets/319568bd-3f0d-49bd-897d-c799c0bf3d8c" controls="controls" style="max-width: 100%; height: auto;"></video>

