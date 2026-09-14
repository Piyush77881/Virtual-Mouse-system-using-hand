import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize
hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Cursor smoothness
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# FPS
pTime = 0

# Drag flag
dragging = False

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame wi
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Important finger landmarks
            index = landmarks[8]
            middle = landmarks[12]
            ring = landmarks[16]
            thumb = landmarks[4]

            index_x = int(index.x * frame_width)
            index_y = int(index.y * frame_height)
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)
            middle_x = int(middle.x * frame_width)
            middle_y = int(middle.y * frame_height)
            ring_x = int(ring.x * frame_width)
            ring_y = int(ring.y * frame_height)

            # Draw visual markers
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 8, (0, 255, 0), -1)

            # Map hand movement to screen
            screen_x = screen_width / frame_width * index_x
            screen_y = screen_height / frame_height * index_y

            # Smooth cursor movement
            clocX = plocX + (screen_x - plocX) / smoothening
            clocY = plocY + (screen_y - plocY) / smoothening
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

            # Calculate distances
            index_thumb_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
            index_middle_dist = math.hypot(index_x - middle_x, index_y - middle_y)

            # Left click (index + thumb close)
            if index_thumb_dist < 25:
                cv2.putText(frame, "Left Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                pyautogui.click()
                pyautogui.sleep(1)

            # Right click (index + middle close)
            elif index_middle_dist < 25:
                cv2.putText(frame, "Right Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                pyautogui.rightClick()
                pyautogui.sleep(1)

            # Scroll (index and thumb far apart)
            elif index_thumb_dist > 100:
                cv2.putText(frame, "Scrolling...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                pyautogui.scroll(-50)

            # Drag & drop (index + middle + thumb close)
            elif index_thumb_dist < 35 and index_middle_dist < 35:
                if not dragging:
                    cv2.putText(frame, "Dragging...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    # FPS counter
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show window
    cv2.imshow('Virtual Mouse', frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
