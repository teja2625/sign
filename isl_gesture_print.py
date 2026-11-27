"""Indian Sign Language (ISL) detector — OpenCV + MediaPipe, minimal setup."""

import math
import time
import cv2
import mediapipe as mp

# --- config ---
SERIAL_PORT, BAUD = "COM3", 115200
CAMERA, STABLE = 0, 5
USE_SERIAL = True

esp = None
if USE_SERIAL:
    try:
        import serial
        esp = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
        time.sleep(2)
        print(f"ESP32 on {SERIAL_PORT}")
    except Exception as e:
        print(f"No serial ({e}) — camera-only mode.")
        esp = None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Active ISL signs shown and detected (I Love You disabled)
ISL_SIGNS = ("All the best", "Peace", "Okay", "Call Me", "Stop")
ISL_HINT = "All the best | Peace | Okay | Call Me | Stop"


def to_xy(landmarks, shape):
    h, w = shape[:2]
    return [[int(p.x * w), int(p.y * h)] for p in landmarks]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_extended(xy, tip, pip, mcp):
    return xy[tip][1] < xy[pip][1] - 20 and xy[pip][1] < xy[mcp][1]


def thumb_extended(xy):
    return abs(xy[4][0] - xy[5][0]) > 60


def thumb_up(xy):
    return xy[4][1] < xy[3][1] - 30 and abs(xy[4][0] - xy[5][0]) > 40


def finger_count(xy):
    return [
        int(thumb_extended(xy)),
        int(finger_extended(xy, 8, 6, 5)),
        int(finger_extended(xy, 12, 10, 9)),
        int(finger_extended(xy, 16, 14, 13)),
        int(finger_extended(xy, 20, 18, 17)),
    ]


def recognize_isl(xy):
    """ISL signs — Indian Sign Language vocabulary."""
    f = finger_count(xy)
    ti = dist(xy[4], xy[8])
    tp = dist(xy[4], xy[20])
    im = dist(xy[8], xy[12])
    palm = dist(xy[5], xy[17])
    wrist_y = xy[0][1]

    # 1. All the best — ISL thumbs-up
    if f[0] == 1 and f[1:] == [0, 0, 0, 0] and thumb_up(xy):
        return "All the best"

    # 2. Call Me — thumb + pinky (shaka)
    if f == [1, 0, 0, 0, 1] and tp > palm * 1.5:
        return "Call Me"

    # 3. Okay — thumb-index circle
    if f[0] == 1 and (f[2] == 1 or f[3] == 1 or f[4] == 1) and ti < palm * 0.7:
        return "Okay"

    # 4. Peace — index + middle (V sign)
    if f == [0, 1, 1, 0, 0] and im > palm * 0.4:
        return "Peace"

    # 5. Stop — open palm, all fingers up
    if f == [1, 1, 1, 1, 1]:
        tips = [4, 8, 12, 16, 20]
        if all(xy[t][1] < wrist_y for t in tips):
            return "Stop"

    # I Love You — disabled
    # if f == [1, 1, 0, 0, 1]:
    #     if xy[12][1] > xy[10][1] and xy[16][1] > xy[14][1]:
    #         return "I Love You"

    return None


def send(text):
    if esp:
        try:
            esp.write((text + "\n").encode())
            print(f"Sent: {text}")
        except Exception:
            pass


cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise SystemExit(f"Camera {CAMERA} failed — try CAMERA = 1")

buf, confirmed = [], ""
print("ISL signs:", " | ".join(ISL_SIGNS))
print("Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    sign = None

    if res.multi_hand_landmarks:
        hl = res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        xy = to_xy(hl.landmark, frame.shape)
        sign = recognize_isl(xy)

    if sign:
        buf.append(sign)
        if len(buf) > STABLE:
            buf.pop(0)
        if len(buf) == STABLE and buf.count(sign) == STABLE and confirmed != sign:
            confirmed = sign
            print(f"ISL: {confirmed}")
            send(confirmed)
    else:
        buf.clear()

    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    label = confirmed if confirmed else "Show ISL sign"
    cv2.putText(frame, f"ISL: {label}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0) if confirmed else (0, 0, 255), 2)
    cv2.putText(frame, ISL_HINT, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.imshow("ISL Gesture Print", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
if esp:
    esp.close()
