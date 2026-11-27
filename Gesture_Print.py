
"""
Gesture Print Detector with ESP32 Wi-Fi Output
Original 6 gestures + 10 communication signs.
Uses Wi-Fi TCP instead of USB Serial.
No COM port required.
"""

import math
import time
import cv2
import mediapipe as mp
import socket

# ========================
# === USER PARAMETERS ====
# ========================

ESP32_IP = "192.168.4.1"
ESP32_PORT = 8080

DETECTION_CONFIDENCE = 0.75
TRACKING_CONFIDENCE = 0.70

STABILITY_FRAMES = 5
RESET_AFTER_NO_GESTURE = 20

ENABLE_WIFI = True
CAMERA_INDEX = 0


class GesturePrintDetector:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

        self.gesture_buffer = []
        self.confirmed_gesture = ""
        self.no_gesture_frames = 0
        self.debug = []

        self.esp = None

        # Connect to ESP32 through Wi-Fi
        if ENABLE_WIFI:
            self.connect_to_esp32()
        else:
            print("[INFO] Wi-Fi output disabled.")

    # ========================
    # === ESP32 WIFI CONNECT ==
    # ========================

    def connect_to_esp32(self):

        try:
            if self.esp:
                self.esp.close()

            print("[INFO] Connecting to ESP32 Wi-Fi server...")

            self.esp = socket.create_connection(
                (ESP32_IP, ESP32_PORT),
                timeout=3
            )

            print("[INFO] ESP32 connected through Wi-Fi!")
            print(f"[INFO] IP: {ESP32_IP}:{ESP32_PORT}")

            return True

        except OSError as exc:

            self.esp = None

            print(f"[WARN] ESP32 connection failed: {exc}")
            print("[INFO] Will retry when sending the next gesture.")

            return False

    # ========================
    # === LANDMARK FUNCTIONS =
    # ========================

    @staticmethod
    def to_pixel_landmarks(hand_landmarks, frame_shape):

        height, width, _ = frame_shape
        points = []

        for landmark in hand_landmarks.landmark:
            points.append([
                int(landmark.x * width),
                int(landmark.y * height)
            ])

        return points

    @staticmethod
    def distance(p1, p2):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    @staticmethod
    def is_finger_extended(px_landmarks, tip, pip, mcp):

        tip_y = px_landmarks[tip][1]
        pip_y = px_landmarks[pip][1]
        mcp_y = px_landmarks[mcp][1]

        return tip_y < pip_y - 20 and pip_y < mcp_y

    @staticmethod
    def is_thumb_extended(px_landmarks):

        thumb_tip = px_landmarks[4]
        index_mcp = px_landmarks[5]

        return abs(thumb_tip[0] - index_mcp[0]) > 55

    @staticmethod
    def is_thumb_up(px_landmarks):

        thumb_tip = px_landmarks[4]
        thumb_ip = px_landmarks[3]
        index_mcp = px_landmarks[5]

        return (
            thumb_tip[1] < thumb_ip[1] - 25
            and abs(thumb_tip[0] - index_mcp[0]) > 35
        )

    @staticmethod
    def is_thumb_down(mp_landmarks):

        wrist = mp_landmarks[0]
        thumb_tip = mp_landmarks[4]

        other_tips = [
            mp_landmarks[8],
            mp_landmarks[12],
            mp_landmarks[16],
            mp_landmarks[20]
        ]

        return (
            thumb_tip.y > wrist.y
            and all(tip.y < wrist.y for tip in other_tips)
        )

    # ========================
    # === FINGER COUNT =======
    # ========================

    def count_extended_fingers(self, px_landmarks):

        return [
            1 if self.is_thumb_extended(px_landmarks) else 0,

            1 if self.is_finger_extended(
                px_landmarks, 8, 6, 5
            ) else 0,

            1 if self.is_finger_extended(
                px_landmarks, 12, 10, 9
            ) else 0,

            1 if self.is_finger_extended(
                px_landmarks, 16, 14, 13
            ) else 0,

            1 if self.is_finger_extended(
                px_landmarks, 20, 18, 17
            ) else 0,
        ]

    # ========================
    # === GESTURE DETECTION ==
    # ========================

    def detect_gesture(self, mp_landmarks, px_landmarks):

        fingers = self.count_extended_fingers(px_landmarks)

        self.debug = fingers

        thumb_tip = px_landmarks[4]
        index_tip = px_landmarks[8]
        middle_tip = px_landmarks[12]
        pinky_tip = px_landmarks[20]

        thumb_index_dist = self.distance(
            thumb_tip, index_tip
        )

        thumb_pinky_dist = self.distance(
            thumb_tip, pinky_tip
        )

        index_middle_dist = self.distance(
            index_tip, middle_tip
        )

        palm_width = self.distance(
            px_landmarks[5], px_landmarks[17]
        )

        # ===== ORIGINAL SIGNS =====

        if (
            fingers[0] == 1
            and fingers[1:] == [0, 0, 0, 0]
            and self.is_thumb_up(px_landmarks)
        ):
            return "Like"

        if (
            fingers[0] == 1
            and fingers[1:] == [0, 0, 0, 0]
            and self.is_thumb_down(mp_landmarks)
        ):
            return "Dislike"

        if (
            fingers == [0, 1, 1, 0, 0]
            and index_middle_dist > palm_width * 0.4
        ):
            return "Peace"

        if fingers[0] == 1 and (
            fingers[2] == 1
            or fingers[3] == 1
            or fingers[4] == 1
        ):

            if thumb_index_dist < palm_width * 0.7:
                return "Okay"

        if (
            fingers == [1, 0, 0, 0, 1]
            and thumb_pinky_dist > palm_width * 1.45
        ):
            return "Call Me"

        wrist = mp_landmarks[0]

        if (
            fingers == [1, 1, 1, 1, 1]
            and all(
                mp_landmarks[i].y < wrist.y
                for i in [4, 8, 12, 16, 20]
            )
        ):
            return "Hi Fi"

        # ===== NEW COMMUNICATION SIGNS =====

           # ===== NEW SIGNS =====
        # hand is upright when the middle-finger knuckle is above the wrist
        hand_up = mp_landmarks[9].y < mp_landmarks[0].y

        if fingers == [1, 1, 0, 0, 0] and hand_up:
            return "I need help"

        if fingers == [0, 1, 0, 0, 0] and hand_up:
            return "Please wait"

        if fingers == [0, 1, 1, 1, 1] and hand_up:
            return "Please stop"

        if fingers == [0, 1, 0, 0, 1] and hand_up:
            return "I'm in pain"

        if fingers == [1, 1, 1, 0, 0] and hand_up:
            return "Please repeat that"

        if fingers == [0, 0, 1, 1, 1] and hand_up:
            return "I understand"

        if fingers == [0, 0, 0, 1, 1] and hand_up:
            return "I don't understand"

        if fingers == [0, 1, 1, 0, 1] and hand_up:
            return "Thank you"

        # Sorry: thumb out sideways, other fingers closed (thumbs up/down are caught earlier)
        if fingers == [1, 0, 0, 0, 0]:
            return "I'm sorry"

        return None

        return None

    # ========================
    # === SEND TO ESP32 ======
    # ========================

    def send_to_esp32(self, gesture):

        if not self.esp:

            if not self.connect_to_esp32():
                return

        try:

            message = (gesture + "\n").encode("utf-8")

            self.esp.sendall(message)

            print(f"[Wi-Fi] Sent: {gesture}")

        except OSError as exc:

            print(f"[WARN] Wi-Fi send failed: {exc}")

            try:
                self.esp.close()
            except OSError:
                pass

            self.esp = None

    # ========================
    # === MAIN DETECTOR ======
    # ========================

    def run(self):

        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():

            print("[ERROR] Could not open camera.")
            print("[TIP] Try changing CAMERA_INDEX to 1.")
            return

        print("=" * 60)
        print(" Gesture Print Detector Started")
        print(" ESP32 Communication: Wi-Fi TCP")
        print("=" * 60)
        print("Press Q to quit.")

        while True:

            ok, frame = cap.read()

            if not ok:

                print("[WARN] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = self.hands.process(rgb)

            detected = None

            if results.multi_hand_landmarks:

                hand_landmarks = results.multi_hand_landmarks[0]

                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                px_landmarks = self.to_pixel_landmarks(
                    hand_landmarks,
                    frame.shape
                )

                detected = self.detect_gesture(
                    hand_landmarks.landmark,
                    px_landmarks
                )

            else:

                self.debug = []

            # ===== GESTURE STABILITY =====

            if detected:

                self.no_gesture_frames = 0

                self.gesture_buffer.append(detected)

                if len(self.gesture_buffer) > STABILITY_FRAMES:
                    self.gesture_buffer.pop(0)

                if len(self.gesture_buffer) == STABILITY_FRAMES:

                    if self.gesture_buffer.count(detected) == STABILITY_FRAMES:

                        if self.confirmed_gesture != detected:

                            self.confirmed_gesture = detected

                            print(
                                f"[OK] Gesture confirmed: "
                                f"{self.confirmed_gesture}"
                            )

                            self.send_to_esp32(
                                self.confirmed_gesture
                            )

            else:

                self.gesture_buffer.clear()

                self.no_gesture_frames += 1

                if self.no_gesture_frames >= RESET_AFTER_NO_GESTURE:
                    self.confirmed_gesture = ""

            # ===== DISPLAY STATUS =====

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, 170),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.65,
                frame,
                0.35,
                0,
                frame
            )

            label = (
                self.confirmed_gesture
                if self.confirmed_gesture
                else "No Gesture"
            )

            cv2.putText(
                frame,
                f"Sign: {label}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            wifi_status = (
                "ESP32 Wi-Fi: Connected"
                if self.esp
                else "ESP32 Wi-Fi: Disconnected"
            )

            cv2.putText(
                frame,
                wifi_status,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Fingers: {self.debug}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Gesture Print Detector",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

        # ===== CLEANUP =====

        cap.release()
        cv2.destroyAllWindows()

        if self.esp:

            self.esp.close()

            print("[INFO] Wi-Fi connection closed.")

        print("[INFO] Detection stopped.")


# ========================
# === MAIN FUNCTION ======
# ========================

def main():

    detector = GesturePrintDetector()
    detector.run()


if __name__ == "__main__":
    main()
