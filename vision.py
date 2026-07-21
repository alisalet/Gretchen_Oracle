import cv2
import os
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision


class GretchenVision:
    def __init__(self, weights_path="old.pt"):
        try:
            if weights_path:
                self.model = YOLO(weights_path)
                print("YOLO Tarot Model loaded successfully.")
            else:
                self.model = None
                print("❌ YOLO path is None. Card detection disabled.")
        except Exception as e:
            print(f"❌ Could not load YOLO weights: {e}. Card detection disabled.")
            self.model = None

        # 2. Инициализация MediaPipe Hands
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"❌ File {model_path} doesn't exist.")
            self.detector = None
            return

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )

        try:
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
            print("MediaPipe Hand Landmarker loaded successfully.")
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            self.detector = None

    def process_frame(self, frame):
        """Ищет карты на кадре и определяет, какая из них главная."""
        cards_data = {"main_card": None, "all_cards": []}
        if self.model is None:
            return cards_data, frame

        results = self.model(frame, verbose=False)[0]
        detected_cards = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            card_name = self.model.names[cls_id]

            box_coords = box.xyxy[0].tolist()
            x_center = float(box.xywh[0][0])

            detected_cards.append((card_name, x_center))

            cards_data["all_cards"].append({
                "name": card_name,
                "box": box_coords
            })

        if len(detected_cards) == 3:
            detected_cards.sort(key=lambda item: item[1])
            cards_data["main_card"] = detected_cards[1][0]
        elif len(detected_cards) == 1:
            cards_data["main_card"] = detected_cards[0][0]

        return cards_data, results.plot()

    def _get_fingers_state(self, frame):
        """считывает состояние всех 5 пальцев на руке."""
        if self.detector is None:
            return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        landmarks = detection_result.hand_landmarks[0]

        # точечки
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        fingers = []

        # большой палец
        dist_tip = np.hypot(landmarks[4].x - landmarks[17].x, landmarks[4].y - landmarks[17].y)
        dist_joint = np.hypot(landmarks[2].x - landmarks[17].x, landmarks[2].y - landmarks[17].y)
        if dist_tip > dist_joint:
            fingers.append(1)
        else:
            fingers.append(0)

        # остальные пальцы
        tip_ids = [8, 12, 16, 20]
        for tip in tip_ids:
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def count_fingers(self, frame):
        """Возвращает общее число поднятых пальцев (от 0 до 5)."""
        state = self._get_fingers_state(frame)
        if state is None:
            return -1
        return sum(state)

    def is_fist_detected(self, frame):
        """Возвращает True, если рука в кадре, но все пальцы сжаты (0 поднятых пальцев)."""
        fingers = self.count_fingers(frame)
        return fingers == 0

    def get_static_prediction(self, mode, card_name):
        """Читает готовый текст предсказания из локальных файлов базы данных."""
        folder = "YesNo" if mode == "Yes/No" else "Personality"
        file_path = f"database/{folder}/{card_name.replace(' ', '_')}.txt"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return f"You drew {card_name}. The document for this card is missing in my archive."