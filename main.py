import math
import os
import time
import cv2
from gtts import gTTS
from moviepy import VideoFileClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pygame

from database_manager import DatabaseManager
from hardware import GretchenHardware
from tarot_agent import generate_tarot_prediction
from vision import GretchenVision

pygame.mixer.init()

PRIMARY_FONT_PATH = "mystic.ttf"

BACKGROUND_MUSIC = {
    "YesNo": "music/Yes_or_no.mp3",
    "Personality": "music/Personality.mp3",
    "Career": "music/Career.mp3",
    "Love": "music/Love.mp3",
}

MUSIC_VOLUMES = {
    "General": 0.2,
    "YesNo": 0.4,
    "Career": 0.4,
}

def prepare_Shhh(video_path="rickroll.mp4", audio_path="rickroll_temp.mp3"):
    if not os.path.exists(audio_path) and os.path.exists(video_path):
        try:
            clip = VideoFileClip(video_path)
            if clip.audio is not None:
                clip.audio.write_audiofile(audio_path, logger=None)
            clip.close()
        except Exception as e:
            print(f"❌ Error extracting audio: {e}")
    return audio_path if os.path.exists(audio_path) else None


def play_background_music(mode):
    """Запускает фоновую музыку для выбранного режима в бесконечном цикле."""
    music_file = BACKGROUND_MUSIC.get(mode)
    volume = MUSIC_VOLUMES.get(mode, 0.2)
    if not music_file:
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(loops=-1, fade_ms=1000)
    except Exception as e:
        print(f"❌ Music Error ({music_file}): {e}")


def stop_background_music():
    try:
        pygame.mixer.music.fadeout(1000)
    except Exception:
        pass


def speak_text(text):
    """Озвучивает текст."""
    print(f"\nGRETCHEN SPEAKS:\n{text}\n")
    try:
        clean_text = text.replace("\n", " ")
        tts = gTTS(text=clean_text, lang="en", tld="co.uk", slow=False)
        tts.save("speech.mp3")
        speech_sound = pygame.mixer.Sound("speech.mp3")
        speech_sound.set_volume(1.0)
        speech_sound.play()
    except Exception as e:
        print(f"❌ TTS Error: {e}")


def stop_speech():
    try:
        pygame.mixer.stop()
    except Exception:
        pass


def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except IOError:
        return ImageFont.load_default()


def draw_five_point_star(draw, center_x, center_y, size, fill, is_filled=True):
    r_outer = size / 2.0
    r_inner = r_outer * 0.382
    points = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        px = center_x + r * math.cos(angle)
        py = center_y + r * math.sin(angle)
        points.append((px, py))

    if is_filled:
        draw.polygon(points, fill=fill)
    else:
        draw.polygon(points, fill=None, outline=fill)


def get_text_custom_width(text, font):
    total_w = 0
    font_size = getattr(font, "size", 20)
    star_size = int(font_size * 0.85)

    for char in text:
        if char in ["✦", "✧"]:
            total_w += star_size + 6
        else:
            try:
                w = font.getlength(char)
            except AttributeError:
                bbox = font.getbbox(char)
                w = (bbox[2] - bbox[0]) if bbox else 8
            total_w += w if w > 0 else 8
    return total_w


def draw_text_custom(draw, position, text, font, fill):
    x, y = position

    for char in text:
        if char in ["✦", "✧"]:
            font_size = getattr(font, "size", 20)
            star_size = int(font_size * 0.85)

            center_x = x + (star_size // 2) + 2
            center_y = y + (font_size // 2)

            draw_five_point_star(
                draw,
                center_x,
                center_y,
                star_size,
                fill=fill,
                is_filled=(char == "✦"),
            )

            x += star_size + 6
            continue

        draw.text((x, y), char, font=font, fill=fill)

        try:
            char_width = font.getlength(char)
        except AttributeError:
            char_bbox = font.getbbox(char)
            char_width = (char_bbox[2] - char_bbox[0]) if char_bbox else 8

        x += char_width if char_width > 0 else 8


def draw_web_header(frame, title_text, status_text, header_height=130):
    _, w, _ = frame.shape

    header_pil = Image.new("RGBA", (w, header_height), (22, 15, 28, 255))
    draw = ImageDraw.Draw(header_pil)

    font_title = get_font(PRIMARY_FONT_PATH, 24)
    font_status = get_font(PRIMARY_FONT_PATH, 16)

    title_lines = [line.strip() for line in title_text.split("//")]
    status_lines = [line.strip() for line in status_text.split("\n")]

    current_y = 15

    for line in title_lines:
        line_w = get_text_custom_width(line, font_title)
        center_x = max(10, (w - line_w) // 2)

        draw_text_custom(
            draw, (center_x, current_y), line, font_title, fill=(180, 70, 220, 255)
        )
        current_y += 30

    current_y += 5

    pink_purple_color = (220, 160, 230, 255)
    for line in status_lines:
        draw_text_custom(
            draw, (25, current_y), line, font_status, fill=pink_purple_color
        )
        current_y += 22

    header_bgr = cv2.cvtColor(np.array(header_pil), cv2.COLOR_RGBA2BGR)
    cv2.line(
        header_bgr,
        (0, header_height - 2),
        (w, header_height - 2),
        (180, 70, 220),
        2,
    )

    return np.vstack((header_bgr, frame))


def create_prediction_window(
        text, title="✦ THE ORACLE REVEALS ✦", width=680, height=520
):
    canvas = Image.new("RGBA", (width, height), (20, 12, 28, 255))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle(
        [12, 12, width - 12, height - 12], outline=(212, 175, 55, 255), width=2
    )
    draw.rectangle(
        [18, 18, width - 18, height - 18],
        outline=(147, 112, 219, 160),
        width=1,
    )

    title_font = get_font(PRIMARY_FONT_PATH, 22)
    footer_font = get_font(PRIMARY_FONT_PATH, 14)

    # Отрисовка заголовка
    title_w = get_text_custom_width(title, title_font)
    title_x = max(10, (width - title_w) // 2)
    draw_text_custom(
        draw, (title_x, 35), title, title_font, fill=(255, 215, 0, 255)
    )

    margin = 40
    max_w = width - (margin * 2)
    top_y = 80
    bottom_y = height - 45
    available_height = bottom_y - top_y

    font_size = 17
    line_spacing = 25
    lines = []

    while font_size >= 11:
        body_font = get_font(PRIMARY_FONT_PATH, font_size)
        line_spacing = int(font_size * 1.45)
        lines = []

        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            current_line = ""
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = draw.textbbox((0, 0), test_line, font=body_font)
                if bbox[2] <= max_w:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

        total_text_height = len(lines) * line_spacing

        # Если текст влезает по высоте — выходим из цикла
        if total_text_height <= available_height:
            break

        # Иначе уменьшаем размер шрифта и пробуем снова
        font_size -= 1

    # Отрисовка подогнанных строк текста
    offset_y = top_y
    body_font = get_font(PRIMARY_FONT_PATH, font_size)

    for line in lines:
        draw_text_custom(
            draw,
            (margin, offset_y),
            line,
            body_font,
            fill=(235, 230, 250, 255),
        )
        offset_y += line_spacing

    # Отрисовка подсказки снизу
    draw_text_custom(
        draw,
        (width // 5, height - 32),
        "Show FIST to close and return",
        footer_font,
        fill=(210, 140, 225, 255),
    )

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGR)


# Константы состояний
STATE_START_MENU = "START_MENU"
STATE_WAITING_FOR_CARD = "WAITING_FOR_CARD"
STATE_SHOWING_RESULT = "SHOWING_RESULT"
STATE_RICKROLL = "RICKROLL"


def main():
    vision = GretchenVision(weights_path="old.pt")
    db = DatabaseManager()
    hw = GretchenHardware()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: no webcam :(")
        return

    cv2.namedWindow("Gretchen Oracle", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gretchen Oracle", 1280, 840)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_state = STATE_START_MENU
    selected_mode_name = None
    prediction_text = ""

    gesture_hold_start = None
    last_detected_gesture = -1

    card_confidence_counter = 0
    REQUIRED_STABLE_FRAMES = 20

    rickroll_cap = None
    rickroll_start_time = 0
    rickroll_frame_delay = 0.033
    rickroll_frame_count = 0
    rickroll_last_frame = None

    # Предварительное извлечение аудио при старте
    prepare_Shhh("rickroll.mp4", "rickroll_temp.mp3")

    print("\n🔮 Gretchen is fully ready for the session!")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = cv2.flip(frame, 1)

        header_title = ""
        header_status = ""

        # СОСТОЯНИЕ 1: ГЛАВНОЕ МЕНЮ
        if current_state == STATE_START_MENU:
            header_title = "✦ GRETCHEN ORACLE ✦ // CHOOSE MODE"
            header_status = (
                "1: Shhh... | 2: Personality | 3: Career | 4: Love | 5: Yes or No"
            )

            fingers = vision.count_fingers(clean_frame)
            if fingers in [1, 2, 3, 4, 5]:
                if fingers == last_detected_gesture:
                    if gesture_hold_start and (
                        time.time() - gesture_hold_start > 1.5
                    ):
                        if fingers == 1:
                            gesture_hold_start = None
                            video_path = "rickroll.mp4"
                            audio_path = prepare_Shhh(video_path)

                            rickroll_cap = cv2.VideoCapture(video_path)

                            if rickroll_cap.isOpened():
                                fps = rickroll_cap.get(cv2.CAP_PROP_FPS)
                                rickroll_frame_delay = 1.0 / fps if fps > 0 else 0.033
                                rickroll_frame_count = 0
                                rickroll_last_frame = None

                                if audio_path:
                                    pygame.mixer.music.stop()
                                    pygame.mixer.music.load(audio_path)
                                    pygame.mixer.music.set_volume(1.0)
                                    pygame.mixer.music.play()

                                rickroll_start_time = time.time()
                                hw.nod()
                                current_state = STATE_RICKROLL
                            else:
                                print(f"❌ Error: '{video_path}' file not found!")
                        else:
                            mode_mapping = {
                                2: "Personality",
                                3: "Career",
                                4: "Love",
                                5: "YesNo",
                            }
                            selected_mode_name = mode_mapping[fingers]
                            gesture_hold_start = None
                            hw.nod()
                            hw.look_at_table()
                            current_state = STATE_WAITING_FOR_CARD
                else:
                    last_detected_gesture = fingers
                    gesture_hold_start = time.time()
            else:
                gesture_hold_start = None

            video_to_show = clean_frame.copy()

        # --- СОСТОЯНИЕ: RICKROLL ---
        elif current_state == STATE_RICKROLL:
            header_title = "✦ SECRET ORACLE DESTINY ✦ // Smile, baby <3"
            header_status = "Show FIST to stop the secret spell"

            is_fist = vision.is_fist_detected(clean_frame)

            if is_fist or not (rickroll_cap and rickroll_cap.isOpened()):
                pygame.mixer.music.stop()
                if rickroll_cap:
                    rickroll_cap.release()
                    rickroll_cap = None

                hw.look_at_face()
                current_state = STATE_START_MENU
                video_to_show = clean_frame.copy()
            else:
                elapsed_time = time.time() - rickroll_start_time
                target_frame_idx = int(elapsed_time / rickroll_frame_delay)

                rick_ret = True
                while rickroll_frame_count <= target_frame_idx and rick_ret:
                    rick_ret, rick_frame = rickroll_cap.read()
                    if rick_ret:
                        rickroll_frame_count += 1
                        rickroll_last_frame = rick_frame

                if not rick_ret and rickroll_last_frame is None:
                    pygame.mixer.music.stop()
                    if rickroll_cap:
                        rickroll_cap.release()
                        rickroll_cap = None

                    hw.look_at_face()
                    current_state = STATE_START_MENU
                    video_to_show = clean_frame.copy()
                else:
                    cam_h, cam_w, _ = clean_frame.shape
                    frame_to_render = (
                        rickroll_last_frame
                        if rickroll_last_frame is not None
                        else clean_frame
                    )
                    video_to_show = cv2.resize(frame_to_render, (cam_w, cam_h))

        # СОСТОЯНИЕ 2: ОЖИДАНИЕ КАРТЫ
        elif current_state == STATE_WAITING_FOR_CARD:
            is_triple_mode = selected_mode_name in ["Career", "Love"]
            expected = "3 CARDS" if is_triple_mode else "1 CARD"

            header_title = (
                f"✦ REALM: {selected_mode_name.upper()} ✦ // SCANNING..."
            )
            header_status = f"Place {expected} on table... [{card_confidence_counter}/{REQUIRED_STABLE_FRAMES}]"

            cards_data, annotated_frame = vision.process_frame(clean_frame)
            all_visible_cards = cards_data.get("all_cards", [])
            main_card = cards_data.get("main_card", None)

            valid_layout = (is_triple_mode and len(all_visible_cards) == 3) or (
                not is_triple_mode and len(all_visible_cards) == 1
            )

            if valid_layout:
                card_confidence_counter += 1

                if card_confidence_counter >= REQUIRED_STABLE_FRAMES:
                    if is_triple_mode:
                        sorted_cards = [
                            card["name"]
                            for card in sorted(
                                all_visible_cards, key=lambda c: c["box"][0]
                            )
                        ]

                        if len(sorted_cards) != 3:
                            card_confidence_counter = 0
                            continue

                        prediction_text = generate_tarot_prediction(
                            mode=selected_mode_name,
                            main_card=sorted_cards[0],
                            additional_cards=[sorted_cards[1], sorted_cards[2]],
                        )
                    else:
                        prediction_text = (
                            f"Card: {main_card}\n"
                            + db.get_prediction(selected_mode_name, main_card)
                        )

                    hw.look_at_face()

                    if selected_mode_name == "YesNo":
                        pred_lower = prediction_text.lower()
                        if "yes" in pred_lower:
                            hw.nod()
                        elif "no" in pred_lower:
                            hw.shake()

                    play_background_music(selected_mode_name)
                    speak_text(prediction_text)

                    current_state = STATE_SHOWING_RESULT
                    card_confidence_counter = 0
            else:
                if card_confidence_counter > 0:
                    card_confidence_counter -= 1

            video_to_show = annotated_frame

        # СОСТОЯНИЕ 3: ПОКАЗ РЕЗУЛЬТАТА
        elif current_state == STATE_SHOWING_RESULT:
            header_title = "✧ PREDICTION GENERATED ✧"
            header_status = "Show FIST to clear and return"

            modal_window = create_prediction_window(prediction_text)
            cv2.imshow("Gretchen Prediction", modal_window)

            is_fist = vision.is_fist_detected(clean_frame)
            if is_fist:
                stop_speech()
                stop_background_music()

                if (
                    cv2.getWindowProperty(
                        "Gretchen Prediction", cv2.WND_PROP_VISIBLE
                    )
                    >= 1
                ):
                    cv2.destroyWindow("Gretchen Prediction")

                hw.look_at_face()
                current_state = STATE_START_MENU
                selected_mode_name = None
                prediction_text = ""

            video_to_show = clean_frame.copy()

        final_ui = draw_web_header(video_to_show, header_title, header_status)

        cv2.imshow("Gretchen Oracle", final_ui)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if rickroll_cap:
        rickroll_cap.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()