import os
import cv2
import numpy as np
import random
import glob


def apply_extreme_color_augmentation(image):
    """
    Применяет сильные изменения яркости, контраста, насыщенности
    и имитирует изменение цветовой температуры (синее в сумерках / желтее при свечах).
    """
    # 1. Изменение контраста (alpha) и яркости (beta)
    alpha = random.uniform(0.5, 1.5)
    beta = random.randint(-45, 45)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 2. Изменение насыщенности (Saturation) через HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Случайный коэффициент насыщенности от 0.4 (блеклые) до 1.6 (сочные цвета)
    s_scale = random.uniform(0.4, 1.6)
    s = np.clip(s * s_scale, 0, 255).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Имитация изменения баланса белого (цветовой температуры) через LAB
    # Канал 'b' отвечает за баланс желтый/синий
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Отрицательный сдвиг уводит в синеву (сумерки), положительный — в желтизну (свечи)
    color_temp_shift = random.randint(-25, 25)
    b_ch = np.clip(b_ch.astype(np.int32) + color_temp_shift, 0, 255).astype(np.uint8)

    lab = cv2.merge([l_ch, a_ch, b_ch])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def get_warped_card_and_bbox(card_img, bbox_normalized, bg_width, bg_height):
    """
    Применяет случайное перспективное искажение, масштабирование и поворот к карте,
    а также пересчитывает координаты ограничивающей рамки (bounding box).
    """
    img_h, img_w = card_img.shape[:2]

    # Денормализуем координаты исходной рамки карты
    class_id, x_c, y_c, w_box, h_box = bbox_normalized
    x1 = int((x_c - w_box / 2) * img_w)
    y1 = int((y_c - h_box / 2) * img_h)
    x2 = int((x_c + w_box / 2) * img_w)
    y2 = int((y_c + h_box / 2) * img_h)

    # Координаты 4 углов исходной рамки
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.float32)

    # 1. Масштабирование карты (карта занимает от 30% до 55% высоты будущего фона)
    scale = random.uniform(0.3, 0.55)
    target_h = int(bg_height * scale)
    target_w = int(img_w * (target_h / img_h))

    card_resized = cv2.resize(card_img, (target_w, target_h))
    corners = corners * (target_h / img_h)

    # 2. Создаем черный холст размером с фон и помещаем карту по центру
    canvas = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)
    dx = (bg_width - target_w) // 2
    dy = (bg_height - target_h) // 2
    canvas[dy:dy + target_h, dx:dx + target_w] = card_resized
    corners[:, 0] += dx
    corners[:, 1] += dy

    # Маска карты на холсте для последующего вырезания
    card_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
    card_mask[dy:dy + target_h, dx:dx + target_w] = 255

    # 3. Перспективное искажение (имитация съемки под углом)
    max_distortion = int(min(bg_width, bg_height) * 0.12)
    src_pts = np.float32([[0, 0], [bg_width, 0], [bg_width, bg_height], [0, bg_height]])

    dst_pts = np.float32([
        [random.randint(0, max_distortion), random.randint(0, max_distortion)],
        [bg_width - random.randint(0, max_distortion), random.randint(0, max_distortion)],
        [bg_width - random.randint(0, max_distortion), bg_height - random.randint(0, max_distortion)],
        [random.randint(0, max_distortion), bg_height - random.randint(0, max_distortion)]
    ])

    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Случайный поворот карты вокруг центра (на все 360 градусов)
    angle = random.uniform(-180, 180)
    M_rot = cv2.getRotationMatrix2D((bg_width / 2, bg_height / 2), angle, 1.0)

    # Объединяем трансформации в одну общую матрицу 3x3
    M_rot_3x3 = np.eye(3)
    M_rot_3x3[:2, :] = M_rot
    M_total = np.dot(M_rot_3x3, M_persp)

    # Применяем трансформацию к изображению карты и к маске
    warped_card = cv2.warpPerspective(canvas, M_total, (bg_width, bg_height), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(card_mask, M_total, (bg_width, bg_height), flags=cv2.INTER_NEAREST)

    # Трансформируем точки углов рамки разметки
    ones = np.ones((4, 1))
    corners_homg = np.hstack((corners, ones))
    transformed_corners = np.dot(M_total, corners_homg.T).T
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:]

    # Находим новые границы рамки (min/max по осям X и Y)
    x_min = np.min(transformed_corners[:, 0])
    x_max = np.max(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    y_max = np.max(transformed_corners[:, 1])

    # Обрезаем рамку, если она вышла за пределы экрана
    x_min = max(0, min(x_min, bg_width - 1))
    x_max = max(0, min(x_max, bg_width - 1))
    y_min = max(0, min(y_min, bg_height - 1))
    y_max = max(0, min(y_max, bg_height - 1))

    # Рассчитываем новые нормализованные координаты для YOLO
    new_w = (x_max - x_min) / bg_width
    new_h = (y_max - y_min) / bg_height
    new_x_c = (x_min + x_max) / (2 * bg_width)
    new_y_c = (y_min + y_max) / (2 * bg_height)

    if new_w < 0.05 or new_h < 0.05:
        return None, None, None

    new_bbox = (class_id, new_x_c, new_y_c, new_w, new_h)

    return warped_card, warped_mask, new_bbox


def augment_dataset(cards_dir, bg_dir, output_dir, imgs_per_card=50, target_size=(640, 640)):
    """
    Основная функция для генерации аугментированного YOLO датасета.
    """
    img_formats = ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG']
    card_paths = []
    for fmt in img_formats:
        card_paths.extend(glob.glob(os.path.join(cards_dir, fmt)))

    bg_paths = []
    for fmt in img_formats:
        bg_paths.extend(glob.glob(os.path.join(bg_dir, fmt)))

    if not card_paths:
        print(f"Ошибка: В папке '{cards_dir}' не найдены изображения карт!")
        return
    if not bg_paths:
        print(f"Ошибка: В папке '{bg_dir}' не найдены изображения фонов!")
        return

    # Создаем структуру папок по стандарту YOLO
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_lbl_dir = os.path.join(output_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_lbl_dir = os.path.join(output_dir, 'val', 'labels')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    bg_width, bg_height = target_size

    print(f"Найдено карт для обработки: {len(card_paths)}")
    print(f"Найдено текстур фонов: {len(bg_paths)}")
    print(f"Генерируем по {imgs_per_card} вариаций на каждую карту...")

    total_generated = 0

    for card_path in card_paths:
        base_name = os.path.splitext(os.path.basename(card_path))[0]
        txt_path = os.path.join(cards_dir, base_name + '.txt')

        if not os.path.exists(txt_path):
            print(f"Пропущено: Для '{base_name}' не найден .txt файл разметки.")
            continue

        card_img = cv2.imread(card_path)
        if card_img is None:
            continue

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue

        # Считываем первую строку разметки (индекс класса и координаты)
        parts = lines[0].strip().split()
        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:]]
        bbox_normalized = [class_id] + bbox

        for i in range(imgs_per_card):
            bg_path = random.choice(bg_paths)
            bg_img = cv2.imread(bg_path)
            if bg_img is None:
                continue

            # Подгоняем фон под размер 640x640
            bg_resized = cv2.resize(bg_img, target_size)

            # Искажаем карту и пересчитываем её bounding box
            warped_card, mask, new_bbox = get_warped_card_and_bbox(
                card_img, bbox_normalized, bg_width, bg_height
            )

            if warped_card is None:
                continue

            # Соединяем карту и фон с помощью бинарной маски
            mask_3ch = cv2.merge([mask, mask, mask])
            composite = np.where(mask_3ch == 255, warped_card, bg_resized)

            # ПРИМЕНЯЕМ ЦВЕТОВУЮ АУГМЕНТАЦИЮ (Сумрак, свечи, тени, контраст)
            composite = apply_extreme_color_augmentation(composite)

            # Эффект легкого размытия (имитация фокуса камеры смартфона)
            blur_k = random.choice([1, 3, 5])
            if blur_k > 1:
                composite = cv2.GaussianBlur(composite, (blur_k, blur_k), 0)

            # Распределяем изображения: ~15% уходит в валидационную выборку
            is_val = random.random() < 0.15
            out_img_dir = val_img_dir if is_val else train_img_dir
            out_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            # Сохраняем картинку сцены и файл новой разметки YOLO
            save_name = f"{base_name}_aug_{i}"
            cv2.imwrite(os.path.join(out_img_dir, f"{save_name}.jpg"), composite)

            with open(os.path.join(out_lbl_dir, f"{save_name}.txt"), 'w') as f_out:
                f_out.write(f"{new_bbox[0]} {new_bbox[1]:.6f} {new_bbox[2]:.6f} {new_bbox[3]:.6f} {new_bbox[4]:.6f}\n")

            total_generated += 1

    print(f"\nГенерация успешно завершена! Создано изображений: {total_generated}")


if __name__ == '__main__':
    # Скрипт готов к импорту или самостоятельному запуску
    pass