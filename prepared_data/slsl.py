import os
import cv2
import numpy as np
from ultralytics import YOLO


def resize_with_padding(img, target_size=640):
    """
    Уменьшает изображение до target_size (640x640), сохраняя пропорции,
    и добавляет черные поля (padding) по бокам или сверху/снизу.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Меняем размер
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Создаем черный квадратный холст 640x640
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Вставляем уменьшенное фото по центру холста
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized

    return canvas


def process_and_label(model_path, input_dir, output_dir, target_size=640, conf=0.55):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]

    print(f"Найдено {len(images)} больших фото. Начинаем сжатие и разметку...")
    count = 0

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue

        # 1. Сжимаем картинку до 640x640 с сохранением пропорций
        compressed_img = resize_with_padding(orig_img, target_size)

        # Сохраняем сжатую картинку в новую папку
        base_name = os.path.splitext(img_name)[0]
        save_img_path = os.path.join(output_dir, f"{base_name}.jpg")
        cv2.imwrite(save_img_path, compressed_img)

        # 2. Запускаем YOLO на уже СЖАТОМ изображении (так разметка будет идеальной)
        results = model.predict(source=compressed_img, conf=conf, verbose=False)

        yolo_lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                xywhn = box.xywhn[0].tolist()
                yolo_lines.append(f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")

        # Если модель нашла карты, сохраняем .txt файл разметки рядом со сжатой картинкой
        if yolo_lines:
            txt_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                f.writelines(yolo_lines)
            count += 1

    print(f"\nГотово! {count} фото успешно сжаты до 640x640 и размечены в папке '{output_dir}'.")


if __name__ == "__main__":
    process_and_label(
        model_path="../old.pt",
        input_dir="JSJSNJ",  # Откуда брать большие фото
        output_dir="JSJSNJ_2"  # Куда складывать готовые маленькие фото и .txt
    )