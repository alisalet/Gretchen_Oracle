import os
from ultralytics import YOLO


def auto_label_images(model_path, images_dir, output_labels_dir, conf_threshold=0.6):
    """
    Прогоняет фотографии через YOLO и сохраняет предсказания в .txt файлы YOLO-формата.
    """
    # Загружаем твою обученную модель
    model = YOLO(model_path)

    # Убеждаемся, что папка для сохранения существует
    os.makedirs(output_labels_dir, exist_ok=True)

    # Получаем список всех картинок (jpg, png, jpeg)
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]

    print(f"Найдено {len(image_files)} изображений для авто-разметки...")

    success_count = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)

        # Запускаем предсказание
        # conf_threshold отсекает неуверенные предсказания
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)

        # Имя текстового файла должно совпадать с именем картинки
        base_name = os.path.splitext(img_name)[0]
        txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")

        yolo_lines = []

        # Обрабатываем результаты (даже если на фото несколько карт, YOLO найдет их все)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Извлекаем класс и координаты
                cls_id = int(box.cls[0])

                # Координаты в нормализованном виде (center_x, center_y, width, height)
                xywhn = box.xywhn[0].tolist()

                # Формируем строку для YOLO: класс x y w h
                line = f"{cls_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                yolo_lines.append(line)

        # Если модель что-то нашла, сохраняем .txt файл
        if yolo_lines:
            with open(txt_path, 'w') as f:
                f.writelines(yolo_lines)
            success_count += 1

    print(f"\nГотово! Успешно размечено {success_count} из {len(image_files)} фотографий.")
    print(f"Файлы разметки сохранены в папку: {output_labels_dir}")


if __name__ == "__main__":
    # Запуск
    auto_label_images(
        model_path="../old.pt",  # Твоя обученная модель
        images_dir="new_photos",  # Папка с новыми фотографиями
        output_labels_dir="new_photos",  # Сохраняем .txt прямо к картинкам (чтобы сразу открыть в AnyLabeling)
        conf_threshold=0.55  # Порог уверенности (можно подкрутить)
    )