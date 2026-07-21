import os
import cv2
import matplotlib.pyplot as plt

# Настраиваем пути
DATASET_DIR = "compressed_dataset"
CLASSES_FILE = "classes.txt"

# 1. Читаем классы карт
if not os.path.exists(CLASSES_FILE):
    CLASSES_FILE = os.path.join(DATASET_DIR, "notes.txt")
    if not os.path.exists(CLASSES_FILE):
        CLASSES_FILE = os.path.join(DATASET_DIR, "classes.txt")

with open(CLASSES_FILE, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# 2. Ищем сжатые картинки
valid_exts = ('.jpg', '.jpeg', '.png')
images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(valid_exts)]
images.sort()

if not images:
    print(f"В папке '{DATASET_DIR}' пусто! Сначала запусти скрипт сжатия.")
    exit()

# Посмотрим первые 3 картинки для проверки
for idx in range(min(3, len(images))):
    img_name = images[idx]
    img_path = os.path.join(DATASET_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    txt_path = os.path.join(DATASET_DIR, f"{base_name}.txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Если YOLO создал разметку — рисуем её
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, nw, nh = map(float, parts[1:])

                    # Переводим координаты YOLO обратно в пиксели
                    xmin = int((cx - nw / 2) * w)
                    ymin = int((cy - nh / 2) * h)
                    xmax = int((cx + nw / 2) * w)
                    ymax = int((cy + nh / 2) * h)

                    name = classes[cls_id] if cls_id < len(classes) else f"ID {cls_id}"
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                    cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Отрисовка через Matplotlib (вкладка SciView или отдельное окно)
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(f"Проверка: {img_name}")
    plt.axis('off')
    plt.show()