from ultralytics import YOLO
import cv2

# Загружаем модель
model = YOLO("../old.pt")

# Открываем веб-камеру (0 - индекс стандартной камеры)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем YOLO на текущем кадре
    # stream=True позволяет обрабатывать видеопоток быстрее и без утечек памяти
    results = model(frame, stream=True, conf=0.5)

    # Визуализируем результаты прямо на кадре
    for r in results:
        annotated_frame = r.plot()  # Рисует рамки и подписи классов

    # Показываем окно с видеопотоком
    cv2.imshow("YOLO Tarot Detector", annotated_frame)

    # Нажми клавишу 'q' на клавиатуре, чтобы выйти
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()