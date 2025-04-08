from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Загрузка предварительно обученной модели YOLOv8
model = YOLO('neural_network_models/yolov8n.pt')

# Открытие видео файла
cap = cv2.VideoCapture(0)

# Создание словаря для хранения истории треков объектов
track_history = defaultdict(lambda: [])

# Цикл для обработки каждого кадра видео
while cap.isOpened():
    # Считывание кадра из видео
    success, frame = cap.read()
    if not success:
        break

    # Применение YOLOv8 для отслеживания объектов на кадре, с сохранением треков между кадрами
    results = model.track(frame, persist=True)

    # Проверка на наличие объектов
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Получение координат боксов и идентификаторов треков
        boxes = results[0].boxes.xywh.cpu()  # xywh координаты боксов
        track_ids = results[0].boxes.id.int().cpu().tolist()  # идентификаторы треков

        # Визуализация результатов на кадре
        annotated_frame = results[0].plot()

        # Отрисовка треков
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box  # координаты центра и размеры бокса
            track = track_history[track_id]
            track.append((float(x), float(y)))  # добавление координат центра объекта в историю
            if len(track) > 30:  # ограничение длины истории до 30 кадров
                track.pop(0)

            # Рисование линий трека
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Отображение аннотированного кадра
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
    else:
    # Если объекты не обнаружены, просто отображаем кадр
        cv2.imshow("YOLOv8 Tracking", frame)
     
    # Прерывание цикла при нажатии клавиши 'Esc'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение видеозахвата и закрытие всех окон OpenCV
cap.release()
cv2.destroyAllWindows()

