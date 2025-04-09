from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import uuid
import json

# Загрузка предварительно обученной модели YOLOv8
model = YOLO('neural_network_models/yolov8n.pt')

# Открытие видео файла
cap = cv2.VideoCapture(0)

# Создание словаря для хранения истории треков объектов
track_history = defaultdict(lambda: [])
saved_track_ids = set()  # Множество для хранения уже сохраненных track IDs

# Создание директории для сохранения изображений, если она не существует
output_dir = 'result_images'
os.makedirs(output_dir, exist_ok=True)

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
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # идентификаторы классов

        # Визуализация результатов на кадре для отображения
        display_frame = results[0].plot()

        # Отрисовка треков и сохранение изображения только для уникальных track_id
        for box, track_id, cls in zip(boxes, track_ids, class_ids):
            x, y, w, h = box  # координаты центра и размеры бокса
            track = track_history[track_id]
            track.append((float(x), float(y)))  # добавление координат центра объекта в историю
            if len(track) > 30:  # ограничение длины истории до 30 кадров
                track.pop(0)

            # Рисование линий трека
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Рисование прямоугольника и метки только для уникального track_id
            if track_id not in saved_track_ids:
                label = model.model.names[cls]  # Имя класса
                annotated_frame = frame.copy()
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label} ID:{track_id}', (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Сохранение изображения с уникальным track_id и GUID
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                unique_id = uuid.uuid4()  # Генерация GUID
                base_filename = f"{timestamp}_{label}_ID_{track_id}_{unique_id}"
                image_filename = f"{output_dir}/{base_filename}.jpg"
                cv2.imwrite(image_filename, annotated_frame)

                # Создание JSON файла с метаданными
                json_filename = f"{output_dir}/{base_filename}.json"
                json_data = {
                    "DateTimeDetection": timestamp,
                    "ClassName": label,
                    "Latitude": "N/A",
                    "Longitude": "N/A",
                    "SectionOfRoad": "Участок дороги",
                    "CriticalLevel": 1,
                    "RoadClass": "Автомагистраль",
                    "RoadCategory": "IВ Общее число полос движения 4 и более",
                    "Contractor": "Подрядная организация",
                    "ImageName": os.path.basename(image_filename)
                }
                with open(json_filename, 'w', encoding='utf-8') as json_file:
                    json.dump(json_data, json_file, indent=4, ensure_ascii=False)

                saved_track_ids.add(track_id)  # Отмечаем, что изображение для этого track_id уже сохранено
                print(f"Saved image: {image_filename}")  # Debug statement
                print(f"Saved JSON: {json_filename}")  # Debug statement

        # Отображение аннотированного кадра
        cv2.imshow("YOLOv8 Tracking", display_frame)
    else:
        # Если объекты не обнаружены, просто отображаем кадр
        cv2.imshow("YOLOv8 Tracking", frame)

    # Прерывание цикла при нажатии клавиши 'Esc'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение видеозахвата и закрытие всех окон OpenCV
cap.release()
cv2.destroyAllWindows()
