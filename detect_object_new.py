import cv2
from ultralytics import YOLO

# Загрузка предварительно обученной модели YOLOv8
model = YOLO('neural_network_models/yolov8n.pt')

cap = cv2.VideoCapture(0)

# Чтение и обработка каждого кадра видео
while True:
    success, frame = cap.read()
    if not success:
        break

    # Распознавание объектов на кадре
    results = model(frame)

    # Отображение результатов на кадре
    annotated_frame = results[0].plot()

    # Получение информации о распознанных объектах
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else 'N/A'
            label = model.model.names[cls]  # Имя класса
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Рисование прямоугольника и текста на кадре
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение кадра с распознанными объектами
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
