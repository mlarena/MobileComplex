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

    # Отображение кадра с распознанными объектами
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
