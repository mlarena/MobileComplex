import cv2

def list_available_cameras():
    max_cameras_to_check = 10  # Максимальное количество камер для проверки
    available_cameras = []

    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            print(f"Камера {i} доступна.")
        cap.release()

    if not available_cameras:
        print("Нет доступных камер.")
    else:
        print(f"Доступные камеры: {available_cameras}")

list_available_cameras()
