import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from datetime import datetime
import glob
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import uuid
import json

class WebcamApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Webcam Viewer")
        
        # Переменные состояния
        self.vid = None
        self.recording = False
        self.showing = False
        self.out = None
        self.model = None
        self.use_network = False
        self.track_history = defaultdict(lambda: [])
        self.saved_track_ids = set()
        
        # Создание папок
        for directory in ["result", "result_images"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
        # Верхняя панель с кнопками
        self.control_frame = ttk.Frame(window)
        self.control_frame.pack(fill="x", pady=5)
        
        self.toggle_button = ttk.Button(self.control_frame, text="Показать видео", command=self.toggle_video)
        self.toggle_button.pack(side="left", padx=5)
        
        self.record_button = ttk.Button(self.control_frame, text="Записывать", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=5)
        
        ttk.Label(self.control_frame, text="Применить нейросеть:").pack(side="left", padx=5)
        self.model_var = tk.StringVar()
        self.models = self.get_model_list()
        self.model_combo = ttk.Combobox(self.control_frame, textvariable=self.model_var, 
                                      values=self.models, state="readonly")
        self.model_combo.pack(side="left", padx=5)
        if self.models:
            self.model_combo.set(self.models[0])
            
        self.apply_button = ttk.Button(self.control_frame, text="Применить", command=self.apply_network)
        self.apply_button.pack(side="left", padx=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Прекратить применение", command=self.stop_network)
        self.stop_button.pack(side="left", padx=5)
        
        # Область видео
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Информационная панель
        self.info_frame = ttk.Frame(window)
        self.info_frame.pack(fill="x", pady=5)
        self.info_label = ttk.Label(self.info_frame, text="Готово к работе", wraplength=600)
        self.info_label.pack()
        
        self.init_camera()
        self.update()
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def init_camera(self):
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            self.info_label.config(text="Ошибка: Не удалось подключиться к веб-камере")
            self.vid = None
        else:
            self.info_label.config(text="Камера успешно инициализирована")
            
    def get_model_list(self):
        if not os.path.exists("neural_network_models"):
            os.makedirs("neural_network_models")
        models = glob.glob("neural_network_models/*.pth") + glob.glob("neural_network_models/*.pt")
        return [os.path.basename(m) for m in models] or ["Нет моделей"]
        
    def toggle_video(self):
        if not self.vid:
            return
        self.showing = not self.showing
        self.toggle_button.config(text="Скрыть видео" if self.showing else "Показать видео")
        self.info_label.config(text=f"Видео {'показано' if self.showing else 'скрыто'}")
        
    def toggle_recording(self):
        if not self.vid or not self.showing:
            self.info_label.config(text="Ошибка: Нет видео для записи")
            return
        self.recording = not self.recording
        self.record_button.config(text="Остановить запись" if self.recording else "Записывать")
        if self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result/video_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            self.info_label.config(text=f"Начата запись: {filename}")
        else:
            if self.out:
                self.out.release()
                self.out = None
            self.info_label.config(text="Запись остановлена")
                
    def apply_network(self):
        if not self.vid or not self.showing:
            self.info_label.config(text="Ошибка: Нет видео для обработки")
            return
        selected_model = self.model_var.get()
        if selected_model == "Нет моделей":
            self.info_label.config(text="Ошибка: Модели не найдены")
            return
        try:
            model_path = os.path.join("neural_network_models", selected_model)
            self.model = YOLO(model_path)
            self.use_network = True
            self.info_label.config(text=f"Модель {selected_model} применяется")
        except Exception as e:
            self.info_label.config(text=f"Ошибка загрузки модели: {str(e)}")
            self.model = None
            self.use_network = False
            
    def stop_network(self):
        if self.use_network:
            self.use_network = False
            self.model = None
            self.track_history.clear()
            self.saved_track_ids.clear()
            self.info_label.config(text="Применение нейросети прекращено")
        else:
            self.info_label.config(text="Нейросеть не применялась")
            
    def update(self):
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret and self.showing:
                if self.use_network and self.model:
                    results = self.model.track(frame, persist=True)
                    
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        class_ids = results[0].boxes.cls.int().cpu().tolist()
                        
                        display_frame = results[0].plot()
                        
                        for box, track_id, cls in zip(boxes, track_ids, class_ids):
                            x, y, w, h = box
                            track = self.track_history[track_id]
                            track.append((float(x), float(y)))
                            if len(track) > 30:
                                track.pop(0)
                                
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(display_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                            
                            if track_id not in self.saved_track_ids:
                                label = self.model.model.names[cls]
                                annotated_frame = frame.copy()
                                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), 
                                            (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f'{label} ID:{track_id}', 
                                          (int(x - w / 2), int(y - h / 2) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
                                unique_id = uuid.uuid4()
                                base_filename = f"{timestamp}_{label}_ID_{track_id}_{unique_id}"
                                image_filename = f"result_images/{base_filename}.jpg"
                                cv2.imwrite(image_filename, annotated_frame)
                                
                                json_filename = f"result_images/{base_filename}.json"
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
                                    
                                self.saved_track_ids.add(track_id)
                                self.info_label.config(text=f"Сохранено: {base_filename}.jpg")
                        
                        frame = display_frame
                    else:
                        frame = results[0].plot()
                
                if self.recording and self.out:
                    self.out.write(frame)
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            elif not self.showing:
                self.canvas.delete("all")
                
        self.window.after(10, self.update)
        
    def on_closing(self):
        if self.vid:
            self.vid.release()
        if self.out:
            self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()