import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from datetime import datetime
import glob
from ultralytics import YOLO

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
        
        # Создание папки для записи
        if not os.path.exists("result"):
            os.makedirs("result")
            
        # Верхняя панель с кнопками
        self.control_frame = ttk.Frame(window)
        self.control_frame.pack(fill="x", pady=5)
        
        # Кнопка показать/скрыть
        self.toggle_button = ttk.Button(self.control_frame, text="Показать видео", command=self.toggle_video)
        self.toggle_button.pack(side="left", padx=5)
        
        # Кнопка записи
        self.record_button = ttk.Button(self.control_frame, text="Записывать", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=5)
        
        # Выпадающий список моделей
        ttk.Label(self.control_frame, text="Применить нейросеть:").pack(side="left", padx=5)
        self.model_var = tk.StringVar()
        self.models = self.get_model_list()
        self.model_combo = ttk.Combobox(self.control_frame, textvariable=self.model_var, 
                                      values=self.models, state="readonly")
        self.model_combo.pack(side="left", padx=5)
        if self.models:
            self.model_combo.set(self.models[0])
            
        # Кнопка применения нейросети
        self.apply_button = ttk.Button(self.control_frame, text="Применить", command=self.apply_network)
        self.apply_button.pack(side="left", padx=5)
        
        # Область видео
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()
        
        # Попытка инициализации камеры
        self.init_camera()
        self.update()
        
        # Обработка закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def init_camera(self):
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            messagebox.showerror("Ошибка", "Не удалось подключиться к веб-камере")
            self.vid = None
            
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
        
    def toggle_recording(self):
        if not self.vid or not self.showing:
            return
            
        self.recording = not self.recording
        self.record_button.config(text="Остановить запись" if self.recording else "Записывать")
        
        if self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result/video_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        else:
            if self.out:
                self.out.release()
                self.out = None
                
    def apply_network(self):
        if not self.vid or not self.showing:
            return
        selected_model = self.model_var.get()
        if selected_model == "Нет моделей":
            messagebox.showinfo("Информация", "Модели не найдены в папке neural_network_models")
            return
            
        try:
            model_path = os.path.join("neural_network_models", selected_model)
            self.model = YOLO(model_path)
            self.use_network = True
            messagebox.showinfo("Успех", f"Модель {selected_model} успешно загружена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
            self.model = None
            self.use_network = False
            
    def update(self):
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret and self.showing:
                # Применение нейросети, если выбрана
                if self.use_network and self.model:
                    results = self.model(frame)
                    frame = results[0].plot()
                
                if self.recording and self.out:
                    self.out.write(frame)
                    
                # Конвертация для Tkinter
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