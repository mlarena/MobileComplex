import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_source = 0  # Используем первую доступную веб-камеру
        try:
            self.vid = cv2.VideoCapture(self.video_source)
            if not self.vid.isOpened():
                raise ValueError("Не удалось открыть веб-камеру.")
        except Exception as e:
            messagebox.showwarning("Предупреждение", f"Ошибка: {e}")
            self.vid = None

        self.is_video_on = False
        self.is_recording = False
        self.out = None

        # Создание области кнопок вверху экрана
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_toggle = ttk.Button(self.control_frame, text="Показать видео", command=self.toggle_video)
        self.btn_toggle.pack(side=tk.LEFT, padx=5)

        self.record_var = tk.BooleanVar()
        self.record_check = ttk.Checkbutton(self.control_frame, text="Записывать видео", variable=self.record_var, command=self.toggle_recording)
        self.record_check.pack(side=tk.LEFT, padx=5)

        self.model_label = ttk.Label(self.control_frame, text="Применить нейросеть:")
        self.model_label.pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar()
        self.model_menu = ttk.Combobox(self.control_frame, textvariable=self.model_var)
        self.model_menu.pack(side=tk.LEFT, padx=5)

        self.load_models()

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) if self.vid else 640, height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) if self.vid else 480)
        self.canvas.pack()

        self.update()

    def load_models(self):
        models_dir = "neural_network_models"
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            models = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
            self.model_menu['values'] = models
            if models:
                self.model_var.set(models[0])

    def toggle_video(self):
        if self.is_video_on:
            self.btn_toggle.config(text="Показать видео")
            self.is_video_on = False
        else:
            self.btn_toggle.config(text="Прекратить показывать")
            self.is_video_on = True

    def toggle_recording(self):
        if self.record_var.get():
            if not self.out:
                now = datetime.now()
                filename = now.strftime("result/%Y_%m_%d_%H_%M_%S.avi")
                os.makedirs("result", exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(filename, fourcc, 20.0, (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.is_recording = True
        else:
            if self.out:
                self.out.release()
                self.out = None
            self.is_recording = False

    def update(self):
        if self.vid is None:
            return

        if self.is_video_on:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                if self.is_recording and self.out:
                    self.out.write(frame)

        self.root.after(10, self.update)

    def on_closing(self):
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
