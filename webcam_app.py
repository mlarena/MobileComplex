import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_source = 0  # Используем первую доступную веб-камеру
        self.vid = cv2.VideoCapture(self.video_source)

        if not self.vid.isOpened():
            messagebox.showwarning("Предупреждение", "Не удалось открыть веб-камеру.")
            self.vid = None

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) if self.vid else 640, height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) if self.vid else 480)
        self.canvas.pack()

        self.btn_toggle = ttk.Button(root, text="Показать видео", command=self.toggle_video)
        self.btn_toggle.pack(anchor=tk.CENTER, expand=True)

        self.is_video_on = False
        self.update()

    def toggle_video(self):
        if self.is_video_on:
            self.btn_toggle.config(text="Показать видео")
            self.is_video_on = False
        else:
            self.btn_toggle.config(text="Прекратить показывать")
            self.is_video_on = True

    def update(self):
        if self.vid is None:
            return

        if self.is_video_on:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.update)

    def on_closing(self):
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
