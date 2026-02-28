# ---------------------------------------
# Loading Libraries
# ---------------------------------------
import easyocr
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import queue

# ---------------------------------------
# Tkinter GUI Setup
# ---------------------------------------
class VideoTextDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Text Detection (EasyOCR)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.cap = None
        self.reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU
        self.frame_count = 0
        self.skip_frames = 10
        self.cached_text = []
        self.running = False
        self.ocr_thread = None
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep latest frame

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.ocr_thread = threading.Thread(target=self.ocr_worker, daemon=True)
            self.ocr_thread.start()
            self.update_frame()

    def stop_detection(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        if self.ocr_thread:
            self.ocr_thread.join(timeout=1)
            self.ocr_thread = None

    def update_frame(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # Put frame in queue for OCR (only keep latest)
        if not self.frame_queue.full():
            self.frame_queue.put(frame.copy())

        # Draw cached OCR results
        display_frame = frame.copy()
        for (bbox, text, confidence) in self.cached_text:
            top_left = tuple(bbox[0])
            bottom_right = tuple(bbox[2])
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            cv2.rectangle(display_frame, (top_left[0], top_left[1] - text_h - 10), (top_left[0] + text_w, top_left[1]), (0, 255, 0), -1)
            cv2.putText(display_frame, text, (top_left[0], top_left[1] - 5), font, font_scale, (0, 0, 0), thickness)

        # Convert frame to RGB and PIL Image
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Schedule next frame update
        self.root.after(10, self.update_frame)

    def ocr_worker(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            results = self.reader.readtext(small_frame)
            new_text = []
            for (bbox, text, confidence) in results:
                new_bbox = [(int(pt[0] * 2), int(pt[1] * 2)) for pt in bbox]
                new_text.append((new_bbox, text, confidence))
            self.cached_text = new_text

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTextDetectionApp(root)
    root.mainloop()
