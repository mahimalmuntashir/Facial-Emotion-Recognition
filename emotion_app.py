import os
import json
import time
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import tensorflow as tf

APP_TITLE = "Facial Emotion Detection - CNN Project For AI"

MODEL_PATHS_TRY = ["fer2013_vgg_like.keras", "best_vgg_like.keras"]
LABELS_PATH = "class_names.json"

CAM_INDEX = 0
PREDICT_EVERY_N_FRAMES = 5
SIDEBAR_WIDTH = 420

# Optional watermark (transparent background because we draw on frame)
WATERMARK_TEXT = ""  # e.g. "Your Name"
WATERMARK_MARGIN = 15
WATERMARK_SCALE = 0.8
WATERMARK_THICKNESS = 2


def load_class_names(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Put class_names.json beside emotion_app.py")
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError("class_names.json must be a JSON list of strings.")
    return names


def load_model_any(paths: list[str]) -> tf.keras.Model:
    for p in paths:
        if os.path.exists(p):
            print(f"[INFO] Loading model: {p}")
            return tf.keras.models.load_model(p, compile=False)
    raise FileNotFoundError("Model not found. Put fer2013_vgg_like.keras or best_vgg_like.keras in this folder.")


def get_model_input_hw_c(model: tf.keras.Model):
    """Return expected (H,W,C) from model.input_shape."""
    shp = model.input_shape
    if isinstance(shp, list):
        shp = shp[0]

    # (None,H,W,C) or (None,H,W)
    if len(shp) == 4:
        _, h, w, c = shp
        if c is None:
            c = 1
    elif len(shp) == 3:
        _, h, w = shp
        c = 1
    else:
        raise ValueError(f"Unsupported model input shape: {shp}")

    if h is None or w is None:
        raise ValueError(f"Model input shape has None H/W: {shp}")

    return int(h), int(w), int(c)


def add_watermark_bgr(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return frame_bgr
    out = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = WATERMARK_MARGIN
    y = out.shape[0] - WATERMARK_MARGIN

    # outline
    cv2.putText(out, text, (x, y), font, WATERMARK_SCALE, (0, 0, 0),
                thickness=WATERMARK_THICKNESS + 2, lineType=cv2.LINE_AA)
    # main text
    cv2.putText(out, text, (x, y), font, WATERMARK_SCALE, (255, 255, 255),
                thickness=WATERMARK_THICKNESS, lineType=cv2.LINE_AA)
    return out


def detect_face_and_crop_bgr(frame_bgr: np.ndarray, face_cascade):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_bgr.shape[1], x + w + pad)
        y2 = min(frame_bgr.shape[0], y + h + pad)
        return frame_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

    # fallback center square
    h0, w0 = frame_bgr.shape[:2]
    s = min(h0, w0)
    x1 = (w0 - s) // 2
    y1 = (h0 - s) // 2
    return frame_bgr[y1:y1+s, x1:x1+s], None


def preprocess_bgr_to_model_input(face_bgr: np.ndarray, target_hw: tuple[int, int], channels: int):
    """
    For your VGG-like notebook model:
      - grayscale
      - resize to 40x40
      - normalize to [-1,1] using x/127.5 - 1
      - output shape: (1,H,W,1)
    """
    H, W = target_hw

    if channels == 1:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        x = gray.astype(np.float32)
        x = x / 127.5 - 1.0                     # ✅ [-1,1]
        x = x.reshape(1, H, W, 1)               # ✅ ALWAYS (1,H,W,1)
        return x

    # if ever using RGB model later:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32)
    x = x / 127.5 - 1.0
    x = x.reshape(1, H, W, 3)
    return x


def softmax_topk(prob: np.ndarray, class_names: list[str], k=3):
    idx = np.argsort(prob)[::-1][:k]
    return [(class_names[i], float(prob[i])) for i in idx]


class EmotionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1300x720")
        self.root.minsize(1100, 650)

        # grid: preview expands, sidebar fixed
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        self.class_names = load_class_names(LABELS_PATH)
        self.model = load_model_any(MODEL_PATHS_TRY)

        self.in_h, self.in_w, self.in_c = get_model_input_hw_c(self.model)
        print(f"[INFO] Model expects: (None, {self.in_h}, {self.in_w}, {self.in_c})")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.cap = None
        self.is_camera_on = False
        self.frame_count = 0
        self.last_frame_bgr = None

        self._build_ui()

    def _build_ui(self):
        # left
        self.left = tk.Frame(self.root, padx=8, pady=8)
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_rowconfigure(0, weight=1)
        self.left.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(self.left, bd=2, relief="groove", bg="white", text="Camera / Image Preview")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # right
        self.right = tk.Frame(self.root, padx=8, pady=8, width=SIDEBAR_WIDTH)
        self.right.grid(row=0, column=1, sticky="ns")
        self.right.grid_propagate(False)

        btn_frame = tk.LabelFrame(self.right, text="Controls", padx=10, pady=10)
        btn_frame.pack(fill=tk.X)

        self.btn_start = tk.Button(btn_frame, text="Start Camera", command=self.start_camera, width=18)
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop = tk.Button(btn_frame, text="Stop Camera", command=self.stop_camera, width=18)
        self.btn_stop.grid(row=0, column=1, padx=5, pady=5)

        self.btn_capture = tk.Button(btn_frame, text="Capture + Predict", command=self.capture_and_predict, width=18)
        self.btn_capture.grid(row=1, column=0, padx=5, pady=5)

        self.btn_upload = tk.Button(btn_frame, text="Upload Image + Predict", command=self.upload_and_predict, width=18)
        self.btn_upload.grid(row=1, column=1, padx=5, pady=5)

        self.auto_var = tk.IntVar(value=1)
        self.chk_auto = tk.Checkbutton(btn_frame, text="Auto Predict (Live)", variable=self.auto_var)
        self.chk_auto.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        res_frame = tk.LabelFrame(self.right, text="Prediction", padx=10, pady=10)
        res_frame.pack(fill=tk.X, pady=10)

        self.pred_title = tk.Label(res_frame, text="Emotion: -", font=("Arial", 18, "bold"))
        self.pred_title.pack(anchor="w")

        self.pred_conf = tk.Label(res_frame, text="Confidence: -", font=("Arial", 12))
        self.pred_conf.pack(anchor="w", pady=(5, 0))

        self.topk_label = tk.Label(res_frame, text="Top-3: -", font=("Arial", 12),
                                   wraplength=SIDEBAR_WIDTH-40, justify="left")
        self.topk_label.pack(anchor="w", pady=(5, 0))

        bars_frame = tk.LabelFrame(self.right, text="Probabilities", padx=10, pady=10)
        bars_frame.pack(fill=tk.BOTH, expand=True)

        self.bars_canvas = tk.Canvas(bars_frame, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.bars_canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(self.right, text="Ready", anchor="w")
        self.status.pack(fill=tk.X, pady=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_camera(self):
        if self.is_camera_on:
            return
        self.cap = cv2.VideoCapture(CAM_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera. Try CAM_INDEX = 1 or 2.")
            return
        self.is_camera_on = True
        self.status.config(text="Camera started")
        self.frame_count = 0
        self._update_camera_frame()

    def stop_camera(self):
        self.is_camera_on = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status.config(text="Camera stopped")

    def _update_camera_frame(self):
        if not self.is_camera_on or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status.config(text="Camera read failed")
            self.root.after(100, self._update_camera_frame)
            return

        self.last_frame_bgr = frame
        self.frame_count += 1

        # draw face box
        _, bbox = detect_face_and_crop_bgr(frame, self.face_cascade)
        disp = frame.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)

        disp = add_watermark_bgr(disp, WATERMARK_TEXT)
        self._show_bgr_on_preview(disp)

        if self.auto_var.get() == 1 and (self.frame_count % PREDICT_EVERY_N_FRAMES == 0):
            self._predict_from_bgr(frame, live=True)

        self.root.after(10, self._update_camera_frame)

    def _show_bgr_on_preview(self, frame_bgr: np.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        self.root.update_idletasks()
        w = max(200, self.preview_label.winfo_width())
        h = max(200, self.preview_label.winfo_height())

        # fill the preview area
        img = ImageOps.fit(img, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))

        imgtk = ImageTk.PhotoImage(img)
        self.preview_label.imgtk = imgtk
        self.preview_label.config(image=imgtk, text="")

    def capture_and_predict(self):
        if self.last_frame_bgr is None:
            messagebox.showwarning("No Frame", "No camera frame available. Start the camera first.")
            return
        self._predict_from_bgr(self.last_frame_bgr, live=False)

    def upload_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path)
        if bgr is None:
            messagebox.showerror("File Error", "Could not read this image file.")
            return

        bgr = add_watermark_bgr(bgr, WATERMARK_TEXT)
        self.last_frame_bgr = bgr
        self._show_bgr_on_preview(bgr)
        self._predict_from_bgr(bgr, live=False)

    def _predict_from_bgr(self, frame_bgr: np.ndarray, live: bool):
        # ✅ prevent the GUI from freezing on exceptions
        try:
            t0 = time.time()

            face_bgr, _ = detect_face_and_crop_bgr(frame_bgr, self.face_cascade)
            x = preprocess_bgr_to_model_input(face_bgr, (self.in_h, self.in_w), self.in_c)

            prob = self.model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(prob))
            pred_label = self.class_names[pred_idx]
            pred_conf = float(prob[pred_idx])

            top3 = softmax_topk(prob, self.class_names, k=3)

            self.pred_title.config(text=f"Emotion: {pred_label}")
            self.pred_conf.config(text=f"Confidence: {pred_conf:.3f}")
            self.topk_label.config(text="Top-3: " + " | ".join([f"{n}({p:.2f})" for n, p in top3]))

            self._draw_prob_bars(prob)

            dt_ms = int((time.time() - t0) * 1000)
            self.status.config(text=f"{'Live ' if live else ''}predicted in {dt_ms} ms")

        except Exception as e:
            self.status.config(text=f"Prediction error: {type(e).__name__}")
            print("[ERROR]", repr(e))

    def _draw_prob_bars(self, prob: np.ndarray):
        c = self.bars_canvas
        c.delete("all")

        self.root.update_idletasks()
        W = max(360, int(c.winfo_width()))
        H = max(280, int(c.winfo_height()))

        left_pad = 120
        right_pad = 30
        percent_gap = 70
        top_pad = 12

        percent_x = W - right_pad
        max_bar_w = (percent_x - percent_gap) - left_pad
        bar_h = max(20, (H - 2 * top_pad) // max(1, len(self.class_names)))

        for i, name in enumerate(self.class_names):
            p = float(prob[i])
            y1 = top_pad + i * bar_h
            y2 = y1 + bar_h - 6

            c.create_text(10, (y1 + y2) // 2, text=name, anchor="w", font=("Arial", 10))
            bw = int(max_bar_w * p)
            c.create_rectangle(left_pad, y1, left_pad + bw, y2, fill="#4a90e2", outline="")
            c.create_text(percent_x, (y1 + y2) // 2, text=f"{p*100:4.1f}%",
                          anchor="e", font=("Arial", 10))

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    EmotionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
