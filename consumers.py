import os
from collections import deque
from threading import Lock
from time import sleep, time

import cv2
import numpy as np
from model_onnx import YoloONNX
from PyQt6.QtCore import QThread, pyqtSignal


class VideoThread(QThread):
    # change_pixmap_signal = pyqtSignal(QImage)
    position_signal = pyqtSignal(int)

    def __init__(self, stream, source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_flag = True
        self.source = source
        self.deque = stream
        self.stream_lock = Lock()
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.frame = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = 1.0 / self.fps

        # preallocate arrays
        self.frame = np.zeros(
            shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3),
            dtype=np.uint8,
        )
        self.rgb_frame = np.zeros(
            shape=(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, 3),
            dtype=np.uint8,
        )

        cnt = 0
        while self._run_flag:
            with self.stream_lock:
                ret, frame = self.cap.read(self.frame)
                if ret:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, self.rgb_frame)
                    self.deque.append(image_rgb)

            if cnt % (self.fps // 2) == 0:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.position_signal.emit(int(current_frame / self.total_frames * 1000))
            sleep(delay)

        self.cap.release()

    def stop(self):
        self._run_flag = False

    def restart(self):
        self._run_flag = True

    def set_position(self, new_pos):
        if self.cap is not None:
            with self.stream_lock:
                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES, (new_pos / 1000 * self.total_frames)
                )

    def duration(self):
        # in seconds
        return self.total_frames // self.fps


class OnnxRunner(QThread):

    def __init__(
        self,
        stream,
        output_stream,
        onnx_model=None,
        device_mode="cpu",
        confidence=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not onnx_model:
            raise ValueError("Provide parameter onnx_model as model path")

        self._run_flag = True
        self.input_stream = stream
        self.output = output_stream
        self.frames_to_process = []
        if device_mode == "cpu":
            self.batch = max(
                4, os.cpu_count() - 4
            )  # остальная система тоже хочет ресурсов
        else:
            self.batch = 8  # 8images batch for gpu
        self.model = YoloONNX(
            onnx_model, device=device_mode, batch=self.batch, confidence=confidence
        )

        self.fps_q = deque(maxlen=10)

    def fps(self):
        fps = (
            "0"
            if not len(self.fps_q)
            else f"{1 / (sum(self.fps_q) / len(self.fps_q)):.2f}"
        )
        return fps

    def run(self):

        while self._run_flag:
            if self.input_stream.qsize() >= self.batch:
                self.frames_to_process = self.input_stream.get_batch()

                start = time()
                frames = self.model(self.frames_to_process)
                self.fps_q.append((time() - start) / self.batch)

                for frame in frames:
                    self.output.put_nowait(frame)
                self.frames_to_process.clear()
            elif self.input_stream.qsize() == 0:
                sleep(0.1)
            else:
                sleep(0.02)

    def stop(self):
        self._run_flag = False

    def restart(self):
        self._run_flag = True
