from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage
from collections import deque
from time import time, sleep
from queue import Empty, Queue

class Render(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, imput_queue: Queue, width: int, height: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source = imput_queue
        self.width = width
        self.height = height

        self.update_interval = 30

        self.timer = QTimer()
        self.timer.timeout.connect(self.render_frame)
        self.timer.setInterval(self.update_interval)
        self.fifo_fill_level = 20
        self.cnt = 0
        self.frame_fps = deque(maxlen=10)

    def reset(self):
        """
        clear input buffer
        """
        self.source.queue.clear()
    @property
    def fps(self):
        if len(self.frame_fps):
            return sum(self.frame_fps) / len(self.frame_fps)

        return 0

    def render_frame(self):
        try:
            #read frame and convert to pyqt format
            frame = self.source.get_nowait()

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(
                frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            )
            p = convert_to_Qt_format.scaled(self.width, self.height)
            self.change_pixmap_signal.emit(p)

        except Empty:
            return

        mean_fill = self.source.qsize()
        self.cnt += 1

        #update render timer interval
        if self.cnt % 10 == 0:
            error = mean_fill - self.fifo_fill_level
            err = error * 0.002
            self.update_interval *= 1 - err
            self.update_interval = int(self.update_interval)

            if self.update_interval < 5:
                self.update_interval = 5

            if self.update_interval > 120:
                self.update_interval = 120

            self.frame_fps.append(1000 / self.update_interval)

            self.timer.setInterval(self.update_interval)

        sleep(0.01)

    def stop(self):
        self.timer.stop()

    def start(self):
        self.timer.start()
