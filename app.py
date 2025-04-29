import os
import sys
from collections import deque
from queue import Empty, Queue
from time import sleep

# import onnxruntime as ort


from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from consumers import OnnxRunner, VideoThread
from model_onnx import get_providers
from resample_queue import ResampleQueue


class Renderer(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, frame_queue, width, height, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source = frame_queue
        self.width = width
        self.height = height

        self.update_interval = 30

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(self.update_interval)
        self.timer.start()
        self.fifo_fill_level = 20
        self.cnt = 0
        self._run_flag = False
        self.frame_fps = deque(maxlen=10)

    def reset_buf(self):
        self.source.queue.clear()

    def fps(self):
        if len(self.frame_fps):
            return sum(self.frame_fps) / len(self.frame_fps)

        return 0

    def update_frame(self):
        try:
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

    def run(self):
        while self._run_flag:
            sleep(0.1)

    def stop(self):
        self._run_flag = True


class cQLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def __init__(self, widget):
        super().__init__(widget)

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Onnx real view runner")
        self.setGeometry(100, 100, 1920 // 2, 1080 // 2)
        self.width = 800
        self.height = 600

        self.timer = QTimer()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.file_label = QLabel(self)
        self.file_label.setText("Видеофайл")
        self.video_source = cQLineEdit(self.central_widget)
        self.video_source.clicked.connect(self.choose_sorce)

        self.models_label = QLabel(self)
        self.models_label.setText("Модели")
        self.model_folder = cQLineEdit(self.central_widget)
        self.model_folder.setText(".")
        self.model_folder.clicked.connect(self.choose_model_folder)

        self.device_label = QLabel(self)
        self.model_file = QComboBox()

        self.model_file.addItems(
            [i for i in os.listdir(self.model_folder.text()) if i.endswith(".onnx")]
        )

        self.conf_value = QLabel(self)
        self.conf_value.setText("Confidence %")

        self.conf = QSlider(self)
        self.conf.setOrientation(Qt.Orientation.Horizontal)
        self.conf.setMaximum(100)
        self.conf.setValue(50)
        self.conf.setMinimum(10)
        self.conf.setTickInterval(1)

        self.conf.valueChanged.connect(
            lambda value: self.conf_value.setText(f"Confidence {self.conf.value()} %")
        )

        self.device_label.setText("Устройство (CUDA 11.8)")
        self.perf_mode = QComboBox()
        self.perf_mode.addItems(get_providers())

        self.start_button = QPushButton("Старт")
        self.start_button.clicked.connect(self.start_video)
        self.stop_button = QPushButton("Стоп")
        self.stop_button.clicked.connect(self.stop_video)

        self.perf_label = QLabel(self)
        self.perf_label.setText("0 FPS")

        self.video_fps = QLabel(self)
        self.video_fps.setText("0 FPS")

        control_layout = QVBoxLayout()
        control_layout.addWidget(self.file_label)
        control_layout.addWidget(self.video_source)
        control_layout.addWidget(self.models_label)
        control_layout.addWidget(self.model_folder)
        control_layout.addWidget(self.model_file)
        control_layout.addWidget(self.device_label)
        control_layout.addWidget(self.perf_mode)
        control_layout.addWidget(self.conf_value)

        control_layout.addWidget(self.conf)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.perf_label)
        control_layout.addWidget(self.video_fps)

        self.time_slider = QSlider(Qt.Orientation.Horizontal, self.central_widget)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.valueChanged.connect(self.on_time_changed)

        self.time_label = QLabel("00:00 / 00:00", self.central_widget)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.time_label.setStyleSheet("QLabel { padding: 2px; }")
        self.time_label.setFixedHeight(20)

        # ------------------------------------------------
        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setMaximumWidth(200)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.image_label)

        time_layout = QHBoxLayout()
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)

        video_layout.addLayout(time_layout)

        central_layout = QHBoxLayout()
        central_layout.addLayout(video_layout)
        central_layout.addWidget(control_widget)

        # Устанавливаем растяжку для метки с видео
        central_layout.setStretchFactor(video_layout, 1)
        central_layout.setStretchFactor(control_widget, 0)

        self.central_widget.setLayout(central_layout)

        self.qt_img = None

        self.frame_queue = ResampleQueue()
        self.render_source = Queue()
        self.video_thread = None
        self.onnx_thread = None

    def update_slider_position(self, pos):
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(pos)
        self.time_slider.blockSignals(False)

        total_duration = self.video_thread.duration()

        total_secs = total_duration * pos / 1000
        mins = int(total_secs // 60)
        secs = int(total_secs % 60)

        total_mins = int(total_duration // 60)
        total_secs = int(total_duration % 60)

        self.time_label.setText(
            f"{mins}:{str(secs).zfill(2)} / {total_mins}:{total_secs}"
        )

    def on_time_changed(self, pos):
        self.video_thread.set_position(pos)
        self.render.reset_buf()

    def start_video(self):
        if not self.video_source.text():
            QMessageBox.warning(self, "Warning", "Нужно выбрать файл")
            return

        if not self.model_file.currentText():
            QMessageBox.warning(self, "Warning", "Нужно выбрать файл модели")
            return

        onnx_cfg = {}
        onnx_cfg["onnx_model"] = (
            f"{self.model_folder.text()}/{self.model_file.currentText()}"
        )
        onnx_cfg["device_mode"] = self.perf_mode.currentText()
        onnx_cfg["confidence"] = self.conf.value() / 100

        self.video_thread = VideoThread(self.frame_queue, self.video_source.text())
        self.video_thread.position_signal.connect(self.update_slider_position)
        self.video_thread.start()

        self.onnx_thread = OnnxRunner(self.frame_queue, self.render_source, **onnx_cfg)
        self.onnx_thread.start()

        self.render = Renderer(self.render_source, 1920 // 2, 1080 // 2)
        self.render.change_pixmap_signal.connect(self.update_image)
        self.render.start()

        self.timer.timeout.connect(self.print_fps)
        self.timer.start(1000)  # время обновления FPS

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()

        if self.onnx_thread:
            self.onnx_thread.stop()
            self.onnx_thread.quit()
            self.onnx_thread.wait()

        if self.render:
            self.render.stop()
            self.render.quit()
            self.render.wait()

    def choose_model_folder(self):
        options = QFileDialog.Option.DontUseNativeDialog
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку", options=options
        )
        if folder_path:
            self.model_file.clear()
            onnx_files = [i for i in os.listdir(folder_path) if i.endswith(".onnx")]
            self.model_file.addItems(onnx_files)

    def choose_sorce(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл",
            "",
            "Все файлы (*);;Видео файлы (*.mp4)",
            options=options,
        )

        if file_name:
            self.video_source.setText(file_name)

    def print_fps(self):
        if self.onnx_thread and self.render:
            self.perf_label.setText(f"Network: {self.onnx_thread.fps()} FPS")
            self.video_fps.setText(f"Rendering: {self.render.fps():.2f} FPS")

    def update_image(self, cv_img):
        self.qt_img = QPixmap.fromImage(cv_img)
        self.image_label.setPixmap(self.qt_img)
        original_size = self.qt_img.size()

        self.scale_image()
        self.setMinimumSize(original_size.width(), original_size.height())

    def scale_image(self):
        if self.qt_img is not None:
            scaled_pixmap = self.qt_img.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.scale_image()
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.quit()
            self.video_thread.wait()

        if self.onnx_thread:
            self.onnx_thread.stop()
            self.onnx_thread.quit()
            self.onnx_thread.wait()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec())
