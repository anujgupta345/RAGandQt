from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background-color: rgba(255,255,255,220);')
        l = QVBoxLayout(self)
        l.addStretch()

        self.msg_label = QLabel('Processing Documents...')
        self.msg_label.setAlignment(Qt.AlignCenter)
        l.addWidget(self.msg_label)

        self.spinner_label = QLabel('◐')
        self.spinner_label.setAlignment(Qt.AlignCenter)
        font = self.spinner_label.font()
        font.setPointSize(24)
        self.spinner_label.setFont(font)
        l.addWidget(self.spinner_label)

        l.addStretch()

        self.frames = ['◐', '◓', '◑', '◒']
        self.current_frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_spinner)
        self.hide()

    def update_spinner(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.spinner_label.setText(self.frames[self.current_frame])

    def set_message(self, text):
        self.msg_label.setText(text)

    def showEvent(self, event):
        self.timer.start(100)
        super().showEvent(event)

    def hideEvent(self, event):
        if hasattr(self, 'timer'):
            self.timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, e):
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(e)