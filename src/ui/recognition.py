from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Recognition(object):
    def setupUi(self, Recognition):
        Recognition.setObjectName("Recognition")
        Recognition.resize(640, 480)
        self.layout = QtWidgets.QVBoxLayout(Recognition)
        self.layout.setObjectName("layout")
        self.feed_view = QtWidgets.QLabel(parent=Recognition)
        self.feed_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.feed_view.setObjectName("feed_view")
        self.layout.addWidget(self.feed_view)

        self.btn_select_video = QtWidgets.QPushButton(parent=Recognition)
        self.btn_select_video.setObjectName("btn_select_video")
        self.layout.addWidget(self.btn_select_video)

        self.btn_use_camera = QtWidgets.QPushButton(parent=Recognition)
        self.btn_use_camera.setObjectName("btn_use_camera")
        self.layout.addWidget(self.btn_use_camera)

        self.btn_start = QtWidgets.QPushButton(parent=Recognition)
        self.btn_start.setObjectName("btn_start")
        self.layout.addWidget(self.btn_start)
        self.btn_stop = QtWidgets.QPushButton(parent=Recognition)
        self.btn_stop.setObjectName("btn_stop")
        self.layout.addWidget(self.btn_stop)

        self.retranslateUi(Recognition)
        QtCore.QMetaObject.connectSlotsByName(Recognition)

    def retranslateUi(self, Recognition):
        _translate = QtCore.QCoreApplication.translate
        Recognition.setWindowTitle(_translate("Recognition", "Video/Camera Player"))
        self.feed_view.setText(
            _translate("Recognition", "Video or Camera will be displayed here")
        )
        self.btn_select_video.setText(_translate("Recognition", "Select Video"))
        self.btn_use_camera.setText(_translate("Recognition", "Use Camera"))
        self.btn_start.setText(_translate("Recognition", "Start"))
        self.btn_stop.setText(_translate("Recognition", "Stop"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Recognition = QtWidgets.QWidget()
    ui = Ui_Recognition()
    ui.setupUi(Recognition)
    Recognition.show()
    sys.exit(app.exec())
