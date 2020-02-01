# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(426, 178)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(426, 178))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.pushButton_run = QPushButton(self.centralwidget)
        self.pushButton_run.setObjectName(u"pushButton_run")
        self.pushButton_run.setGeometry(QRect(305, 95, 112, 32))
        self.pushButton_test = QPushButton(self.centralwidget)
        self.pushButton_test.setObjectName(u"pushButton_test")
        self.pushButton_test.setGeometry(QRect(5, 95, 112, 32))
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(5, 5, 416, 81))
        self.pushButton_input_file = QPushButton(self.groupBox)
        self.pushButton_input_file.setObjectName(u"pushButton_input_file")
        self.pushButton_input_file.setGeometry(QRect(300, 20, 112, 32))
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 25, 58, 16))
        self.lineEdit_input_file = QLineEdit(self.groupBox)
        self.lineEdit_input_file.setObjectName(u"lineEdit_input_file")
        self.lineEdit_input_file.setGeometry(QRect(132, 25, 156, 21))
        self.lineEdit_n_proc = QLineEdit(self.groupBox)
        self.lineEdit_n_proc.setObjectName(u"lineEdit_n_proc")
        self.lineEdit_n_proc.setGeometry(QRect(132, 50, 156, 21))
        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 50, 116, 16))
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(0, 85, 426, 16))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 426, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
#if QT_CONFIG(statustip)
        self.pushButton_run.setStatusTip(QCoreApplication.translate("MainWindow", u"Run simulation", None))
#endif // QT_CONFIG(statustip)
        self.pushButton_run.setText(QCoreApplication.translate("MainWindow", u"Run", None))
#if QT_CONFIG(statustip)
        self.pushButton_test.setStatusTip(QCoreApplication.translate("MainWindow", u"Make and save a template input file", None))
#endif // QT_CONFIG(statustip)
        self.pushButton_test.setText(QCoreApplication.translate("MainWindow", u"Make a Temp.", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"GroupBox", None))
        self.pushButton_input_file.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Input File", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"No. of Processes", None))
    # retranslateUi

