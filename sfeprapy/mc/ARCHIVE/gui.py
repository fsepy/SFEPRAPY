import sys

from PySide2.QtWidgets import QDialog, QApplication
from PySide2.QtWidgets import QHBoxLayout, QVBoxLayout
from PySide2.QtWidgets import QLineEdit, QLabel, QPushButton
import PySide2.QtWidgets as QtWidgets
from PySide2 import QtGui, QtCore
import pandas
import os


class TabForm(QtWidgets.QTabWidget):
    """"""

    def __init__(self, parent=None):
        """Constructor"""
        super(TabForm, self).__init__(parent)

    def button_click(self):
        for i in self.list_qt_edit:
            print(i.text())
        print('hello')


class Form(QDialog):
    """"""

    def __init__(self, df_data, parent=None):
        """Constructor"""
        super(Form, self).__init__(parent)

        self.setWindowTitle('SfePraPy')

        layout_main = QtWidgets.QVBoxLayout()
        layout_param = QtWidgets.QFormLayout()

        list_qt_label = []
        self.list_qt_edit = []
        dict_validator = {'int': QtGui.QIntValidator(), 'float': QtGui.QDoubleValidator(), 'str': None}
        dict_fmt = {'int': '{:.0f}', 'float': '{}', 'str': '{}'}
        for i, r in df_data.iterrows():

            list_qt_label.append(QtWidgets.QLabel(r['DISPLAY NAME']))

            self.list_qt_edit.append(QLineEdit())

            self.list_qt_edit[-1].setValidator(dict_validator[r['TYPE']])
            self.list_qt_edit[-1].setText(dict_fmt[r['TYPE']].format(r['VALUE']))

            layout_param.addRow(list_qt_label[-1], self.list_qt_edit[-1])

        layout_main.addLayout(layout_param, stretch=1)

        button = QPushButton('&Submit')
        button.clicked.connect(self.button_click)

        layout_main.addWidget(button)
        self.setLayout(layout_main)

    def button_click(self):
        for i in self.list_qt_edit:
            print(i.text())


def gui_portal(df_data=None):

    if df_data is None:
        df_data = pandas.read_csv('gui.csv', index_col='INDEX')

    app = QApplication([])
    form = Form(df_data)
    form.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    gui_portal()
