from PySide2 import QtCore, QtGui, QtWidgets


class Dialog(QtWidgets.QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self):
        super(Dialog, self).__init__()

        self.createMenu()
        self.createGridGroupBox()
        self.createFormGroupBox()

        bigEditor = QtWidgets.QTextEdit()
        bigEditor.setPlainText("This widget takes up all the remaining space "
                               "in the top-level layout.")

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.setMenuBar(self.menuBar)
        # mainLayout.addWidget(self.horizontalGroupBox)
        mainLayout.addWidget(self.gridGroupBox)
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(bigEditor)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Basic Layouts")


    def createMenu(self):
        self.menuBar = QtWidgets.QMenuBar()

        self.fileMenu = QtWidgets.QMenu("&File", self)
        self.exitAction = self.fileMenu.addAction("E&xit")
        self.menuBar.addMenu(self.fileMenu)

        self.exitAction.triggered.connect(self.accept)


    def createGridGroupBox(self):
        self.gridGroupBox = QtWidgets.QGroupBox("Grid layout")
        layout = QtWidgets.QGridLayout()

        # row one
        r1_label = QtWidgets.QLabel("Door exit capacity")
        r1_edit = QtWidgets.QLineEdit()
        r1_button = QtWidgets.QPushButton("Folder")
        layout.addWidget(r1_label, 0, 0)
        layout.addWidget(r1_edit, 0, 1)
        layout.addWidget(r1_button, 0, 2, 1, 2)

        r2_label = QtWidgets.QLabel("Preferences")
        r2_edit = QtWidgets.QLineEdit()

        def r2_act_button_1(): r2_edit.setText(self.get_file_path(filter="YAML (*.yaml, *.txt)"))

        r2_button_1 = QtWidgets.QPushButton("File")
        r2_button_1.setFixedWidth(45)
        r2_button_1.clicked.connect(r2_act_button_1)
        r2_button_2 = QtWidgets.QPushButton("Editor")
        r2_button_2.setFixedWidth(45)
        layout.addWidget(r2_label, 1, 0)
        layout.addWidget(r2_edit, 1, 1)
        layout.addWidget(r2_button_1, 1, 2)
        layout.addWidget(r2_button_2, 1, 3)

        r3_label = QtWidgets.QLabel("Inputs")
        r3_edit = QtWidgets.QLineEdit()

        def r3_act_button_1(): r3_edit.setText(self.get_file_path(filter="YAML (*.yaml, *.txt)"))

        r3_button_1 = QtWidgets.QPushButton("File")
        r3_button_1.setFixedWidth(45)
        r3_button_1.clicked.connect(r3_act_button_1)
        r3_button_2 = QtWidgets.QPushButton("Editor")
        r3_button_2.setFixedWidth(45)
        layout.addWidget(r3_label, 4, 0)
        layout.addWidget(r3_edit, 4, 1)
        layout.addWidget(r3_button_1, 4, 2)
        layout.addWidget(r3_button_2, 4, 3)

        r2_label = QtWidgets.QLabel("Settings:")
        line_edit_2 = QtWidgets.QLineEdit()

        self.smallEditor = QtWidgets.QTextEdit()
        self.smallEditor.setPlainText("This widget takes up about two thirds "
                                      "of the grid layout.")

        layout.setColumnStretch(1, 10)
        layout.setColumnStretch(2, 20)
        self.gridGroupBox.setLayout(layout)

    def get_file_path(self, caption=None, dir=None, filter=None):
        my_dialog = QtWidgets.QFileDialog(self)
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(my_dialog, caption=caption, dir=dir, filter=filter)
        return file_name

    def create_action(self):
        self.openAct = QtWidgets.QAction(QtGui.QIcon(':/images/open.png'),
                                         "&Open...", self, shortcut=QtGui.QKeySequence.Open,
                                         statusTip="Open an existing file", triggered=self.open)

    def createFormGroupBox(self):
        self.formGroupBox = QtWidgets.QGroupBox("Form layout")
        layout = QtWidgets.QFormLayout()
        layout.addRow(QtWidgets.QLabel("Line 1:"), QtWidgets.QLineEdit())
        layout.addRow(QtWidgets.QLabel("Line 2, long text:"), QtWidgets.QComboBox())
        layout.addRow(QtWidgets.QLabel("Line 3:"), QtWidgets.QSpinBox())
        self.formGroupBox.setLayout(layout)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    dialog = Dialog()
    sys.exit(dialog.exec_())
