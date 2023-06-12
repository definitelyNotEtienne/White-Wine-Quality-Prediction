# imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# import PyQt6
from PyQt6 import QtGui, QtCore

# import Q Widgets
from PyQt6.QtWidgets import (
    QWidget,
    QApplication,
    QMainWindow,
    QToolBar,
    QStatusBar,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QDial,
    QSlider,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QStackedLayout,
    QPushButton,

)

from PyQt6.QtGui import QAction, QIcon, QColor, QPalette
from PyQt6.QtCore import QCoreApplication, Qt

# import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# import matplotlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, FigureCanvasQTAgg
from matplotlib.figure import Figure

# global vars
app_title = 'White Wine Quality Prediction'
white_wine = []
IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution)  # ax=self.sc.axes,

# Color Widget Class
class Color(QWidget):
    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)

# creates a canvas
class Canvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(Canvas, self).__init__(self.figure)


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global white_wine
        white = pd.read_csv('winequality-white.csv', sep=';')
        # using this because can't get global data
        widget = QWidget()
        # Vertical Layout
        layout2 = QVBoxLayout()
        # Create a new Canvas
        self.setWindowTitle("White Wine Histogram")
        self.sc2 = Canvas(self, width=10, height=6, dpi=100)
        white.hist(ax=self.sc2.axes)  # currently having UserWarning Error
        save_fig(self.sc2.figure, "Histogram", tight_layout=True, fig_extension="png", resolution=300)  # extra code
        layout2.addWidget(self.sc2)
        self.setCentralWidget(widget)
        widget.setLayout(layout2)


class PredictionGraph(QMainWindow):
    def __init__(self):
        super().__init__()
        global white_wine

        self.setWindowTitle("White Wine Quality Prediction Result")
        self.sc = Canvas(self, width=5, height=4, dpi=100)
        


# Main app window
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Wine Quality Prediction")
        self.setFixedSize(600, 600)

        self.initUI()
        self.loadWhiteWineData()
        self.compute_regression()

    # def show_interface(self):
    # if self.interface.isHidden():
    #   self.interface.show()

    # to show histogram window
    def show_histogram(self):
        if self.histogram_window.isHidden():
            self.histogram_window.show()

    # to show prediction result window
    def show_prediction(self):
        # compute list of variables in the right order ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        x = [self.facid.value(), self.vacid.value(), self.cacid.value(), self.sugar.value(), self.chloride.value(), self.total_ds.value(), self.density.value(), self.ph.value(), self.sulphate.value(), self.alcohol.value()]

        # predict the probabilities and the quality class
        x = np.array(x)
        pred1 = self.model.predict_proba(x.reshape(1,-1))
        pred1 = pred1.reshape(1,-1).tolist()[0]
        # print(pred1)
        print(self.model.predict(x.reshape(1,-1)))
        # plot the figure 
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        classes = ["3","4","5","6","7","8","9"]
        values = pred1
        ax.bar(classes,values)
        plt.show()

        if self.prediction_window.isHidden():
            self.prediction_window.show()

    def initUI(self):
        global red_wine
        global white_wine

        # self.interface = InterfaceWindow()
        self.histogram_window = HistogramWindow()
        self.prediction_window = PredictionGraph()
        # set Appilcation default styles
        font = QtGui.QFont("Sanserif", 12)
        # ont.setStyleHint(QtGui.QFont.)
        QApplication.setFont(QtGui.QFont(font))
        # QApplication.setWindowIcon(QtGui.QIcon(''))

        #layout = QGridLayout()

        # Free Sulfur Dioxide Widget
        self.freelabel = QLabel(self)
        self.freelabel.setText("Free Sulfur Dioxide: ")
        self.free_ds = QDial()
        self.free_ds.setRange(0, 289)
        self.free_ds.setValue(1)
        self.free_ds.valueChanged.connect(self.free_ds_value_changed)

        # Total Sulfur Dioxide Widget
        self.total_ds_label = QLabel(self)
        self.total_ds_label.setText("Total Sulfur Dioxide: ")
        self.total_ds = QSlider(Qt.Orientation.Horizontal)
        self.total_ds.setRange(9, 440)
        self.total_ds.valueChanged.connect(self.total_ds_value_changed)

        # Volatile Acidity Widget
        self.vacid_label = QLabel(self)
        self.vacid_label.setText("Volatile Acidity: ")
        self.vacid = QDoubleSpinBox()
        self.vacid.setMinimum(0.00)
        self.vacid.setMaximum(1.20)
        self.vacid.valueChanged.connect(self.vacid_value_changed)

        # Citric Acid Widget
        self.cacid_label = QLabel(self)
        self.cacid_label.setText("Citric Acid: ")
        self.cacid = QDoubleSpinBox()
        self.cacid.setMinimum(0.00)
        self.cacid.setMaximum(1.70)
        self.cacid.valueChanged.connect(self.cacid_value_changed)

        # Fixed Acidity Widget
        self.facid_label = QLabel(self)
        self.facid_label.setText("Fixed Acidity: ")
        self.facid = QDoubleSpinBox()
        self.facid.setMinimum(1.0)
        self.facid.setMaximum(14.5)
        self.facid.valueChanged.connect(self.facid_value_changed)

        # Residual Sugar Widget
        self.sugar_label = QLabel(self)
        self.sugar_label.setText("Residual Sugar: ")
        self.sugar = QDoubleSpinBox()
        self.sugar.setMinimum(0.6)
        self.sugar.setMaximum(66.0)
        self.sugar.valueChanged.connect(self.sugar_value_changed)

        # Chloride Widget
        self.chloride_label = QLabel(self)
        self.chloride_label.setText("Chloride: ")
        self.chloride = QDoubleSpinBox()
        self.chloride.setMinimum(0.001)
        self.chloride.setMaximum(0.350)
        self.chloride.valueChanged.connect(self.chloride_value_changed)

        # Density Widget
        self.density_label = QLabel(self)
        self.density_label.setText("Density: ")
        self.density = QDoubleSpinBox()
        self.density.setMinimum(0.98711)
        self.density.setMaximum(1.03898)
        self.density.valueChanged.connect(self.density_value_changed)

        # pH Widget
        self.ph_label = QLabel(self)
        self.ph_label.setText("pH Value: ")
        self.ph = QDoubleSpinBox()
        self.ph.setMinimum(2.71)
        self.ph.setMaximum(3.85)
        self.ph.valueChanged.connect(self.ph_value_changed)

        # Sulphate Widget
        self.sulphate_label = QLabel(self)
        self.sulphate_label.setText("Sulphate: ")
        self.sulphate = QDoubleSpinBox()
        self.sulphate.setMinimum(0.22)
        self.sulphate.setMaximum(1.08)
        self.sulphate.valueChanged.connect(self.sulphate_value_changed)

        # Alcohol Widget
        self.alcohol_label = QLabel(self)
        self.alcohol_label.setText("Alcohol: ")
        self.alcohol = QDoubleSpinBox()
        self.alcohol.setMinimum(8.0)
        self.alcohol.setMaximum(14.2)
        self.alcohol.valueChanged.connect(self.alcohol_value_changed)

        self.button = QAction("Histogram", self)
        self.button.setStatusTip("White Wine Histogram")
        self.button.triggered.connect(self.show_histogram)

        self.button1 = QAction("Save Prediction", self)
        self.button1.setStatusTip("Save the prediction result.")
        self.button1.triggered.connect(self.save_click)

        self.button2 = QAction("Close Application", self)
        self.button2.setStatusTip("Close Apllication")
        self.button2.triggered.connect(self.closeEvent)

        # Button to get prediction result
        self.predict = QPushButton("Predict", self)
        self.predict.setStatusTip("Show Prediction")
        self.predict.clicked.connect(self.show_prediction)

        # menu bar with File button
        menu = self.menuBar()
        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.button)
        self.file_menu.addAction(self.button1)
        self.file_menu.addAction(self.button2)

        self.setStatusBar(QStatusBar(self))

        # create layout in dummy widget
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QGridLayout(widget)
        # add widgets into layout
        layout.addWidget(self.free_ds, 1, 0)
        layout.addWidget(self.freelabel, 0, 0)
        layout.addWidget(self.total_ds, 3, 0)
        layout.addWidget(self.total_ds_label, 2, 0)
        layout.addWidget(self.vacid, 5, 0)
        layout.addWidget(self.vacid_label, 4, 0)
        layout.addWidget(self.cacid, 7, 0)
        layout.addWidget(self.cacid_label, 6, 0)
        layout.addWidget(self.facid, 9, 0)
        layout.addWidget(self.facid_label, 8, 0)
        layout.addWidget(self.sugar, 11, 0)
        layout.addWidget(self.sugar_label, 10, 0)
        layout.addWidget(self.chloride, 13, 0)
        layout.addWidget(self.chloride_label, 12, 0)
        layout.addWidget(self.density, 15, 0)
        layout.addWidget(self.density_label, 14, 0)
        layout.addWidget(self.ph, 17, 0)
        layout.addWidget(self.ph_label, 16, 0)
        layout.addWidget(self.sulphate, 1, 2)
        layout.addWidget(self.sulphate_label, 1, 1)
        layout.addWidget(self.alcohol, 2, 2)
        layout.addWidget(self.alcohol_label, 2, 1)
        layout.addWidget(self.predict, 3, 2)

    def free_ds_value_changed(self):
        val = self.free_ds.value()
        self.freelabel.setText("Free Sulfur Dioxide: " + str(val))
        print(val)

    def total_ds_value_changed(self):
        val = self.total_ds.value()
        self.total_ds_label.setText("Fixed Acidity: " + str(val))
        print(val)

    def vacid_value_changed(self):
        val = self.vacid.value()
        self.vacid_label.setText("Volatile Acidity: " + str(val))
        print(val)

    def cacid_value_changed(self):
        val = self.cacid.value()
        self.cacid_label.setText("Citric Acid: " + str(val))
        print(val)

    def facid_value_changed(self):
        val = self.facid.value()
        self.facid_label.setText("Fixed Acidity: " + str(val))
        print(val)

    def sugar_value_changed(self):
        val = self.sugar.value()
        self.sugar_label.setText("Residual Sugar: " + str(val))
        print(val)

    def chloride_value_changed(self):
        val = self.chloride.value()
        self.chloride_label.setText("Chloride: " + str(val))
        print(val)

    def density_value_changed(self):
        val = self.density.value()
        self.density_label.setText("Density: " + str(val))
        print(val)

    def ph_value_changed(self):
        val = self.ph.value()
        self.ph_label.setText("pH Value: " + str(val))
        print(val)

    def sulphate_value_changed(self):
        val = self.sulphate.value()
        self.sulphate_label.setText("Sulphate: " + str(val))
        print(val)

    def alcohol_value_changed(self):
        val = self.alcohol.value()
        self.alcohol_label.setText("Alcohol: " + str(val))
        print(val)

    def save_click(self):
        save_fig(self.sc.figure, "prediction_plot")
  
    # load data set 

    def loadWhiteWineData(self):
        global white_wine

        # open the data
        self.white_wine = pd.read_csv('winequality-white.csv', delimiter=';')

        # shows 12 rows of the data
        print("Data Preview: ")
        print(self.white_wine.head(12))

        # shows info about the data
        print("Info: ")
        print(self.white_wine.info())

        # describe data
        print("Description: ")
        print(self.white_wine.describe())

        # find correlation between columns
        print("Data Correlation: ")
        corr_white = self.white_wine.corr()
        print(corr_white)

        # calucalte max values
        print("Max Values:")
        max_white = self.white_wine.max()
        print(max_white)

        # calculate min vlaues
        print("Min Vlaues: ")
        min_white = self.white_wine.min()
        print(min_white)

        # clean data 
         
        # assign to global variable
        white_wine = self.white_wine

     
    # data preperation 
    def compute_regression(self):
        global white_wine
        # https://www.kaggle.com/code/sergey18650/wine-quality-randomforest-acc-82
        # splitting dataset for training
        X = white_wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
          'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].to_numpy()
        y = white_wine[['quality']].to_numpy().ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train, y_test = y_train.ravel(), y_test.ravel()

        self.model = make_pipeline(StandardScaler(),RandomForestClassifier(criterion='gini',
                                                                n_estimators=150,
                                                                random_state=1,
                                                                n_jobs=2))
        self.model.fit(X_train,y_train)

        y_pred = self.model.predict(X_test)
        print('Acc:', accuracy_score(y_test, y_pred))

    # WIP close dialog
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)

        if (reply == QMessageBox.StandardButton.Close):
            print("Close")
            sys.exit()
        else:
            if (reply == QMessageBox.StandardButton.Save):
                save_fig(self.sc.figure, "Wine Quality Prediction Plot")
                sys.exit()
            else:
                print("Cancel")
                if not type(event) == bool:
                    event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    form = MainWindow()
    form.show()

    app.exec()

