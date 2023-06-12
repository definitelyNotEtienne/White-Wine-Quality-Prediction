# Prediction of White Wine Quality

# Project Description 
It is an assistance system application which assist the user in predicting the quality of the white wine through the user's input such as the value of pH, alcohol level, density etc.

The dataset used is from https://www.kaggle.com/datasets/yasserh/wine-quality-dataset.

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# Installation 
For MacOS Ventura 13.1: Required Python 3.9

1. Download the file test-project-for-a-system.zip file and extract it.
2. Install the required modules from the requirement.txt file.

# Basic Usage

1. Run the main.py file. 
2. Insert your desired value in the user interface.

# Implementation and Request
1. The Dataset is used from https://www.kaggle.com/datasets/yasserh/wine-quality-dataset in the csv format.
2. The Data is read and created a histogram after clicking on it in the menu button. 
```python
# menu bar with File button
        menu = self.menuBar()
        self.file_menu = menu.addMenu("&File")
        self.file_menu.addAction(self.button)
        self.file_menu.addAction(self.button1)
        self.file_menu.addAction(self.button2)
```
The code above implements the "File" menu button and it has another button under it that reads the data and create the histogram.
4. The data is then analyzed using Pandas method and the overview of the data is shown.
```python
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
...
```
5. QtLabels are used to label the widgets in the GUI meanwhile, dial, sliders and double spin boxes are used as widgets for user to input the variable data. A function is then created respectively to state the changes of the variables made by the user.
```python
# example for dial code
        self.freelabel = QLabel(self)
        self.freelabel.setText("Free Sulfur Dioxide: ")
        self.free_ds = QDial()
        self.free_ds.setRange(0, 289)
        self.free_ds.setValue(1)
        self.free_ds.valueChanged.connect(self.free_ds_value_changed)

# example for slider
        self.total_ds_label = QLabel(self)
        self.total_ds_label.setText("Total Sulfur Dioxide: ")
        self.total_ds = QSlider(Qt.Orientation.Horizontal)
        self.total_ds.setRange(9, 440)
        self.total_ds.valueChanged.connect(self.total_ds_value_changed)

# example for double spin box
        self.vacid_label = QLabel(self)
        self.vacid_label.setText("Volatile Acidity: ")
        self.vacid = QDoubleSpinBox()
        self.vacid.setMinimum(0.00)
        self.vacid.setMaximum(1.20)
        self.vacid.valueChanged.connect(self.vacid_value_changed)

# example function that states changes
def free_ds_value_changed(self):
        val = self.free_ds.value()
        self.freelabel.setText("Free Sulfur Dioxide: " + str(val))
        print(val)
```
6. Logistic Regression and Random Forest Classifier is used to train the dataset. 


