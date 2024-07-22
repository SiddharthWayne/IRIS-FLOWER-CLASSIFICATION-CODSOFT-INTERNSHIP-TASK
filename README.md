
# Iris Flower Classification - CODSOFT INTERNSHIP TASK

The Iris flower dataset consists of three species: setosa, versicolor, and virginica. These species can be distinguished based on their measurements. The objective of this project is to train a machine learning model that can learn from these measurements and accurately classify the Iris flowers into their respective species.

You can run any one of these files to see the model validation. The choice of file depends on your requirements: for a quick and efficient run, use model.py or app.py.


## Description

The project consists of four main files:

1) model.py: This file contains the code to train and run the machine learning model efficiently.

2) app.py: This file sets up a Streamlit web application for an efficient and user-friendly interface to input transaction details and predict fraud.

You can run any one of these files to see the model validation. The choice of file depends on your requirements: for a quick and efficient run, use model.py or app.py.

The dataset used for this project is iris.csv, which contains various features related to credit card transactions.
## Acknowledgements

 We would like to thank the following resources and individuals for their contributions and support:

Streamlit: For offering an easy-to-use framework for deploying machine learning models.

Scikit-learn: For providing powerful machine learning tools and libraries.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

## Demo

https://drive.google.com/file/d/10KiUh0Qd57Ga3rA5M88lONp8ovtpK2w3/view?usp=drive_link

You can see a live demo of the application by running the app.py file. The Streamlit app allows you to input flower measurements and get a predicted species based on the trained model.
## Features

Data Loading and Preprocessing: The model can load and preprocess data from the iris.csv file.

Model Training: Utilizes a classification algorithm to train the model on flower measurements.

Interactive User Input: Through the Streamlit app, users can input flower measurements and receive a predicted species.

Model Evaluation: Evaluates model performance using metrics like accuracy.
## Technologies Used

Python: The programming language used to implement the model and the 
Streamlit app.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

Scikit-learn: For building and training the machine learning model.

Streamlit: For creating the interactive web application.
## Installation

To get started with this project, follow these steps:

1) Clone the repository:

git clone https://github.com/SiddharthWayne/IRIS-FLOWER-CLASSIFICATION-CODSOFT-INTERNSHIP-TASK.git

cd iris-flower-classification

2) Install the required packages:

pip install -r requirements.txt

Ensure that requirements.txt includes the necessary dependencies like pandas, numpy, scikit-learn, and streamlit.

3) Download the dataset:

Place the iris.csv file in the project directory. Make sure the path in model.py and app.py is correctly set to this file.



## Usage/Examples

1) Running the Model (model.py)

To train and run the model using the command line, execute the following:

python model.py

This will train the model and allow you to input transaction details via the command line interface to get a predicted fraud status.

2) Running the Streamlit App (app.py)

To run the Streamlit app for an interactive experience, execute the following:

streamlit run app.py

This will start the Streamlit server, and you can open your web browser to the provided local URL to use the app.


Example:

Once the Streamlit app is running, you can input flower measurements such as:

Sepal Length: 5.1 cm

Sepal Width: 3.5 cm

Petal Length: 1.4 cm

Petal Width: 0.2 cm

Click the "Predict Species" button to get the predicted species.



