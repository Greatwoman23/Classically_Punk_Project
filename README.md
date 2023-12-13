# Welcome to Classically Punk
***

## Task
The task at hand involves developing an application that automatically classifies different musical genres from audio snippets.

 The primary challenge lies in identifying relevant features within the music files for effective classification.

## Description
In this project, we aim to build a Python application that can classify music genres using machine learning techniques. The process involves collecting and cleaning data, exploring its characteristics, visualizing key features, and implementing a machine learning model.

PROJECT STAGES

DATA COLLECTING/CLEANING

Identified and used the librosa library for reading music files in Python.

Collected a diverse dataset of audio snippets, ensuring proper labeling of different music genres.

Cleaned the dataset by addressing missing or inconsistent data, handling outliers, and standardizing file formats.

DATA EXPLORATION
The extracted audio csv feature files codes  are placed in another python file , so we can easily work on our csv file in this project.

Explored metadata from audio files, including duration, sample rate, etc.

Extracted relevant audio features such as MFCC, chroma, and tempo.

DATA VISUALIZATIONS:

Created visualizations to understand the distribution of audio features across different genres.

Explored the correlation between features and visualized the distribution of genres in the dataset.

MACHINE LEARNING

Prepared the data by splitting it into training and testing sets.

Encoded categorical labels into numerical format.

Applied feature scaling to ensure equal contribution from all features.

Explored various machine learning models, including Random Forest, SVM, and Neural Network.

Trained and evaluated models using appropriate metrics like accuracy, precision, recall, and F1-score.

COMMUNICATION

Developed a presentation explaining the approach, assumptions, and implications.

Included visualizations to support findings.

Provided well-documented and modularized code for deployment by the DevOps team.

## Installation
Necessary libraries were installed such as:
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


## Usage
Upon running the application, explore the generated results and visualizations. 

These outputs will provide insights into how the model classifies different music genres based on the identified features.

 Pay attention to accuracy, precision, recall, and F1-score metrics for a comprehensive evaluation.
```
./Classically_Punk argument1 argument2
```

### The Core Team
deniran_o


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>
