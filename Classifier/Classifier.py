# Load libraries...
import os.path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Encoder import Encoder

#Global Default Settings
USE_SEQUENTIAL_ENCODING = True
EXCLUDE_NUMBERS = True

#Gather training data
my_path = os.path.abspath(os.path.dirname(__file__))
DATA_SET = open(os.path.join(my_path, "./TrainingData/data.txt"), "r").read().splitlines()
LABEL_MATRIX = pd.read_csv(os.path.join(my_path, "./TrainingData/labelMatrix.csv"))
EXCLUDE_SETS = {
    "class(knownEntities)":  open(os.path.join(my_path, "./TrainingData/knownEntities.txt"), "r").read().splitlines(),
    "class(knownLocations)": open(os.path.join(my_path, "./TrainingData/knownLocations.txt"), "r").read().splitlines()
}

#Instantiate and initialize an encoder
encoder = Encoder()
encoder.set_exclude_sets(EXCLUDE_SETS)
encoder.set_dataset(DATA_SET)
if EXCLUDE_NUMBERS:
    encoder.set_exclude_numbers(True)
if USE_SEQUENTIAL_ENCODING:
    encoder.use_sequential_encoding(USE_SEQUENTIAL_ENCODING)

# Using the raw dataset, create a bag of words, this is a set that contains all unique words in the data set
df = encoder.create_data_frame()

# Iterate over each label in the labelMatrix
# Each iteration will modify the dataframe with the labelled values and train the classifier models
#for label in LABEL_MATRIX.columns:
for label in ['taxi', 'contact']:
    print("Training models to classify " + label + " commands.")

    # set index of dataframe to labels, use unique header 'trainingLabel' to avoid collision with the underlying dataset
    df[label] = LABEL_MATRIX[label]
    df.set_index(label, inplace=True)

    # Train logistic regression classifier with full data set
    x_train, x_test, y_train, y_test = train_test_split(df.values, df.index, test_size=0.05, stratify=df.index)
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
    lr_classifier.fit(x_train, y_train)
    y_pred = lr_classifier.predict(x_test)
    print ("Logistic Regression Classifier (" + label + ") accuracy score: %2.2f" % (
            lr_classifier.score(x_test, y_test) * 100))

    lr_classifier.predict(
        [encoder.create_feature_vector("Big Jet 345 taxi to holding point C1 runway 99 Metro Ground")])
    # clean up dataframe for next iteration, reset index of data frame and drop the trainingLabel column
    df.reset_index(inplace=True)
    df.drop(columns=[label], inplace=True)

