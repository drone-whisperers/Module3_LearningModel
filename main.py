# Load libraries...
from urllib import urlopen
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Encoder import Encoder

DATASET_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/dclaremont/useSequentialEncoding/TrainingData/data.txt'
KNOWN_ENTITIES_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/knownentities.txt'
KNOWN_LOCATIONS_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/knownlocations.txt'
LABEL_MATRIX_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/dclaremont/useSequentialEncoding/TrainingData/labelMatrix.csv'
DATA_SET = urlopen(DATASET_URL).read().splitlines()
LABEL_MATRIX = pd.read_csv(LABEL_MATRIX_URL)
USE_SEQUENTIAL_ENCODING = True
EXCLUDE_NUMBERS = True
EXCLUDE_SETS = {
    "class(knownEntities)": urlopen(KNOWN_ENTITIES_URL).read().splitlines(),
    "class(knownLocations)": urlopen(KNOWN_LOCATIONS_URL).read().splitlines()
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
    print ("Training models to classify " + label + " commands.")

    # set index of dataframe to labels, use unique header 'trainingLabel' to avoid collision with the underlying dataset
    df[label] = LABEL_MATRIX[label]
    df.set_index(label, inplace=True)

    # Train logistic regression classifier with full data set
    x_train, x_test, y_train, y_test = train_test_split(df.values, df.index, test_size=0.25, stratify=df.index)
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
    lr_classifier.fit(x_train, y_train)
    y_pred = lr_classifier.predict(x_test)
    print ("Logistic Regression Classifier (" + label + ") accuracy score: %2.2f" % (
            lr_classifier.score(x_test, y_test) * 100))

    # clean up dataframe for next iteration, reset index of data frame and drop the trainingLabel column
    df.reset_index(inplace=True)
    df.drop(columns=[label], inplace=True)

