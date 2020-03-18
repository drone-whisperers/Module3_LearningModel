# Load libraries...
import os.path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Encoder import Encoder
from TrainingDataGenerator import TrainingDataGenerator
import pickle

#Global Default Settings
USE_SEQUENTIAL_ENCODING = True
EXCLUDE_NUMBERS = True

MY_PATH = os.path.abspath(os.path.dirname(__file__))
TRAINING_DATA = os.path.join(MY_PATH, "TrainingData/generated.training.data.txt")
LABEL_MATRIX_FILE = os.path.join(MY_PATH, "TrainingData/generated.label.matrix.csv")
KNOWN_AIRCRAFT_NAME_FILE = os.path.join(MY_PATH, "TrainingData/generated.aircraft.names.txt")
KNOWN_AIRPORT_ENTITY_FILE = os.path.join(MY_PATH, "TrainingData/generated.airport.entities.txt")
KNOWN_LOCATIONS_FILE = os.path.join(MY_PATH, "TrainingData/generated.known.locations.txt")
ENCODER_FILE = os.path.join(MY_PATH, "Model/encoder.pickle")
SAVE_FILES = {
    "fly": os.path.join(MY_PATH, "Model/fly.command.classifier.pickle"),
    "contact": os.path.join(MY_PATH, "Model/contact.command.classifier.pickle")
}
NEGATIVE_EXAMPLE_PROPORTION = 0.0

class Classifier:
    _encoder = Encoder()
    _classifiers = {}
    _label_matrix = None
    _data_set = None
    _exclude_sets = None

    def __init__(self):
        # If any of the training data is missing then generate new training data
        if not os.path.isfile(os.path.join(MY_PATH,TRAINING_DATA)) or \
                not os.path.isfile(LABEL_MATRIX_FILE) or \
                not os.path.isfile(KNOWN_AIRCRAFT_NAME_FILE) or \
                not os.path.isfile(KNOWN_AIRPORT_ENTITY_FILE) or \
                not os.path.isfile(KNOWN_LOCATIONS_FILE):
            training_data_generator = TrainingDataGenerator()
            training_data_generator.generate(neg_prop=NEGATIVE_EXAMPLE_PROPORTION)

        #Load all training data
        self._data_set = open(os.path.join(MY_PATH, TRAINING_DATA), "r").read().splitlines()
        self._label_matrix = pd.read_csv(os.path.join(MY_PATH, LABEL_MATRIX_FILE))
        self._exclude_sets = {
            "class(aircraft.names)": open(os.path.join(MY_PATH, KNOWN_AIRCRAFT_NAME_FILE),"r").read().splitlines(),
            "class(airport.entities)": open(os.path.join(MY_PATH, KNOWN_AIRPORT_ENTITY_FILE),"r").read().splitlines(),
            "class(known.locations)": open(os.path.join(MY_PATH, KNOWN_LOCATIONS_FILE),"r").read().splitlines()
        }

        #Instantiate and initialize an encoder
        if os.path.isfile(ENCODER_FILE):
            self._encoder = pickle.load(open(ENCODER_FILE, 'rb'))
        else:
            self._encoder.set_exclude_sets(self._exclude_sets)
            self._encoder.set_dataset(self._data_set)
            self._encoder.set_exclude_numbers(EXCLUDE_NUMBERS)
            self._encoder.use_sequential_encoding(USE_SEQUENTIAL_ENCODING)
            self._encoder.init_encoder()
            pickle.dump(self._encoder, open(ENCODER_FILE, 'wb'))

        #Load model if availabe, train otherwise
        for label in self._label_matrix.columns:
            if os.path.isfile(SAVE_FILES[label]):
                model = pickle.load(open(SAVE_FILES[label], 'rb'))
                self._classifiers[label] = model
            else:
                self.__train_classifier()

        return

    def __train_classifier(self):
        # Using the raw dataset, create a bag of words, this is a set that contains all unique words in the data set
        df = self._encoder.create_data_frame()

        # Iterate over each label in the labelMatrix
        # Each iteration will modify the dataframe with the labelled values and train the classifier models
        for label in self._label_matrix.columns:
            print("Training models to classify " + label + " commands.")

            # set index of dataframe to labels, use unique header 'trainingLabel' to avoid collision with the underlying dataset
            df[label] = self._label_matrix[label]
            df.set_index(label, inplace=True)

            # Train logistic regression classifier with full data set
            x_train, x_test, y_train, y_test = train_test_split(df.values, df.index, test_size=0.15, stratify=df.index)
            lr_classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
            lr_classifier.fit(x_train, y_train)
            y_pred = lr_classifier.predict(x_test)
            print ("Logistic Regression Classifier (" + label + ") accuracy score: %2.2f" % (
                    lr_classifier.score(x_test, y_test) * 100))

            #Save classifier for future predictions
            pickle.dump(lr_classifier, open(SAVE_FILES[label], 'wb'))
            self._classifiers[label] = lr_classifier

            # clean up dataframe for next iteration, reset index of data frame and drop the trainingLabel column
            df.reset_index(inplace=True)
            df.drop(columns=[label], inplace=True)

        return


    def classify_command(self, command, print_translation=True):
        classifications = []

        #Iterate through each of the trained classifier models and predict the classification of the command
        for classifier in self._classifiers.keys():
            feature_vector = self._encoder.create_feature_vector(command)

            # Only classify if there is at least 1 word that is recognized in the feature_vector
            if 1 in feature_vector:
                classification = self._classifiers[classifier].predict([feature_vector])
                if (classification[0] == 1):
                    classifications.append(classifier)

        if print_translation:
            print('Input:', command)
            print('Translation:', classifications)

        return classifications
