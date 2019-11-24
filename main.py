# Load libraries...
from urllib import urlopen
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Global Parameters
DATASET_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/data.txt'
KNOWN_ENTITIES_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/knownentities.txt'
KNOWN_LOCATIONS_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/knownlocations.txt'
LABEL_MATRIX_URL = 'https://raw.githubusercontent.com/drone-whisperers/Module3_LearningModel/master/TrainingData/labelMatrix.csv'
DATA_SET = urlopen(DATASET_URL).read().splitlines()
USE_SEQUENTIAL_ENCODING = True
ONE_HOT_ENCODE_NUMBERS_KEY = "one_hot_encode(numbers) key"
ONE_HOT_ENCODE_NUMBERS = True
USE_EXCLUDE_SETS = True
EXCLUDE_SETS = {
    "knownEntities": urlopen(KNOWN_ENTITIES_URL).read().splitlines(),
    "knownLocations": urlopen(KNOWN_LOCATIONS_URL).read().splitlines()
}
LABEL_MATRIX = pd.read_csv(LABEL_MATRIX_URL)

# Creates a bag of words (set).
# Expects the dataset to be a list of sentences.
# If exclude_numbers is enabled, will not add any numbers to the bagOfWords,
# will return the max number of instances of a number (for a single example in the data set).
# If an excludeSet is provided, any items in this list will not be included in the bagOfWords
def create_word_bag(data_set, exclude_sets={}, exclude_numbers=False):
    wordBag = set()
    excludeSetEncodingLegend = {}
    maxSequenceLength = 0           # This is used to determine the sequence length necessary to encode each example in the data set

    # Instantiate excludeSetEncodingLegend based on exclude_sets
    for exclude_set in exclude_sets:
        excludeSetEncodingLegend[exclude_set] = 0
    if exclude_numbers:
        excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY] = 0

    for example in data_set:
        numericOccurrences = 0
        sequenceLength = 0

        # Remove any words in the exclude_sets from inclusion in the bag of words
        for exclude_set in exclude_sets:
            excludeSetOccurrences = 0
            for excluded in exclude_sets[exclude_set]:
                if excluded in example:
                    excludeSetOccurrences += 1
                    sequenceLength += example.count(excluded)           # Increment sequenceLength by the number of occurrences of the excluded value in the sentence
                    example = example.replace(excluded, "")

            # Compare number of occurrences of excluded_set to current max
            excludeSetEncodingLegend[exclude_set] = max(excludeSetOccurrences,
                                                            excludeSetEncodingLegend[exclude_set])

        # Iterate through each word in the example, add each word to the bag
        for word in example.split():
            sequenceLength += 1
            # Determine the maximum number of instances of a number (for a single example in the data set)
            if exclude_numbers and isnumber(word):
                numericOccurrences += 1
                continue

            wordBag.add(word)

        maxSequenceLength = max(sequenceLength, maxSequenceLength)

        # Compare number of occurrences of numbers to current max
        if exclude_numbers:
            excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY] = max(numericOccurrences, excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY])

    return wordBag, excludeSetEncodingLegend, maxSequenceLength


def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def encode_sequential(word_bag, sentence, sequence_encoding_length, exclude_sets={}):
    featureVector = []

    # In string replacement of any excluded classes of entities from exclude_set with the class name of the entity
    if exclude_sets:
        for exclude_set in exclude_sets:
            # Replace any classes from this exclude_set from the sentence
            for excluded in exclude_sets[exclude_set]:
                if excluded in sentence:
                    # strip any
                    sentence = sentence.replace(excluded.strip(), exclude_set)

    # Iterate through each word in the bag, encode 1 if this word is present in the sentence, 0 otherwise
    for word in word_bag:
        if word in sentence:
            count = sentence.count(word)
            featureVector.append(1)
        else:
            for i in range(0, sequence_encoding_length):
                featureVector.append(0)

    print ("hello world")

# Create a feature vector for a sentence, based on the bag of words.
# If exclude_sets are provided, then any word contained within an exclude set
# from exclude_sets will not be used to encode the feature vector. However, if
# exclude_set_encoding_length is provided then each exclude_set with a defined
# encoding length will be one hot encoded.
# If exclude_set_encoding_length an entry for "numbers", all numbers will be one hot encoded.
# (Ex. if the length is 5 and the count of numbers in a sentence is:
#     1 - the one hot encoding would be 1 0 0 0 0
#     2 - the one hot encoding would be 0 1 0 0 0
# ... etc.)
def create_feature_vector(word_bag, sentence, exclude_sets={}, exclude_set_encoding_length={}, sequence_encoding_length=0):
    featureVector = []
    excludeSetOccurrences = {}

    if sequence_encoding_length:
        encode_sequential(word_bag, sentence, sequence_encoding_length, exclude_sets)

    # Instantiate excludeSetOccurrences based on exclude_sets
    for exclude_set in exclude_sets:
        excludeSetOccurrences[exclude_set] = 0

    # Ensure items in the excludeSets are not encoded in featureVector
    if exclude_sets:
        for exclude_set in exclude_sets:
            # Remove any words in the exclude_set from the sentence
            for excluded in exclude_sets[exclude_set]:
                if excluded in sentence:
                    excludeSetOccurrences[exclude_set] += 1
                    sentence = sentence.replace(excluded, "")

    # Iterate through each word in the bag, encode 1 if this word is present in the sentence, 0 otherwise
    for word in word_bag:
        if word in sentence:
            featureVector.append(1)
        else:
            featureVector.append(0)

    # Add one hot encoding of exclude_sets
    if exclude_sets:
        for exclude_set in exclude_sets:
            if exclude_set_encoding_length[exclude_set]:
                for i in range(1, exclude_set_encoding_length[exclude_set] + 1):
                    if i == excludeSetOccurrences[exclude_set]:
                        featureVector.append(1)
                    else:
                        featureVector.append(0)

    # Add one hot encoding of numbers
    if exclude_set_encoding_length.has_key(ONE_HOT_ENCODE_NUMBERS_KEY):
        numericOccurrences = 0
        # Determine how many occurrences of numbers are in this sentence
        for word in sentence.split():
            if isnumber(word):
                numericOccurrences += 1

        # Encode the number of occurrences of numbers using one hot encoding
        for i in range(1, exclude_set_encoding_length[ONE_HOT_ENCODE_NUMBERS_KEY] + 1):
            if i == numericOccurrences:
                featureVector.append(1)
            else:
                featureVector.append(0)

    return featureVector


def create_data_frame(dataset, exclude_sets={}, exclude_numbers=False, sequential_encoding=False):
    featureVectors = []
    wordBag, excludeSetEncodingLegend, maxSequenceLength = create_word_bag(data_set=dataset,
                                                                           exclude_sets=exclude_sets,
                                                                           exclude_numbers=exclude_numbers)

    # Create a feature vector for each example in the data set
    for example in dataset:
        featureVectors.append(
            create_feature_vector(wordBag, example, exclude_sets, excludeSetEncodingLegend, maxSequenceLength))

    # Create a list from the word bag to be used as the columns for the data frame.
    # The following steps will be modifying this list, as such it is import to retain the current ordering
    # as the feature vector encoding was created using this order
    columns = list(wordBag)

    # Add one hot encoding of exclude_sets
    if exclude_sets:
        for exclude_set in exclude_sets:
            if excludeSetEncodingLegend[exclude_set]:
                for i in range(1, excludeSetEncodingLegend[exclude_set] + 1):
                    columns.append("one_hot_encode(" + exclude_set + ")[{}]".format(i))

    # Add columns for one hot encoded numbers
    if exclude_numbers:
        for i in range(1, excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY] + 1):
            columns.append("one_hot_encode(numbers)[{}]".format(i))

    return pd.DataFrame(data=featureVectors, columns=columns), wordBag, excludeSetEncodingLegend


# Using the raw dataset, create a bag of words, this is a set that contains all unique words in the data set
if USE_EXCLUDE_SETS:
    df, word_bag, exclude_set_encoding_legend = create_data_frame(dataset=DATA_SET,
                           exclude_sets=EXCLUDE_SETS,
                           exclude_numbers=ONE_HOT_ENCODE_NUMBERS)
else:
    df, word_bag, exclude_set_encoding_legend = create_data_frame(dataset=DATA_SET,
                           exclude_numbers=ONE_HOT_ENCODE_NUMBERS)


# Iterate over each label in the labelMatrix
# Each iteration will modify the dataframe with the labelled values and train the classifier models
for label in LABEL_MATRIX.columns:
    print ("Training models to classify " + label + " commands.")

    # set index of dataframe to labels, use unique header 'trainingLabel' to avoid collision with the underlying dataset
    df['trainingLabel'] = LABEL_MATRIX[label]
    df.set_index('trainingLabel', inplace=True)

    # Train logistic regression classifier with full data set
    x_train, x_test, y_train, y_test = train_test_split(df.values, df.index, test_size=0.25, stratify=df.index)
    lr_classifier = LogisticRegression(solver='lbfgs', max_iter=5000)
    lr_classifier.fit(x_train, y_train)
    y_pred = lr_classifier.predict(x_test)
    print ("Logistic Regression Classifier (" + label + ") accuracy score: %2.2f" % (
            lr_classifier.score(x_test, y_test) * 100))

    # clean up dataframe for next iteration, reset index of data frame and drop the trainingLabel column
    df.reset_index(inplace=True)
    df.drop(columns=['trainingLabel'], inplace=True)
