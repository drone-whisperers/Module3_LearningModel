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
LABEL_MATRIX = pd.read_csv(LABEL_MATRIX_URL)
USE_SEQUENTIAL_ENCODING = True
ONE_HOT_ENCODE_NUMBERS_KEY = "class(numbers)"
EXCLUDE_NUMBERS = True
USE_EXCLUDE_SETS = True
EXCLUDE_SETS = {
    "class(knownEntities)": urlopen(KNOWN_ENTITIES_URL).read().splitlines(),
    "class(knownLocations)": urlopen(KNOWN_LOCATIONS_URL).read().splitlines()
}
if EXCLUDE_NUMBERS:
    EXCLUDE_SETS[ONE_HOT_ENCODE_NUMBERS_KEY] = []



# Creates a bag of words (set).
# Expects the dataset to be a list of sentences.
# If exclude_numbers is enabled, will not add any numbers to the bagOfWords,
# will return the max number of instances of a number (for a single example in the data set).
# If an excludeSet is provided, any items in this list will not be included in the bagOfWords
def create_word_bag(data_set, exclude_sets={}):
    wordBag = set()
    excludeSetEncodingLegend = {}   # This is used to determine the maximum # of occurrences in a single example for each excludeSetName
    maxSequenceLength = 0           # This is used to determine the maximum sequence length in the data set (This is not the same as word count)

    # Instantiate excludeSetEncodingLegend based on exclude_sets
    for excludeSetName in exclude_sets:
        excludeSetEncodingLegend[excludeSetName] = 0

    for example in data_set:
        numericOccurrences = 0
        sequenceLength = 0

        # Remove any words in the exclude_sets from inclusion in the bag of words
        for excludeSetName in exclude_sets:
            excludeSetOccurrences = 0
            for excluded in exclude_sets[excludeSetName]:
                if excluded in example:
                    excludeSetOccurrences += 1
                    sequenceLength += example.count(excluded)           # Increment sequenceLength by the number of occurrences of the excluded value in the sentence
                    example = example.replace(excluded, "")

            # Compare number of occurrences of excluded_set to current max
            excludeSetEncodingLegend[excludeSetName] = max(excludeSetOccurrences, excludeSetEncodingLegend[excludeSetName])

        # Iterate through each word in the example, add each word to the bag
        for word in example.split():
            sequenceLength += 1
            # Determine the maximum number of instances of a number (for a single example in the data set)
            if exclude_sets.has_key(ONE_HOT_ENCODE_NUMBERS_KEY) and isnumber(word):
                numericOccurrences += 1
                continue

            wordBag.add(word)

        maxSequenceLength = max(sequenceLength, maxSequenceLength)

        # Compare number of occurrences of numbers to current max
        if exclude_sets.has_key(ONE_HOT_ENCODE_NUMBERS_KEY):
            excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY] = max(numericOccurrences, excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY])

    return wordBag, excludeSetEncodingLegend, maxSequenceLength


# A simple method to determine whether a string contains a valid number
def isnumber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# A simple method to encode a binary array. The array is encoded with all 0's except for the index values in indices
# which are encoded as 1's.
def encode(sequence_encoding_length, indices=[]):
    encoding = [0] * sequence_encoding_length
    
    for i in indices:
        encoding[i] = 1
    
    return encoding


# Searches a string sentence to find all instances of a word within it.
# Returns a list containing each index for matches.
def getMatchIndices(target, sentence):
    indices = []
    i = 0
    
    for word in sentence.split():
        if word == target:
            indices.append(i)
        i += 1
        
    return indices


# Encode sentence relative to the words in the word_bag while preserving sequence information.
# Each word that is present in both the sentence and the word_bag will be binary encoded in a
# sequence that is sequence_encoding_length in size.
#   Ex. If the sentence is "This is a test sentence, test number 1 of 20"
#       The encoding for test would be: 0 0 0 1 0 1 0 0 0 0.
#       The encoding for class(numbers) would be: 0 0 0 0 0 0 0 1 0 1
# A map containing the encoding for each word in the sentence, exclude_class, and class(number) is returned.
def encode_sequential(word_bag, sentence, sequence_encoding_length, exclude_sets={}):
    encodingMap = {}

    # "Clean" the sentence of any of the classes in exclude sets, replace these occurrences with the common class entity name (format is -> class(<className>))
    if exclude_sets:
        for excludeSetName in exclude_sets:
            if excludeSetName == ONE_HOT_ENCODE_NUMBERS_KEY:
                for word in sentence.split():
                    if isnumber(word):
                        sentence = sentence.replace(word, excludeSetName)
            else:
                for excluded in exclude_sets[excludeSetName]:
                    if excluded in sentence:
                        sentence = sentence.replace(excluded.strip(), excludeSetName)

    # add exclude_set encoding to the encodingMap
    if exclude_sets:
        for excludeSetName in exclude_sets:
            encodingMap[excludeSetName] = encode(sequence_encoding_length, getMatchIndices(excludeSetName, sentence))

    # Add encoding of each word to the encodingMap
    for word in word_bag:
        if word in sentence:
            encodingMap[word] = encode(sequence_encoding_length, getMatchIndices(word, sentence))
        else:
            encodingMap[word] = encode(sequence_encoding_length)

    return encodingMap


# Encode sentence relative to the words in the word_bag.
# If exclude_sets are provided, then any item contained within an exclude set
# from exclude_sets will be encode by its class. However, if
# exclude_set_encoding_length is provided then each exclude_set with a defined
# encoding length will be one hot encoded, by occurrence.
# (Ex. if the length is 5 and the count of numbers in a sentence is:
#     1 - the one hot encoding would be 1 0 0 0 0
#     2 - the one hot encoding would be 0 1 0 0 0
# ... etc.)
def encode_occurrence(word_bag, sentence, exclude_sets={}, exclude_set_encoding_length={}):
    encodingMap = {}
    excludeSetOccurrences = {}

    # Instantiate excludeSetOccurrences and encodingMap based on exclude_sets
    for excludeSetName in exclude_sets:
        excludeSetOccurrences[excludeSetName] = 0
        encodingMap[excludeSetName] = []

    # Ensure items in the excludeSets are not encoded in featureVector
    if exclude_sets:
        for excludeSetName in exclude_sets:
            # Remove any words in the excludeSetName from the sentence
            if excludeSetName == ONE_HOT_ENCODE_NUMBERS_KEY:
                for word in sentence.split():
                    if isnumber(word):
                        excludeSetOccurrences[excludeSetName] += 1
                        sentence = sentence.replace(word, "")
            else:
                for excluded in exclude_sets[excludeSetName]:
                    if excluded in sentence:
                        excludeSetOccurrences[excludeSetName] += 1
                        sentence = sentence.replace(excluded, "")

    # Iterate through each word in the bag, encode 1 if this word is present in the sentence, 0 otherwise
    for word in word_bag:
        if word in sentence:
            encodingMap[word] = [1]
        else:
            encodingMap[word] = [0]

    # Add one hot encoding of exclude_sets
    if exclude_sets:
        for excludeSetName in exclude_sets:
            if exclude_set_encoding_length[excludeSetName]:
                for i in range(1, exclude_set_encoding_length[excludeSetName] + 1):
                    if i == excludeSetOccurrences[excludeSetName]:
                        encodingMap[excludeSetName].append(1)
                    else:
                        encodingMap[excludeSetName].append(0)

    return encodingMap



def create_feature_vector(encodingMap, columns):
    featureVector = []

    for word in columns:
        for value in encodingMap[word]:
            featureVector.append(value)

    return featureVector


def create_data_frame(dataset, exclude_sets={}, sequential_encoding=False):
    featureVectors = []
    wordBag, excludeSetEncodingLegend, maxSequenceLength = create_word_bag(data_set=dataset, exclude_sets=exclude_sets)

    # Create a list from the word bag to be used as the columns for the data frame.
    # The following steps will be modifying this list, as such it is import to retain the current ordering
    # as the feature vector encoding was created using this order
    columns = list(wordBag)
    [columns.append(i) for i in exclude_sets.keys()]

    # Create a feature vector for each example in the data set
    for example in dataset:
        if sequential_encoding:
            encodingMap = encode_sequential(wordBag, example, maxSequenceLength, exclude_sets)
        else:
            encodingMap = encode_occurrence(wordBag, example, exclude_sets, excludeSetEncodingLegend)
        featureVectors.append(create_feature_vector(encodingMap, columns))

    #TODO Create appropriate column names
    #columns = list(wordBag)

    # Add one hot encoding of exclude_sets
    #if exclude_sets:
    #    for exclude_set in exclude_sets:
    #        if excludeSetEncodingLegend[exclude_set]:
    #            for i in range(1, excludeSetEncodingLegend[exclude_set] + 1):
    #                columns.append("one_hot_encode(" + exclude_set + ")[{}]".format(i))

    # Add columns for one hot encoded numbers
    #if exclude_numbers:
    #    for i in range(1, excludeSetEncodingLegend[ONE_HOT_ENCODE_NUMBERS_KEY] + 1):
    #        columns.append("one_hot_encode(numbers)[{}]".format(i))

    return pd.DataFrame(data=featureVectors), wordBag, columns, excludeSetEncodingLegend


# Using the raw dataset, create a bag of words, this is a set that contains all unique words in the data set
if USE_EXCLUDE_SETS:
    df, word_bag, columns, excludeSetEncodingLegend = create_data_frame(dataset=DATA_SET,
                                                                  exclude_sets=EXCLUDE_SETS,
                                                                  sequential_encoding=USE_SEQUENTIAL_ENCODING)
else:
    df, word_bag, columns, excludeSetEncodingLegend = create_data_frame(dataset=DATA_SET)









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
