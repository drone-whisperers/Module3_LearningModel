import pandas as pd

ONE_HOT_ENCODE_NUMBERS_KEY = "class(numbers)"
WILDCARD_CHARACTER = "*"

class Encoder:
    _use_sequential_encoding = False
    _exclude_sets = {}
    _dataset = list()
    _word_bag = set()
    _max_sequence_length = 0
    _init = False
    _exclude_set_occurrence_map = {}
    _feature_vector_columns = list()

    def set_exclude_sets(self, exclude_sets):
        self._exclude_sets = exclude_sets

    def set_exclude_numbers(self, do_exclude_sets):
        if do_exclude_sets:
            self._exclude_sets[ONE_HOT_ENCODE_NUMBERS_KEY] = []
        else:
            self._exclude_sets.pop(ONE_HOT_ENCODE_NUMBERS_KEY, None)

    def use_sequential_encoding(self, use_sequential_encoding):
        self._use_sequential_encoding = use_sequential_encoding

    def set_dataset(self, dataset):
        self._dataset = dataset

    # Initializes Encoder for use.
    # Creates a bag of words (set).
    # Expects the dataset to be a list of sentences.
    # If exclude_numbers is enabled, will not add any numbers to the bagOfWords,
    # will return the max number of instances of a number (for a single example in the data set).
    # If an excludeSet is provided, any items in this list will not be included in the bagOfWords
    def __init_encoder(self):
        word_bag = set()
        exclude_set_occurence_map = {}    # This is used to determine the maximum # of occurrences in a single example for each excludeSetName
        max_sequence_length = 0           # This is used to determine the maximum sequence length in the data set (This is not the same as word count)

        # Instantiate exclude_set_occurence_map based on exclude_sets
        for excludeSetName in self._exclude_sets:
            exclude_set_occurence_map[excludeSetName] = 0

        for example in self._dataset:
            numeric_occurrences = 0
            sequence_length = 0

            # Remove any words in the exclude_sets from inclusion in the bag of words
            for excludeSetName in self._exclude_sets:
                exclude_set_occurrences = 0
                for excluded in self._exclude_sets[excludeSetName]:
                    if excluded in example:
                        exclude_set_occurrences += 1
                        sequence_length += example.count(excluded)           # Increment sequence_length by the number of occurrences of the excluded value in the sentence
                        example = example.replace(excluded, "")

                # Compare number of occurrences of excluded_set to current max
                exclude_set_occurence_map[excludeSetName] = max(exclude_set_occurrences, exclude_set_occurence_map[excludeSetName])

            # Iterate through each word in the example, add each word to the bag
            for word in example.split():
                sequence_length += 1
                # Determine the maximum number of instances of a number (for a single example in the data set)
                if ONE_HOT_ENCODE_NUMBERS_KEY in self._exclude_sets.keys() and self.__isnumber(word):
                    numeric_occurrences += 1
                    continue
                if word == WILDCARD_CHARACTER:
                    continue
                word_bag.add(word)

            max_sequence_length = max(sequence_length, max_sequence_length)

            # Compare number of occurrences of numbers to current max
            if ONE_HOT_ENCODE_NUMBERS_KEY in self._exclude_sets.keys():
                exclude_set_occurence_map[ONE_HOT_ENCODE_NUMBERS_KEY] = max(numeric_occurrences, exclude_set_occurence_map[ONE_HOT_ENCODE_NUMBERS_KEY])

        self._word_bag = word_bag
        self._max_sequence_length = max_sequence_length
        self._init = True
        self._exclude_set_occurrence_map = exclude_set_occurence_map

        # Create a list from the word bag to be used as the columns for the data frame.
        # The following steps will be modifying this list, as such it is import to retain the current ordering
        # as the feature vector encoding was created using this order
        self._feature_vector_columns = list(word_bag)
        [self._feature_vector_columns.append(i) for i in self._exclude_sets.keys()]
        return True


    # A simple method to determine whether a string contains a valid number
    def __isnumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    # A simple method to encode a binary array. The array is encoded with all 0's except for the index values in indices
    # which are encoded as 1's.
    def __encode(self, sequence_encoding_length, indices=[]):
        encoding = [0] * sequence_encoding_length

        for i in indices:
            encoding[i] = 1

        return encoding

    # Searches a string sentence to find all instances of a word within it.
    # Returns a list containing each index for matches.
    def __getMatchIndices(self, target, sentence, word_bag):
        indices = []
        i = 0

        for word in sentence.split():
            if word in self._feature_vector_columns:
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
    def __encode_sequential(self, sentence):
        encoding_map = {}

        # "Clean" the sentence of any of the classes in exclude sets, replace these occurrences with the common
        # class entity name (format is -> class(<className>))
        if self._exclude_sets:
            for exclude_set_name in self._exclude_sets:
                if exclude_set_name == ONE_HOT_ENCODE_NUMBERS_KEY:
                    for word in sentence.split():
                        if self.__isnumber(word):
                            sentence = sentence.replace(word, exclude_set_name)
                else:
                    for excluded in self._exclude_sets[exclude_set_name]:
                        if excluded in sentence:
                            sentence = sentence.replace(excluded.strip(), exclude_set_name)

        # add exclude_set encoding to the encoding_map
        if self._exclude_sets:
            for exclude_set_name in self._exclude_sets:
                encoding_map[exclude_set_name] = self.__encode(self._max_sequence_length, self.__getMatchIndices(exclude_set_name, sentence, self._word_bag))

        # Add encoding of each word to the encoding_map
        for word in self._word_bag:
            if word in sentence.split():
                encoding_map[word] = self.__encode(self._max_sequence_length, self.__getMatchIndices(word, sentence, self._word_bag))
            else:
                encoding_map[word] = self.__encode(self._max_sequence_length)

        return encoding_map

    # Encode sentence relative to the words in the word_bag.
    # If exclude_sets are provided, then any item contained within an exclude set
    # from exclude_sets will be encode by its class. However, if
    # exclude_set_encoding_length is provided then each exclude_set with a defined
    # encoding length will be one hot encoded, by occurrence.
    # (Ex. if the length is 5 and the count of numbers in a sentence is:
    #     1 - the one hot encoding would be 1 0 0 0 0
    #     2 - the one hot encoding would be 0 1 0 0 0
    # ... etc.)
    def __encode_occurrence(self, sentence):
        encoding_map = {}
        exclude_set_occurrences = {}

        # Instantiate exclude_set_occurrences and encoding_map based on exclude_sets
        for excludeSetName in self._exclude_sets:
            exclude_set_occurrences[excludeSetName] = 0
            encoding_map[excludeSetName] = []

        # Ensure items in the excludeSets are not encoded in featureVector
        if self._exclude_sets:
            for excludeSetName in self._exclude_sets:
                # Remove any words in the excludeSetName from the sentence
                if excludeSetName == ONE_HOT_ENCODE_NUMBERS_KEY:
                    for word in sentence.split():
                        if self.__isnumber(word):
                            exclude_set_occurrences[excludeSetName] += 1
                            sentence = sentence.replace(word, "")
                else:
                    for excluded in self._exclude_sets[excludeSetName]:
                        if excluded in sentence:
                            exclude_set_occurrences[excludeSetName] += 1
                            sentence = sentence.replace(excluded, "")

        # Iterate through each word in the bag, encode 1 if this word is present in the sentence, 0 otherwise
        for word in self._word_bag:
            if word in sentence.split():
                encoding_map[word] = [1]
            else:
                encoding_map[word] = [0]

        # Add one hot encoding of exclude_sets
        if self._exclude_sets:
            for excludeSetName in self._exclude_sets:
                if self._exclude_set_occurrence_map[excludeSetName]:
                    for i in range(1, self._exclude_set_occurrence_map[excludeSetName] + 1):
                        if i == exclude_set_occurrences[excludeSetName]:
                            encoding_map[excludeSetName].append(1)
                        else:
                            encoding_map[excludeSetName].append(0)

        return encoding_map

    def create_data_frame(self):
        feature_vectors = []
        if not self._init:
            self.__init_encoder()

        # Create a feature vector for each example in the data set
        for example in self._dataset:
            feature_vectors.append(self.create_feature_vector(example))

        # Create appropriate column names
        dataframe_columns = list()
        for feature_vector_column in self._feature_vector_columns:
            if self._use_sequential_encoding:
                for i in range(0, self._max_sequence_length):
                    dataframe_columns.append(feature_vector_column + "[{}]".format(i))
            else:
                dataframe_columns.append(feature_vector_column)

        return pd.DataFrame(data=feature_vectors, columns=dataframe_columns)

    def create_feature_vector(self, sentence):
        if not self._init:
            self.__init_encoder()

        if self._use_sequential_encoding:
            encoding_map = self.__encode_sequential(sentence)
        else:
            encoding_map = self.__encode_occurrence(sentence)

        feature_vector = []

        for word in self._feature_vector_columns:
            for value in encoding_map[word]:
                feature_vector.append(value)

        return feature_vector
