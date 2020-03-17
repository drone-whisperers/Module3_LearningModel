import os
import numpy as np
from numpy import asarray, zeros
from keras.models import Model, load_model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from TrainingDataGenerator import TrainingDataGenerator
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 64
EPOCHS = 15
LSTM_NODES =256
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 2000
EMBEDDING_SIZE = 100
DROPOUT = 0.3
NEGATIVE_EXAMPLE_PROPORTION=0.15
GLOVE_FILE = "./TrainingData/glove.6B.100d.txt"
TRAINING_DATA = "./TrainingData/generated.training.data.txt"
ENCODER_MODEL_SAVE_FILE = "./Model/encoder.model"
DECODER_MODEL_SAVE_FILE = "./Model/decoder.model"
BEST_MODEL_WEIGHTS_FILE = "./Model/best.weights.val.acc.h5"
MY_PATH = os.path.abspath(os.path.dirname(__file__))

class Translater:
    _init = False
    _max_input_len = 0
    _max_output_len = 0
    _num_input_words = 0
    _num_output_words = 0
    _encoder_model = None
    _decoder_model = None
    _idx2word_input = None
    _idx2word_target = None
    _word2idx_input = None
    _word2idx_output = None
    _input_tokenizer = None

    # Initialize Translater. Loads the already trained model and recreates the Translater's state using the trained data.
    # Only useful if these are available in the expected locations.
    def init_translater(self):
        if os.path.isfile(ENCODER_MODEL_SAVE_FILE) and os.path.isfile(DECODER_MODEL_SAVE_FILE):
            if not os.path.isfile(TRAINING_DATA):
                training_data_generator = TrainingDataGenerator()
                training_data_generator.generate(neg_prop=NEGATIVE_EXAMPLE_PROPORTION)

            #Load state of Translater
            input_sentences, output_sentences, output_sentences_inputs = self.__process_training_data()
            encoder_input_sequences, self._word2idx_input, self._input_tokenizer = self.__tokenize_sentences(input_sentences, input_sentences, None)
            decoder_output_sequences, self._word2idx_output, _ = self.__tokenize_sentences(output_sentences + output_sentences_inputs, output_sentences, '')
            decoder_input_sequences, _, _ = self.__tokenize_sentences(output_sentences + output_sentences_inputs,output_sentences_inputs, '')

            self._num_input_words = len(self._word2idx_input) + 1
            self._num_output_words = len(self._word2idx_output) + 1
            self._max_input_len = max(len(sen) for sen in encoder_input_sequences)
            self._max_output_len = max(len(sen) for sen in decoder_output_sequences)

            self._idx2word_input = {v: k for k, v in self._word2idx_input.items()}
            self._idx2word_target = {v: k for k, v in self._word2idx_output.items()}

            #Load encoder and decoder models
            self._encoder_model = load_model(ENCODER_MODEL_SAVE_FILE, compile=False)
            self._decoder_model = load_model(DECODER_MODEL_SAVE_FILE, compile=False)

            self._init = True

    # Default initialization method. Generates training data, if it is not in the expected location. Then creates a model
    # and trains it. Saves model for future use.
    def __init_translater(self):
        if not os.path.isfile(TRAINING_DATA):
            training_data_generator = TrainingDataGenerator()
            training_data_generator.generate(neg_prop=NEGATIVE_EXAMPLE_PROPORTION)

        input_sentences, output_sentences, output_sentences_inputs = self.__process_training_data()
        encoder_input_sequences, self._word2idx_input, self._input_tokenizer = self.__tokenize_sentences(input_sentences, input_sentences, None)
        decoder_output_sequences, self._word2idx_output, _ = self.__tokenize_sentences(output_sentences + output_sentences_inputs, output_sentences, '')
        decoder_input_sequences, _, _ = self.__tokenize_sentences(output_sentences + output_sentences_inputs, output_sentences_inputs, '')

        #Pad the input and output sequences
        padded_encoder_input_sequences = self.__pad_sequences(encoder_input_sequences, 'pre')
        padded_decoder_output_sequences = self.__pad_sequences(decoder_output_sequences, 'post')
        padded_decoder_input_sequences = self.__pad_sequences(decoder_input_sequences, 'post')

        #Create an embedding matrix for all words in input
        input_embedding_matrix = self.__create_word_embedding_matrix(self._word2idx_input)

        self._num_input_words = len(self._word2idx_input) + 1
        self._num_output_words = len(self._word2idx_output) + 1
        self._max_input_len = max(len(sen) for sen in encoder_input_sequences)
        self._max_output_len = max(len(sen) for sen in decoder_output_sequences)
        embedding_layer = Embedding(self._num_input_words, EMBEDDING_SIZE, weights=[input_embedding_matrix], input_length=self._max_input_len)

        decoder_targets_one_hot = self.__create_output_model(padded_decoder_output_sequences)
        encoder, encoder_outputs, encoder_states, encoder_inputs_placeholder = self.__define_encoder(embedding_layer)
        decoder_lstm, decoder_outputs, decoder_embedding, decoder_inputs_placeholder = self.__define_decoder(encoder_states)
        decoder_dense, decoder_outputs = self.__define_dense_layer(decoder_outputs)

        training_model = self.__compile_and_fit_model(
            encoder_inputs_placeholder,
            decoder_inputs_placeholder,
            decoder_outputs,
            padded_encoder_input_sequences,
            padded_decoder_input_sequences,
            decoder_targets_one_hot
        )

        self._encoder_model, self._decoder_model = self.__modify_model_for_predictions(encoder_inputs_placeholder, encoder_states, decoder_embedding, decoder_lstm, decoder_dense)

        self._encoder_model.save(ENCODER_MODEL_SAVE_FILE)
        self._decoder_model.save(DECODER_MODEL_SAVE_FILE)

        #todo: investigate what these contain, and then rename?
        self._idx2word_input = {v:k for k, v in self._word2idx_input.items()}
        self._idx2word_target = {v:k for k, v in self._word2idx_output.items()}

        self._init = True

    # Initial processing for training data. Reads all training data and splits each example into 3 parts.
    #
    # Ex.
    #   'Cessna 212 contact Phoenix Tower 133.0  	 contact 133.0'
    #       input_sentence - "Cessna 212 contact Phoenix Tower 133.0"
    #       output_sentence - "contact 133.0 <eos>"
    #       output_sentence_input - "<sos>contact 133.0"
    def __process_training_data(self):
        input_sentences = []
        output_sentences = []
        output_sentences_inputs = []
        count = 0

        for line in open(os.path.join(MY_PATH, TRAINING_DATA), "r").read().splitlines():
            count += 1

            if '\t' not in line:
                continue

            input_sentence, output = line.rstrip().split('\t')

            #Add eos(end of sentence) and sos(start of sentence) tags
            output_sentence = output + ' <eos>'
            output_sentence_input = '<sos> ' + output

            input_sentences.append(input_sentence)
            output_sentences.append(output_sentence)
            output_sentences_inputs.append(output_sentence_input)

        return input_sentences, output_sentences, output_sentences_inputs

    # Tokenizes sentences.
    # @fit - to fit tokenizer, impacts the assignment of tokens to words
    # @sentences - sentences to tokenize
    # @filters

    # Ex. 2 sentences, total of  6 words,
    #   sentence1: the black cat
    #   sentence2: a red dog
    #
    # A token is arbitrarily assigned to each word, example
    #   the - 1
    #   black - 2
    #   cat - 3
    #   a - 4
    #   red - 5
    #   dog - 6
    # Then the tokenized sentences are:
    #   sentence1: 1 2 3
    #   sentence2: 4 5 6
    #
    # Returns
    #   @sequences - list of all tokenized sentences
    #   @word2idx - a map containing a mapping of integer tokens to words (key -> word, value -> token)
    #   @tokenizer - the tokenizer object, necessary when making translations of new sentences (not from training data)
    def __tokenize_sentences(self, fit, sentences, filters):
        if filters is None:
            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        else:
            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=filters)

        tokenizer.fit_on_texts(fit)
        sequences = tokenizer.texts_to_sequences(sentences)
        word2idx = tokenizer.word_index

        return sequences, word2idx, tokenizer

    # Pads the tokenized sequences. Padding amount is determined by the longest sequence.
    #
    # @sequences - sequences to pad
    # @padding - either 'pre' or 'post' and pads accordingly
    #
    # Ex. padding = 'pre'
    #  sequence = 4 6 9 8
    #  padded_sequence = 0 0 0 4 6 9 8
    def __pad_sequences(self, sequences, padding):
        max_len = max(len(sen) for sen in sequences)
        padded_sequence = pad_sequences(sequences, maxlen=max_len, padding=padding)

        return padded_sequence

    # Create a word embedding matrix for each word from the training data
    # A word embedding essentially a vector of associated words. So for each word in our training data,
    # we associate the word embedding (which is provided by GLoVE project - https://nlp.stanford.edu/projects/glove/).
    # This improves the Model's ability to learn associations. The Word embedding itself is a vector of numbers, where
    # number essentially represents an association with another word.
    #
    # Ex.
    #    Word -> cat
    #    Token -> 5
    #
    #    EmbeddingMatrix[5] = wordEmbedding(cat)
    #
    # where wordEmbedding(cat) is actually something like [45 -35 67 -2 4 ...
    def __create_word_embedding_matrix(self, word2idx):
        embeddings_dictionary = dict()

        #Open Glove predefined word embeddings
        glove_file = open(os.path.join(MY_PATH, GLOVE_FILE), encoding="utf8")

        #Load all word embeddings from pretrained set (GLoVE)
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions

        #Close the Glove embeddings file
        glove_file.close()

        num_words = len(word2idx) + 1
        embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
        for word, index in self._word2idx_input.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix

    def __create_output_model(self, decoder_output_sequences):
        decoder_targets_one_hot = np.zeros((
                len(decoder_output_sequences),
                self._max_output_len,
                self._num_output_words
            ),
            dtype='float32'
        )

        #convert to one hot encoding
        for i, d in enumerate(decoder_output_sequences):
            for t, word in enumerate(d):
                decoder_targets_one_hot[i, t, word] = 1

        return decoder_targets_one_hot

    def __define_encoder(self, embedding_layer):
        encoder_inputs_placeholder = Input(shape=(self._max_input_len,))
        x = embedding_layer(encoder_inputs_placeholder)
        encoder = LSTM(LSTM_NODES, return_state=True, dropout=DROPOUT)

        encoder_outputs, h, c = encoder(x)
        encoder_states = [h, c]

        return encoder, encoder_outputs, encoder_states, encoder_inputs_placeholder

    def __define_decoder(self, encoder_states):
        decoder_inputs_placeholder = Input(shape=(self._max_output_len,))

        decoder_embedding = Embedding(self._num_output_words, LSTM_NODES)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True, dropout=DROPOUT)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

        return decoder_lstm, decoder_outputs, decoder_embedding, decoder_inputs_placeholder

    def __define_dense_layer(self, decoder_outputs):
        decoder_dense = Dense(self._num_output_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        return decoder_dense, decoder_outputs

    # Compile the model and fit(train) to the data. Use a callback to save the best weights that occur throughout
    # training (uses validation accuracy as metric to determine 'best'). Loads the best performing weights before
    # returning the model.
    #
    # @encoder_inputs_placeholder -
    # @decoder_inputs_placeholder -
    # @decoder_outputs -
    # @encoder_input_sequences -
    # @decoder_input_sequences -
    # @decoder_targets_one_hot -
    def __compile_and_fit_model(self, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs, encoder_input_sequences, decoder_input_sequences, decoder_targets_one_hot):
        model = Model([encoder_inputs_placeholder,
          decoder_inputs_placeholder], decoder_outputs)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model_checkpoint_val_acc = ModelCheckpoint(BEST_MODEL_WEIGHTS_FILE, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        r = model.fit(
            [encoder_input_sequences, decoder_input_sequences],
            decoder_targets_one_hot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,
            callbacks=[model_checkpoint_val_acc]
        )

        model.load_weights(BEST_MODEL_WEIGHTS_FILE)
        return model

    # Modify model for predictions, since during actual prediction/translation the entire output is not known at the
    # beginning and is instead predicted word by word over each time step.
    def __modify_model_for_predictions(self, encoder_inputs_placeholder, encoder_states, decoder_embedding, decoder_lstm, decoder_dense):
        encoder_model = Model(encoder_inputs_placeholder, encoder_states)

        decoder_state_input_h = Input(shape=(LSTM_NODES,))
        decoder_state_input_c = Input(shape=(LSTM_NODES,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_inputs_single = Input(shape=(1,))
        decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

        decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

        decoder_states = [h, c]
        decoder_outputs = decoder_dense(decoder_outputs)

        decoder_model = Model(
            [decoder_inputs_single] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    # Translate a sequence (must be padded) using the seq2seq model
    #
    # @input_seq - padded input sequence representing a sentence. Sequence must be encoded using same token
    #              mappings as during training.
    #
    # Return
    #   translated sentence
    def _translate_sequence(self, input_seq):
        states_value = self._encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self._word2idx_output['<sos>']
        eos = self._word2idx_output['<eos>']
        output_sentence = []

        for _ in range(self._max_output_len):
            output_tokens, h, c = self._decoder_model.predict([target_seq] + states_value)
            idx = np.argmax(output_tokens[0, 0, :])

            if eos == idx:
                break

            if idx > 0:
                word = self._idx2word_target[idx]
                output_sentence.append(word)

            target_seq[0, 0] = idx
            states_value = [h, c]

        return ' '.join(output_sentence)

    # Translate a sentence using the seq2seq model
    # @sentence - sentence to translate
    # @print_translation - if True prints input and translation to console
    #
    # Return
    #   translated sentence
    def translate_sentence(self, sentence, print_translation=True):
        if not self._init:
            self.__init_translater()

        sequence = self._input_tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(sequence, maxlen=self._max_input_len, padding='pre')

        translation = self._translate_sequence(padded_sequence[0:0 + 1])
        if print_translation:
            print('Input:', sentence)
            print('Translation:', translation)

        return translation



