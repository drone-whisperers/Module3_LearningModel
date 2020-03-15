import os, sys
import numpy as np
from numpy import asarray, zeros
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

BATCH_SIZE = 64
EPOCHS = 10
LSTM_NODES =256
MAX_SENTENCE_LENGTH = 50
MAX_NUM_WORDS = 2000
EMBEDDING_SIZE = 100
GLOVE_FILE = "./TrainingData/glove.6B.100d.txt"
MY_PATH = os.path.abspath(os.path.dirname(__file__))

class TranslaterCls:
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

    def __init_translater(self):
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

        #todo: investigate what these contain, and then rename?
        self._idx2word_input = {v:k for k, v in self._word2idx_input.items()}
        self._idx2word_target = {v:k for k, v in self._word2idx_output.items()}

        self._init = True

    def __process_training_data(self):
        input_sentences = []
        output_sentences = []
        output_sentences_inputs = []
        count = 0

        for line in open(os.path.join(MY_PATH, "./TrainingData/generatedTrainingData.txt"), "r").read().splitlines():
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

    def __tokenize_sentences(self, fit, sentences, filters):
        if filters is None:
            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        else:
            tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=filters)

        tokenizer.fit_on_texts(fit)
        sequences = tokenizer.texts_to_sequences(sentences)
        word2idx = tokenizer.word_index

        return sequences, word2idx, tokenizer

    def __pad_sequences(self, sequences, padding):
        max_len = max(len(sen) for sen in sequences)
        padded_sequence = pad_sequences(sequences, maxlen=max_len, padding=padding)

        return padded_sequence

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
        encoder = LSTM(LSTM_NODES, return_state=True)

        encoder_outputs, h, c = encoder(x)
        encoder_states = [h, c]

        return encoder, encoder_outputs, encoder_states, encoder_inputs_placeholder

    def __define_decoder(self, encoder_states):
        decoder_inputs_placeholder = Input(shape=(self._max_output_len,))

        decoder_embedding = Embedding(self._num_output_words, LSTM_NODES)
        decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

        decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)

        return decoder_lstm, decoder_outputs, decoder_embedding, decoder_inputs_placeholder

    def __define_dense_layer(self, decoder_outputs):
        decoder_dense = Dense(self._num_output_words, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        return decoder_dense, decoder_outputs

    def __compile_and_fit_model(self, encoder_inputs_placeholder, decoder_inputs_placeholder, decoder_outputs, encoder_input_sequences, decoder_input_sequences, decoder_targets_one_hot):
        model = Model([encoder_inputs_placeholder,
          decoder_inputs_placeholder], decoder_outputs)
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        r = model.fit(
            [encoder_input_sequences, decoder_input_sequences],
            decoder_targets_one_hot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,
        )

        return model

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

            word = ''

            if idx > 0:
                word = self._idx2word_target[idx]
                output_sentence.append(word)

            target_seq[0, 0] = idx
            states_value = [h, c]

        return ' '.join(output_sentence)

    def translate_sentence(self, sentence):
        if not self._init:
            self.__init_translater()

        sequence = self._input_tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(sequence, maxlen=self._max_input_len, padding='pre')
        return self._translate_sequence(padded_sequence[0:0 + 1])



