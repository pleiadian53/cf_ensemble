
from random import randint, seed
from numpy import array
from numpy import argmax
import keras.backend as K
from tensorflow.keras import models
from numpy import array_equal
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# Sample seq2seq problems 
####################################################################

def create_dataset(train_size, test_size, time_steps,vocabulary_size, verbose= False):
    pairs = [get_reversed_pairs(time_steps,vocabulary_size) for _ in range(train_size)]
    pairs=np.array(pairs).squeeze()
    X_train = pairs[:,0]
    y_train = pairs[:,1]
    pairs = [get_reversed_pairs(time_steps,vocabulary_size) for _ in range(test_size)]
    pairs=np.array(pairs).squeeze()
    X_test = pairs[:,0]
    y_test = pairs[:,1] 

    if(verbose):
        print('\nGenerated sequence datasets as follows')
        print('X_train.shape: ', X_train.shape,'y_train.shape: ', y_train.shape)
        print('X_test.shape: ', X_test.shape,'y_test.shape: ', y_test.shape)

        return X_train, y_train, X_test, y_test

def config_sample_problem(n_features=10, n_timesteps_in=4): 
    """ 
    Configure problem: E.g., 
            n_timesteps_in = 4 #@param {type:"integer"} => each input sample has 4 values

            n_features = 10   #@param {type:"integer"} => each value is one_hot_encoded with 10 0/1
            """
    # generate random sequence
    X,y = get_reversed_pairs(n_timesteps_in,  n_features, verbose=True)
    
    # generate datasets
    train_size= 20000 #@param {type:"integer"}
    test_size = 200  #@param {type:"integer"}

    X_train, y_train , X_test,  y_test = create_dataset(train_size, test_size, n_timesteps_in, n_features , verbose=True)

    return (X_train, y_train , X_test, y_test)



# Sample models
####################################################################

def get_mlp(n_units=64, n_timesteps=4, n_features=10, 
                loss='categorical_crossentropy', 
                activation='softmax',
                verbose=1): 

    numberOfPerceptrons = n_units
    model_Multi_Layer_Perceptron = Sequential(name='model_Multi_Layer_Perceptron')
    model_Multi_Layer_Perceptron.add(Input(shape=(n_timesteps, n_features)))
    model_Multi_Layer_Perceptron.add(Dense(4*numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(2*numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(n_features, activation=activation))

    model_Multi_Layer_Perceptron.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    if verbose: 
        model_Multi_Layer_Perceptron.summary()
        plot_model(model_Multi_Layer_Perceptron,show_shapes=True)

    return model_Multi_Layer_Perceptron

def get_lstm(n_timesteps=4, n_features=10, n_units=10, loss='categorical_crossentropy', verbose=1): 

    numberOfUnits = n_units # LSTM output dimension; the number of LSTM cells (each cell outputs 1 value)

    model_Single_LSTM_default_output = Sequential(name='model_Single_LSTM_default_output')
    model_Single_LSTM_default_output.add(Input(shape=(n_timesteps, n_features))) # e.g. n_features=10, 10-D vector per timestep

    # Single LSTM Layer with default output
    model_Single_LSTM_default_output.add(LSTM(numberOfUnits)) # e.g. numberOfUnits=16, 10D with 4 time steps rollout-> 16D (@ last hidden state)

    # Repeat the output of LSTM n_timesteps (4 in our example)
    model_Single_LSTM_default_output.add(RepeatVector(n_timesteps)) # e.g. n_timesteps=4, 16-D => 4 x 16 (the same 16D repeated 4 times)

    # Dense layer recieves 4 x LSTM outputs as input vector
    model_Single_LSTM_default_output.add(Dense(n_features, activation='softmax'))

    model_Single_LSTM_default_output.compile(loss=loss, optimizer='adam', 
        metrics=['accuracy'])

    if verbose: 
        model_Single_LSTM_default_output.summary()

    return model_Single_LSTM_default_output

def get_lstm_time_distributed(n_timesteps=4, n_features=10, n_units=10, loss='categorical_crossentropy', verbose=1): 

    numberOfUnits = n_units # LSTM output dimension

    model_LSTM_return_sequences = Sequential(name='model_LSTM_return_sequences')
    model_LSTM_return_sequences.add(Input(shape=(n_timesteps, n_features)))

    # First LSTM layer with return_sequences=True
    model_LSTM_return_sequences.add(LSTM(numberOfUnits,return_sequences=True)) # note: return_sequence=True

    # The output of the First LSTM has the 3 dimensions as expected by
    # Second LSTM layer
    # Thus, we do not need to use RepeatVector!
    model_LSTM_return_sequences.add(LSTM(numberOfUnits,return_sequences=True)) # note return_sequence=True

    # The output of the Second LSTM has the 3 dimensions 
    # To supply the output to a dense layer
    # we need to use TimeDistributed layer!
    model_LSTM_return_sequences.add(TimeDistributed(Dense(n_features, activation='softmax'))) # <<< TimeDistributed(Dense())
    # ... for each LSTM output at each time step (batch_size, n_features), the Dense layer will produce one output predicition 
    # ... since we have `n_timesteps` (say 4) time steps, this gives as (batch_size, n_timesteps, n_features) as the output for time-distributed dense layer

    model_LSTM_return_sequences.compile(loss=loss, optimizer='adam', 
            metrics=['accuracy'])

    if verbose: 
        model_LSTM_return_sequences.summary()

    return model_LSTM_return_sequences

def get_lstm_connecting_states(n_timesteps=4, n_features=10, n_units=10, 
         loss='categorical_crossentropy', 
         activation='softmax',
         verbose=1):

    numberOfUnits = n_units

    input= Input(shape=(n_timesteps, n_features))
    lstm1 = LSTM(numberOfUnits,return_state=True)

    LSTM_output, state_h, state_c = lstm1(input) # `LSTM_output` is essentially the hidden state, which same as `state_h``
    states = [state_h, state_c]

    repeat=RepeatVector(n_timesteps) # Turn 2D (None, n_features) into 3D (None, n_timesteps, n_features)
    LSTM_output = repeat(LSTM_output) # 2D: (bsize, n_features) => 3D: (bsize, n_timesteps, n_features)

    lstm2 = LSTM(numberOfUnits, return_sequences=True)
    all_state_h = lstm2(LSTM_output, initial_state=states) # Notice: initial_state <- states from previous LSTM layer

    dense = TimeDistributed(Dense(n_features, activation=activation))
    output = dense(all_state_h)

    model_LSTM_return_state = Model(input, output,name='model_LSTM_return_state')
    model_LSTM_return_state.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), 
        metrics=['accuracy'])

    if verbose: 
        model_LSTM_return_state.summary() 

    return model_LSTM_return_state

def get_stacked_lstm(n_timesteps=4, n_features=10, n_units=10, 
                        loss='categorical_crossentropy', 
                        activation='softmax',
                        verbose=1): 
    """
    A model containing Multiple LSTM Layers by connecting them with return_sequences=True & return_state=True

    Note that the second LSTM layer: 
                - initialized by last hidden states and cell states from the first LSTM layer
                - consumes the first LSTM layer's hidden states at all timesteps to produce its output

    Memo
    ----
    - We can argue that first LSTM layer "encodes" the input (X) in a representation that the second LSTM layer 
        can then "decode" it to produce the expected output (y)
    - The encoded representation of the input (X) in the last model comprises all hidden states + last hidden state + last cell state 
    - We can think of 

            [all hidden states + last hidden + last cell]

        as a "context vector" from the first LSTM layer

    - The first LSTM layer can be equated to a encoder whereas the second LSTM layer can be regarded as a decoder
    - We can stack more LSTM layers to construct a deeper encoder-decoder-like network

    - For a pictorial mnemonic on `return_sequence` and `return_state`, see this notebook: 

        <repo>/machine_learning_examples/sequence_model/seq2seq_Part_C_Basic_Encoder_Decoder.ipynb

    """
    numberOfLSTMCells = n_units

    input_seq= Input(shape=(n_timesteps, n_features))
    lstm1 = LSTM(numberOfLSTMCells, return_sequences=True, return_state=True)

    all_state_h, state_h, state_c = lstm1(input_seq) # all_state_h is already 3D referencing hidden states from all time steps
    
    # Note: suppose n_units = 10 and n_timesteps=4, then `all_state_h` has shape (None, 4, 10)
    #       if `return_sequences` <- False, then the LSTM output have been a 2D tensor of shape (None, 10)
    states = [state_h, state_c] # collect hidden and cell states

    lstm2 = LSTM(numberOfLSTMCells, return_sequences=True)
    all_state_h2 = lstm2(all_state_h, initial_state=states) # init this LSTM layer with the states from previous LSTM layer

    dense = TimeDistributed(Dense(n_features, activation=activation))
    output_seq = dense(all_state_h2)
    
    model_LSTM_return_sequences_return_state = Model(input_seq, output_seq,
        name='model_LSTM_all_state_h_return_state')

    model_LSTM_return_sequences_return_state.compile(loss=loss, 
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

    if verbose: 
        model_LSTM_return_sequences_return_state.summary()
        plot_model(model_LSTM_return_sequences_return_state, show_shapes=True)

    return model_LSTM_return_sequences_return_state

def get_hard_coded_decoder_input_model(batch_size, n_timesteps=4, n_features=10, 
                                            n_units=16,  # number of LSTM cells

                                            loss='categorical_crossentropy', 
                                            activation='softmax', 
                                            optimizer='rmsprop',

                                            **kargs):
    numberOfLSTMunits = n_units 

    # The first part is encoder
    encoder_inputs = Input(shape=(n_timesteps, n_features), name='encoder_inputs')
    encoder_lstm = LSTM(numberOfLSTMunits, return_state=True,  name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # initial context vector is the states of the encoder
    states = [state_h, state_c]

    # Set up the decoder layers
    # Attention: decoder receives 1 token at a time &
    # decoder outputs 1 token at a time 
    decoder_inputs = Input(shape=(1, n_features)) # output 1 token at a time => n_timesteps = 1
    decoder_lstm = LSTM(numberOfLSTMunits, return_sequences=True, 
        return_state=True, name='decoder_lstm')

    decoder_dense = Dense(n_features, activation=activation,  name='decoder_dense')

    all_outputs = []

    # Prepare decoder initial input data: just contains the START character 0
    # Note that we made it a constant one-hot-encoded in the model
    # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
    decoder_input_data = np.zeros((batch_size, 1, n_features))
    decoder_input_data[:, 0, 0] = 1 

    # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
    inputs = decoder_input_data
    # decoder will only process one time step at a time
    # loops for fixed number of time steps: n_timesteps
    for _ in range(n_timesteps):
        # Run the decoder on one time step
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states) # the first `states` is the initial context vector from encoder
        outputs = decoder_dense(outputs) # 
            
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs) # NOTE: `outputs` is 3D but with only 1 time step: (b_size, 1, n_feature)
            
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

    # Concatenate all predictions such as [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # Define and compile model 
    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_encoder_decoder.summary()
    return model


# Training, Testing and Evalution
####################################################################

# Function to Train & Test  given model (Early Stopping monitor 'val_loss')
def train_test(model, X_train, y_train, X_test, y_test, epochs=500, verbose=0):

    validation_split = 0.1
    patience = 20

    # patient early stopping
    #es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience=20)
    es = EarlyStopping(monitor='val_loss', 
        mode='min', verbose=1, patience=patience)
    # train model
    print(f"> Training for {epochs} epochs: validation_split={validation_split}, EarlyStopping(monitor='val_loss', patience={patience})....")
    history=model.fit(X_train, y_train, validation_split= 0.1, epochs=epochs,  verbose=verbose, callbacks=[es])
    print(f"> {epochs} epochs training completed ...")

    # report training
    # list all data in history
    # print(history.history.keys())
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print('\nPREDICTION ACCURACY (%):')
    print('Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
    
    # summarize history for accuracy
    try: 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
    except: 
        print(history.history.keys())
        raise ValueError

    plt.title(model.name+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model.name+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    return history


# Data utilities
####################################################################

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
        return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_reversed_pairs(time_steps,vocabulary_size,verbose= False):
    # generate random sequence
    sequence_in = generate_sequence(time_steps, vocabulary_size)
    sequence_out = sequence_in[::-1]
    
    # one hot encode
    X = one_hot_encode(sequence_in, vocabulary_size)
    y = one_hot_encode(sequence_out, vocabulary_size)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))

    if(verbose):
        print('\nSample X and y')
        print('\nIn raw format:')
        print('X[0]=%s, y[0]=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))
        print('\nIn one_hot_encoded format:')
        print('X[0]=%s' % (X[0]))
        print('y[0]=%s' % (y[0]))
        return X,y

# Demo
####################################################################


def test(): 
    return 

    if __name__ == "__main__":
        test()

