
from random import randint, seed
from numpy import array
from numpy import argmax
import keras.backend as K
from tensorflow.keras import models
from numpy import array_equal
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

from sklearn.model_selection import train_test_split


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
                metrics=['accuracy', ],
                verbose=1): 

    numberOfPerceptrons = n_units
    model_Multi_Layer_Perceptron = Sequential(name='model_Multi_Layer_Perceptron')
    model_Multi_Layer_Perceptron.add(Input(shape=(n_timesteps, n_features)))
    model_Multi_Layer_Perceptron.add(Dense(4*numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(2*numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(numberOfPerceptrons))
    model_Multi_Layer_Perceptron.add(Dense(n_features, activation=activation))

    model_Multi_Layer_Perceptron.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=metrics)

    if verbose: 
        model_Multi_Layer_Perceptron.summary()
        plot_model(model_Multi_Layer_Perceptron,show_shapes=True)

    return model_Multi_Layer_Perceptron

def get_lstm(n_timesteps=4, n_features=10, n_units=10, 
                loss='categorical_crossentropy', activation='softmax', 
                metrics=['accuracy', ],
                verbose=1): 

    numberOfUnits = n_units # LSTM output dimension; the number of LSTM cells (each cell outputs 1 value)

    model_Single_LSTM_default_output = Sequential(name='model_Single_LSTM_default_output')
    model_Single_LSTM_default_output.add(Input(shape=(n_timesteps, n_features))) # e.g. n_features=10, 10-D vector per timestep

    # Single LSTM Layer with default output
    model_Single_LSTM_default_output.add(LSTM(numberOfUnits)) # e.g. numberOfUnits=16, 10D with 4 time steps rollout-> 16D (@ last hidden state)

    # Repeat the output of LSTM n_timesteps (4 in our example)
    model_Single_LSTM_default_output.add(RepeatVector(n_timesteps)) # e.g. n_timesteps=4, 16-D => 4 x 16 (the same 16D repeated 4 times)

    # Dense layer recieves 4 x LSTM outputs as input vector
    model_Single_LSTM_default_output.add(Dense(n_features, activation=activation))

    model_Single_LSTM_default_output.compile(loss=loss, optimizer='adam', metrics=metrics)

    if verbose: 
        model_Single_LSTM_default_output.summary()

    return model_Single_LSTM_default_output

def get_lstm_time_distributed(n_timesteps=4, n_features=10, n_units=10, 
                                loss='categorical_crossentropy', activation='softmax', 
                                metrics=['accuracy', ],
                                verbose=1): 

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
    model_LSTM_return_sequences.add(TimeDistributed(Dense(n_features, activation=activation))) # <<< TimeDistributed(Dense())
    # ... for each LSTM output at each time step (batch_size, n_features), the Dense layer will produce one output predicition 
    # ... since we have `n_timesteps` (say 4) time steps, this gives as (batch_size, n_timesteps, n_features) as the output for time-distributed dense layer

    model_LSTM_return_sequences.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=metrics)

    if verbose: 
        model_LSTM_return_sequences.summary()

    return model_LSTM_return_sequences

def get_lstm_connecting_states(n_timesteps=4, n_features=10, n_units=10, 
                                loss='categorical_crossentropy', 
                                activation='softmax',
                                metrics=['accuracy', ],
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
    model_LSTM_return_state.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=metrics)

    if verbose: 
        model_LSTM_return_state.summary() 

    return model_LSTM_return_state

def get_stacked_lstm(n_timesteps=4, n_features=10, n_features_out=None,
                        n_units=10, 
                        loss='categorical_crossentropy', 
                        activation='softmax',
                        metrics=['accuracy', ],
                        optimizer='adam', 
                        verbose=1, **kargs): 
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
    output_bias = kargs.get('output_bias', None)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    dropout = kargs.get("dropout", 0.0)

    numberOfLSTMCells = n_units
    if n_features_out is None: n_features_out = n_features

    input_seq= Input(shape=(n_timesteps, n_features))

    lstm1 = LSTM(numberOfLSTMCells, return_sequences=True, return_state=True, dropout=dropout) # dropout, recurrent_dropout

    all_state_h, state_h, state_c = lstm1(input_seq) # all_state_h is already 3D referencing hidden states from all time steps
    
    # Note: suppose n_units = 10 and n_timesteps=4, then `all_state_h` has shape (None, 4, 10)
    #       if `return_sequences` <- False, then the LSTM output have been a 2D tensor of shape (None, 10)
    states = [state_h, state_c] # collect hidden and cell states

    lstm2 = LSTM(numberOfLSTMCells, return_sequences=True)
    all_state_h2 = lstm2(all_state_h, initial_state=states) # init this LSTM layer with the states from previous LSTM layer

    if output_bias is None: 
        dense = TimeDistributed(Dense(n_features_out, activation=activation))
    else: 
        dense = TimeDistributed(Dense(
                                    n_features_out, 
                                    activation=activation, 
                                    output_bias=output_bias))
    output_seq = dense(all_state_h2)
    
    model_LSTM_return_sequences_return_state = Model(input_seq, output_seq,
        name='model_LSTM_all_state_h_return_state')

    model_LSTM_return_sequences_return_state.compile(loss=loss, 
        optimizer=optimizer, # keras.optimizers.Adam(lr=0.001),
        metrics=metrics)

    if verbose: 
        model_LSTM_return_sequences_return_state.summary()
        plot_model(model_LSTM_return_sequences_return_state, show_shapes=True)

    return model_LSTM_return_sequences_return_state

def get_hard_coded_decoder_input_model(batch_size, n_timesteps=4, n_features=10, 
                                            n_units=16,  # number of LSTM cells

                                            loss='categorical_crossentropy', 
                                            activation='softmax', 
                                            metrics=['accuracy', ],
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
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_encoder_decoder.summary()
    return model

# Sample attention models
################################################################################

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, verbose=0):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.verbose= verbose

    def call(self, query, values):
        if self.verbose:
            print('\n******* Bahdanau Attention STARTS******')
            print('query (decoder hidden state): (batch_size, hidden size) ', query.shape)
            print('values (encoder all hidden state): (batch_size, max_len, hidden size) ', values.shape)

        # Why expand_dims()? 
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)

        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis (of length `max_len`) to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
    
        if self.verbose:
            print('query_with_time_axis:(batch_size, 1, hidden size) ', query_with_time_axis.shape)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
                          self.W1(query_with_time_axis) + self.W2(values))) # V: units -> 1
        # Note: broadcast query_with_time_axis (None, 1, hsize) -> (None, 4, hsize)
        #       at every encoder hidden state, it sees the same `ht`

        if self.verbose:
            print('score: (batch_size, max_length, 1) ',score.shape)
    
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1) # one score per time step; axis=1 => normalize over all time steps
        if self.verbose:
            print('attention_weights: (batch_size, max_length, 1) ', attention_weights.shape)
        
        # context_vector shape before sum == (batch_size, max_len, hidden_size)
        context_vector = attention_weights * values # (None, 4, 1) * (None, 4, 16)
        # Tip: fix first two dimension, then we effectively mutiply the weight (1-D) across each vector of 16-D
        #.     => do this for all examples in the batch and for all time steps
        # Patten: 
        # 
        # x     v v v v v 
        # x     v v v v v
        # x     v v v v v
        # x     v v v v v
        # 
        # (4, 1) * (4, 5) => (4, 5), each weight of 1-D gets multipled cross each vector of 5-D 
        # (4, 1) -> broadcast -> (4, 5)
        # (4, 5) * (4, 5) as in element-wise mul

        if self.verbose:
            print('context_vector before reduce_sum: (batch_size, max_length, hidden_size) ', context_vector.shape)
    
        # context_vector shape after sum == (batch_size, hidden_size) # weighted average over all timesteps => marginalize time axis (axis=1)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if self.verbose:
            print('context_vector after reduce_sum: (batch_size, hidden_size) ', context_vector.shape)
            print('\n******* Bahdanau Attention ENDS******')
    
        return context_vector, attention_weights

def get_attention_encoder_decoder_model(n_timesteps=4, n_features=10, 
                                            n_units=16,  # number of LSTM cells

                                            batch_size=1, 

                                            loss='categorical_crossentropy', 
                                            activation='softmax', 
                                            metrics=['accuracy', ],
                                            optimizer='rmsprop',

                                            input_encoding='one-hot',

                                            **kargs): 
    # tf.keras.backend.set_floatx('float64')

    verbose = kargs.get('verbose', 0)
    start_token = kargs.get('start_token', np.zeros(n_features)) # used only when not using one-hot encoding

    latentSpaceDimension = n_units

    if verbose:
        print('***** Model Hyper Parameters *******')
        print('latentSpaceDimension: ', latentSpaceDimension)
        print('batch_size: ', batch_size)
        print('sequence length: ', n_timesteps)
        print('n_features: ', n_features)

        print('\n***** TENSOR DIMENSIONS *******')

    # The first part is encoder 
    encoder_inputs = Input(shape=(n_timesteps, n_features), name='encoder_inputs')
    encoder_lstm = LSTM(latentSpaceDimension, return_sequences=True, return_state=True,  name='encoder_lstm')
    encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    # NOTE: `encoder_outputs` contains hidden states from all timesteps

    if verbose:
        print ('Encoder output shape: (batch size, sequence length, latentSpaceDimension) {}'.format(encoder_outputs.shape))
        print ('Encoder Hidden state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_h.shape))
        print ('Encoder Cell state shape: (batch size, latentSpaceDimension) {}'.format(encoder_state_c.shape))

    # initial context vector is the states of the encoder
    encoder_states = [encoder_state_h, encoder_state_c]
    if verbose:
        print(f"[info] Encoder states: {encoder_states}")

    # Set up the attention layer
    attention= BahdanauAttention(latentSpaceDimension, verbose=verbose)

    # Set up the decoder layers
    decoder_inputs = Input(shape=(1, (n_features + latentSpaceDimension)),name='decoder_inputs')
    # Each time-step of the decoder input (ht) is the concatenation of raw input and the context vector 
    # i.e. n_features + latentSpaceDimension: input data || latent feature representation

    decoder_lstm = LSTM(latentSpaceDimension,  return_state=True, name='decoder_lstm')
    decoder_dense = Dense(n_features, activation=activation,  name='decoder_dense')

    all_outputs = []

    # 1 initial decoder's input data

    if input_encoding.startswith("one"): 
        # Prepare initial decoder input data that just contains the start character 
        # Note that we made it a constant one-hot-encoded in the model
        # that is, [1 0 0 0 0 0 0 0 0 0] is the first input for each loop
        # one-hot encoded zero(0) is the start symbol
        inputs = np.zeros((batch_size, 1, n_features)) # start token
        inputs[:, 0, 0] = 1 
    else: 
        inputs = np.zeros((batch_size, 1, n_features)) # start token
        assert len(start_token) == n_features
        # if np.isscalar(start_token): start_token = np.repeat(start_token, n_features)
        inputs[:, 0, :] = start_token 

    # 2 initial decoder's state
    # encoder's last hidden state + last cell state
    decoder_outputs = encoder_state_h
    states = encoder_states
    if verbose:
        print('initial decoder inputs: ', inputs.shape) # (1, 1, 10)
        print(f'ht (t=0): {encoder_state_h.shape}') # (None, 16)
        print(f"last encoder states: {states[0].shape}, {states[1].shape}") # (None, 16), (None, 16)


    # decoder will only process one time step at a time.
    for step in range(n_timesteps):

        # 3 pay attention
        # create the context vector by applying attention to 
        # decoder_outputs (last hidden state) + encoder_outputs (all hidden states)
        context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
        if verbose:
            print("[{}] Attention context_vector: (batch size, units) {}".format(step, context_vector.shape)) # (None, 16)
            print("[{}] Attention weights : (batch_size, sequence_length, 1) {}".format(step, attention_weights.shape)) # (None, 4, 1)
            print('     decoder_outputs: (batch_size,  latentSpaceDimension) ', decoder_outputs.shape ) # (None, 16)

        context_vector = tf.expand_dims(context_vector, 1) # 2D -> 3D by adding time dimension
        if verbose:
            print('Reshaped context_vector: ', context_vector.shape ) # (None, 16) -> (None, 1, 16)
            print(f'shape(inputs): {inputs.shape}')
            
            # NOTE: 
            # (None, 1, 16) || (1, 1, 10)? to concatenate `context_vector` and `inputs`, we need to ensure that two tensors have the same float type
            # 
            # tf.keras.backend.set_floatx('float64') # <<< 

        # 4. concatenate the input + context vectore to find the next decoder's input
        inputs = tf.concat([context_vector, inputs], axis=-1)
        
        if verbose:
            print('> After concat inputs: (batch_size, 1, n_features + hidden_size): ', inputs.shape ) 
            # 1. (None, 1, 16)-> (1, 1, 16)  
            # 2. (1, 1, 16) || (1, 1, 10) => (1, 1, 26)

        # 5. passing the concatenated vector to the LSTM
        # Run the decoder on one timestep with attended input and previous states
        decoder_outputs, state_h, state_c = decoder_lstm(inputs,  # <<< inputs with context
                                                initial_state=states)
        # decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
      
        outputs = decoder_dense(decoder_outputs)

        # 6. Use the last hidden state for prediction the output

        # save the current prediction
        # we will concatenate all predictions later
        outputs = tf.expand_dims(outputs, 1)
        all_outputs.append(outputs)

        # 7. Reinject the output (prediction) as the input for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]

        if verbose: print(f"> Step #{step} completed.")

    if verbose: 
        print(f"[info] size(all_outputs): {len(all_outputs)}")
        print(f"... example shape:\n{all_outputs[0].shape}\n")
    # 8. After running Decoder for max time steps
    # we had created a predition list for the output sequence
    # convert the list to output array by Concatenating all predictions 
    # such as [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    # 9. Define and compile model 
    model_encoder_decoder_Bahdanau_Attention = Model(encoder_inputs, 
                                                     decoder_outputs, name='model_encoder_decoder')
    model_encoder_decoder_Bahdanau_Attention.compile(optimizer=optimizer, 
                                                     loss=loss, metrics=metrics)

    return model_encoder_decoder_Bahdanau_Attention


# Training, Testing and Evalution
####################################################################

# Function to Train & Test  given model (Early Stopping monitor 'val_loss')
def train_test(model, X_train, y_train, X_test, y_test, 
                 batch_size=32, epochs=300, **kargs):
    # from sklearn.model_selection import train_test_split

    validation_split = kargs.get('validation_split', 0.1)
    patience = kargs.get('patience', 20)

    verbose = kargs.get('verbose', 0)
    random_state = kargs.get('random_state', 53)
    n_train = X_train.shape[0]

    # patient early stopping
    #es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, patience=20)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    es = EarlyStopping(
              patience=patience,
              min_delta=0.05,
              baseline=0.8,
              mode='min',
              monitor='val_loss',
              restore_best_weights=True,
              verbose=1)
    csv_file = kargs.get('csv_file', 'polarity-seq2seq1-training.csv')

 
    # train validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split, random_state=random_state)

    # Prepare the training dataset   
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size= n_train // 4).batch(batch_size, drop_remainder=True).prefetch(1)
    # NOTE: .prefetch() allows later elements to be prepared while the current element is being processed. 
    #       This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    # Prepare the test dataset 
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True) 


    # train model
    print(f"> Training for {epochs} epochs: validation_split={validation_split}, EarlyStopping(monitor='val_loss', patience={patience})....")
    history=model.fit(  train_dataset,   # X_train, y_train,
                          
                            validation_data=val_dataset,
                            
                            epochs=epochs, 
                            # batch_size=batch_size, 
                            verbose=verbose, 

                            callbacks=[es, CSVLogger(csv_file)])

    print(f"> {epochs} epochs (bsize={batch_size}) training completed ...")

    # report training
    # list all data in history
    # print(history.history.keys())
    # evaluate the model
    _, train_acc = model.evaluate( train_dataset,  # X_train, y_train, 
                                       # batch_size=batch_size, 
                                       verbose=0)
    _, test_acc = model.evaluate( test_dataset,  # X_test, y_test, 
                                        # batch_size=batch_size, 
                                        verbose=0)
    
    print('\nPREDICTION ACCURACY (%):')
    print('> Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
    
    # summarize history for accuracy
    try: 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
    except: 
        print(history.history.keys())
        raise ValueError

    plt.title(model.name +' accuracy')
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

