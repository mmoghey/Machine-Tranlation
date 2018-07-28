# machine-translation
Input data
As the training data we will use the data from http://www.manythings.org/anki/. It contains 135842 English sentences with the corresponding French translation. For simplicity, we filter out long sentences so the resulting training set contains 119854 pairs and the validation set contains 5000 paris.

The dataset class is already implemented for you in the dataset.py file.

Model
The model class, Seq2SeqModel is located in the file model.py. Conceptually the methods of this class are divided into three groups:

Initialization: __init__
Encoder methods: zero_state and encode_sentence
Decoder methods: decoder_state, decoder_initial_inputs, and decode_sentence
In the __init__ method you should declare all layers that you are going to use: an embedding layer, an LSTM layer for the encoder, an LSTM cell for the decoder, and an output projection layer. Note that you should use the torch.nn.LSTM class for the encoder and the torch.nn.LSTMCell class for the decoder. The reason for this is that we will use special PyTorch functions that correctly handle variable lengths sequences (since we have sentence of different length in a batch) for the encoder, and just a for loop for the decoder. The torch.nn.LSTMCell class is more suitable in the latter case.

The zero_state method returns initial hidden state of the encoder. Namely, it should return a tuple of tensors for the hidden state h and the cell state c correspondingly (see LSTM lecture for details on those states). This method is meant to be called from the encode_sentence method.

The encode_sentence method encodes sentences in a batch into hidden vectors z, which is the last hidden state of the LSTM network. Note that since we have sequences of different lengths in a single batch, we cannot just take the last hidden state as follows: z = h[:,-1,:]. Instead, you will need to use the torch.nn.utils.rnn.pack_padded_sequence function. You might find the function get_sequences_lengths in utils.py useful for this task.

The decoder_state method creates initial hidden state of the decoder. Since we want to decode a translation of the input sequence, it takes the hidden vectors z as the argument and returns a tuple of (z,c), where c is a vector of zeros (as in the zero_state). This method is meant to be called from the decode_sentence method.

The decoder_initial_inputs method should just return a batch of indices of the <s> token to be served as input to the decoder at the first timestep. This method is meant to be called from the decode_sentence method.

Finally, the decode_sentence method decodes the outputs translation using a for loop. Its implementation is very similar to that of the second homework.
