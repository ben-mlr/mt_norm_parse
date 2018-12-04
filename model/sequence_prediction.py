import torch
from torch.autograd import Variable
from io_.info_print import printing
from io_.from_array_to_text import output_text
import numpy as np


def greedy_decode(generator, batchIter, model,char_dictionary, batch_size, pad=1, verbose=0):

        with torch.no_grad():

            for batch in batchIter:
                # read src sequence
                src_seq = batch.input_seq
                src_len = batch.input_seq_len
                src_mask = batch.input_seq_mask
                # do something with it : When do you stop decoding ?
                max_len = src_seq.size(1)

                decode_sequence(model=model, generator=generator, char_dictionary=char_dictionary,
                                max_len=max_len, src_seq=src_seq, src_mask=src_mask,src_len=src_len, batch_size=batch_size, pad=pad,
                                verbose=verbose)


def decode_sequence(model, generator, char_dictionary, max_len, src_seq, src_mask, src_len,batch_size, pad=1, verbose=2):
    output_seq = pad*np.ones(src_seq.size(), dtype=np.int64)
    # we start with the _START symbol
    output_seq[:, 0] = 2
    output_mask = np.ones(src_seq.size(), dtype=np.int64)
    output_mask[:, 1:] = 0
    output_len = Variable(torch.from_numpy(np.ones(src_seq.size(0),dtype=np.int64)),requires_grad=False)
    output_mask = Variable(torch.from_numpy(output_mask),requires_grad=False)
    output_seq = Variable(torch.from_numpy(output_seq),requires_grad=False)
    printing("Data Start ".format(output_seq, output_len, output_mask), verbose=verbose, verbose_level=6)
    for step, char_decode in enumerate(range(2,  max_len)):
        decoding_states = model.forward(input_seq=src_seq, output_seq=output_seq, input_mask=src_mask,
                                        input_word_len=src_len, output_mask=output_mask,
                                        output_word_len=output_len)
        # decoding_states = model.forward(input_seq=src_seq, output_seq=None, input_mask=src_mask,
        # input_word_len=src_len, output_mask=None, output_word_len=None)
        # [batch, seq_len, V]
        scores = generator.forward(x=decoding_states)
        # each time step predict the most likely
        # len
        output_len = Variable(torch.from_numpy(np.ones(src_seq.size(0),dtype=np.int64)),requires_grad=False)
        output_len[:] = char_decode
        # mask
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, char_decode:] = 0
        output_mask = Variable(torch.from_numpy(output_mask),requires_grad=False)
        # new seq
        predictions = scores.argmax(dim=2)
        printing("Prediction {} ".format(predictions), verbose=verbose, verbose_level=5)
        printing("scores: {} scroes {} scores sized  {} predicion size {} prediction {} outputseq ".format(scores,scores.size(), predictions.size(), predictions[:, -1], output_seq.size()), verbose=verbose,
                 verbose_level=5)
        output_seq[:, char_decode-1] = predictions[:, -1]
        sequence = [" ".join([char_dictionary.get_instance(output_seq[batch, char_i]) for char_i in range(max_len)]) + " / " for batch in range(batch_size)]
        printing("Decoding step {} decoded target {} ".format(step, sequence), verbose=verbose,
                 verbose_level=2)
        text = output_text(predictions, char_dictionary)
        printing("PREDICTION : {} ".format(text), verbose=verbose,
                 verbose_level=2)
    return text


def decode_seq_begins_with(seq_string, dictionary, model, generator, char_dictionary, pad=1, max_len=10, verbose=2):
    with torch.no_grad():
        _seq_string = ["_START"]


        _seq_string.extend(list(seq_string))
        seq_string = _seq_string #+ ["_PAD_CHAR"]#["_END_CHAR"]#["_PAD_CHAR"]
        if len(seq_string) > max_len:
            # cutting to respect dim requirements
            seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
        printing("INPUT SEQ is {} ".format(seq_string), verbose=verbose, verbose_level=2)
        sequence_characters = [dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
        masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
        char_seq = Variable(torch.from_numpy(np.array([sequence_characters,sequence_characters])), requires_grad=False)
        char_mask = Variable(torch.from_numpy(np.array([masks,masks])), requires_grad=False)
        char_len = Variable(torch.from_numpy(np.array([[min(max_len, len(seq_string))],[min(max_len, len(seq_string))]])))
        print(char_len)
        batch_size = 2
        decode_sequence(model=model, generator=generator, char_dictionary=char_dictionary,
                        max_len=max_len,batch_size=batch_size,
                        src_seq=char_seq, src_len=char_len, src_mask=char_mask,
                        pad=pad, verbose=verbose)


def decode_interacively(dictionary, model, generator, char_dictionary,  max_len, verbose,pad=1):

    while True:
        seq_string = input("What type do you want to normalize ? to stop type : stop    ")
        if seq_string=="":
            continue
        if seq_string=="stop":
            break
        else:
            decode_seq_begins_with(seq_string, dictionary, model, generator, char_dictionary, pad, max_len, verbose)