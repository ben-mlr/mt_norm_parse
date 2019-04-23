from env.importing import *
from pytorch_pretrained_bert import BertForTokenClassification, BertConfig, BertTokenizer

from env.project_variables import *
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from io_.info_print import printing
import toolbox.deep_learning_toolbox as dptx

if False:
    input_ids = torch.LongTensor([[31, 51, 0]])
    input_mask = torch.LongTensor([[1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 0]])
    labels = torch.LongTensor([[0, 32000, 1]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 32001
    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask, labels=labels)
    print("logits original",logits)
    optimizer = dptx.get_optimizer(model.parameters(), lr=0.0001)
    logits.backward()
    optimizer.step()
    optimizer.zero_grad()
    logits = model(input_ids, token_type_ids, input_mask, labels=torch.LongTensor([[0, 1, 1]]))
    print("new logits", logits)

TOKEN_BPE_BERT_START = "[CLS]"
TOKEN_BPE_BERT_SEP = "[SEP]"
PAD_ID_BERT = 0
PAD_BERT = "[PAD]"


def preprocess_batch_string_for_bert(batch):
    #batch = batch[0]
    for i in range(len(batch)):
        batch[i][0] = TOKEN_BPE_BERT_START
        batch[i][-1] = TOKEN_BPE_BERT_SEP
        batch[i] = " ".join(batch[i])
    return batch


def sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True):
    n_sentence = len(tokens_tensor)
    try:
        assert len(segments_tensors) == n_sentence, "ERROR BATCH segments_tensors {} not same len as tokens ids {}".format(segments_tensors, n_sentence)
        assert len(tokenized_ls) == n_sentence, "ERROR BATCH  tokenized_ls {} not same len as tokens ids {}".format(tokenized_ls, n_sentence)
        assert len(aligned_index) == n_sentence, "ERROR BATCH aligned_index {} not same len as tokens ids {}".format(aligned_index, n_sentence)
    except AssertionError as e:
        if raising_error:
            raise(e)
        else:
            print(e)
    for index, segment, token_str, index in zip(tokens_tensor, segments_tensors, tokenized_ls, aligned_index):
        n_token = len(index)
        try:
            #assert len(segment) == n_token, "ERROR sentence {} segment not same len as index {}".format(segment, index)
            assert len(token_str) == n_token, "ERROR sentence {} token_str not same len as index {}".format(token_str, index)
            assert len(index) == n_token, "ERROR sentence {} index not same len as index {}".format(index, index)
        except AssertionError as e:
            if raising_error:
                raise(e)
            else:
                print(e)


def get_indexes(list_pretokenized_str, tokenizer, verbose):

    all_tokenized_ls = [tokenizer.tokenize(inp) for inp in list_pretokenized_str]
    tokenized_ls = [tup[0] for tup in all_tokenized_ls]
    aligned_index = [tup[1] for tup in all_tokenized_ls]
    segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]

    printing("DATA : bpe tokenized {}", var=[tokenized_ls], verbose=verbose, verbose_level=2)

    ids_ls = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_ls]
    max_sent_len = max([len(inp) for inp in tokenized_ls])
    ids_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in ids_ls]
    segments_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in segments_ids]
    mask = [[1 for _ in inp]+[0 for _ in  range(max_sent_len - len(inp))] for inp in segments_ids]

    mask = torch.LongTensor(mask)
    tokens_tensor = torch.LongTensor(ids_padded)
    segments_tensors = torch.LongTensor(segments_padded)

    printing("DATA {}", var=[tokens_tensor], verbose=verbose, verbose_level=2)

    sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True)

    return tokens_tensor, segments_tensors, tokenized_ls, aligned_index, mask


def aligned_output(input_tokens_tensor, output_tokens_tensor, input_alignement_with_raw, output_alignement_with_raw):

    not_the_end_of_input = True
    _i_input = 0
    _i_output = 0
    output_tokens_tensor_aligned = torch.empty_like(input_tokens_tensor)
    _1_to_n_token = False
    for ind_sent, (_input_alignement_with_raw, _output_alignement_with_raw) in enumerate(
            zip(input_alignement_with_raw, output_alignement_with_raw)):
        output_tokens_tensor_aligned_sent = []
        while not_the_end_of_input:
            n_to_1_token = _input_alignement_with_raw[_i_input] < _output_alignement_with_raw[_i_output]
            _1_to_n_token = _input_alignement_with_raw[_i_input] > _output_alignement_with_raw[_i_output]
            # if the otuput token don't change we have to shift the input of one
            if _1_to_n_token:
                print("WARNING : _1_to_n_token --> next batch ")
                break
            if n_to_1_token:
                output_tokens_tensor_aligned_sent.append(NULL_TOKEN_INDEX)
            else:
                output_tokens_tensor_aligned_sent.append(output_tokens_tensor[ind_sent, _i_output])

            _i_input += 1
            _i_output += (1 - n_to_1_token)

            if _i_input == len(_input_alignement_with_raw):
                not_the_end_of_input = False

        if _1_to_n_token:
            break
        output_tokens_tensor_aligned[ind_sent] = torch.Tensor(output_tokens_tensor_aligned_sent)
    return output_tokens_tensor_aligned, _1_to_n_token


def epoch_run(batchIter_train, tokenizer, iter,n_iter_max,bert_with_classifier,
                skip_1_t_n=True,writer=None,optimizer=None,
                verbose=0):

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0

    while True:

        try:
            batch_i += 1
            batch = batchIter_train.__next__()
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)
            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(batch.raw_input, tokenizer, verbose)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = get_indexes(batch.raw_output, tokenizer, verbose)

            printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                     verbose_level=1, verbose=verbose)
            _verbose = verbose

            if input_tokens_tensor.size(1) != output_tokens_tensor.size(1):
                print("-------------- Alignement broken")
                if input_tokens_tensor.size(1) > output_tokens_tensor.size(1):
                    print("N to 1 like : NOISY splitted MORE than standard")
                    noisy_over_splitted += 1
                elif input_tokens_tensor.size(1) < output_tokens_tensor.size(1):
                    print("1 to N : NOISY splitted LESS than standard")
                    noisy_under_splitted += 1
                    if skip_1_t_n:
                        print("WE SKIP IT ")
                        continue
                _verbose += 1
            else:
                aligned += 1

            # logging
            printing("DATA : pre-tokenized input {} ", var=[batch.raw_input], verbose_level=3,
                     verbose=_verbose)
            printing("DATA : BPEtokenized input ids {}", var=[input_tokens_tensor], verbose_level=3,
                     verbose=verbose)

            printing("DATA : pre-tokenized output {} ", var=[batch.raw_output],
                     verbose_level=4,
                     verbose=_verbose)
            printing("DATA : BPE tokenized output ids  {}", var=[output_tokens_tensor],
                     verbose_level=4,
                     verbose=verbose)
            # BPE
            printing("DATA : BPE tokenized input  {}", var=[inp_bpe_tokenized], verbose_level=4,
                     verbose=_verbose)
            printing("DATA : BPE tokenized output  {}", var=[out_bpe_tokenized], verbose_level=4,
                     verbose=_verbose)

            # aligning output BPE with input (we are rejecting batch with at least one 1 to n case (that we don't want to handle)
            output_tokens_tensor_aligned, _1_to_n_token = aligned_output(input_tokens_tensor, output_tokens_tensor,
                                                                         input_alignement_with_raw, output_alignement_with_raw)

            if batch_i == n_iter_max:
                break
            if _1_to_n_token:
                skipping_batch_n_to_1 += 1
                continue
            # CHECKING ALIGNEMENT
            assert output_tokens_tensor_aligned.size(0) == input_tokens_tensor.size(0)
            assert output_tokens_tensor_aligned.size(1) == input_tokens_tensor.size(1)
            # we consider only 1 sentence case
            output_tokens_tensor_aligned = output_tokens_tensor_aligned
            token_type_ids = torch.zeros_like(input_tokens_tensor)

            logits = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask, labels=output_tokens_tensor_aligned)
            loss += logits
            logits.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
                mode = "train"
            else:
                mode = "dev"

            if writer is not None:

                writer.add_scalars("loss",
                                    {"loss-{}-bpe".format(mode):
                                     logits.clone().cpu().data.numpy()}, iter+batch_i)
        except StopIteration:
            break

    printing("Out of {} batch of {} sentences each : {} batch aligned ; {} with at least 1 sentence noisy MORE SPLITTED "
             "; {} with  LESS SPLITTED  (skipped {} ) (CORRUPTED BATCH {}) ",
             var=[batch_i, batch.input_seq.size(0), aligned, noisy_over_splitted, noisy_under_splitted, skip_1_t_n,
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    iter += batch_i
    return loss, iter


def setup_repoting_location(model_suffix="", verbose=1):
    model_local_id = str(uuid4())[:5]
    if model_suffix != "":
        model_local_id += "-"+model_suffix
    model_location = os.path.join(CHECKPOINT_BERT_DIR, model_local_id)
    dictionaries = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "dictionaries")
    tensorboard_log = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "tensorboard")
    os.mkdir(model_location)
    os.mkdir(dictionaries)
    os.mkdir(tensorboard_log)
    printing("CHECKPOINTING {} for checkpoints {} for dictionaries created", var=[model_location, dictionaries], verbose_level=1, verbose=verbose)
    return model_local_id, model_location, dictionaries, tensorboard_log


def run(tasks,  train_path, dev_path, n_iter_max_per_epoch,
        voc_tokenizer, auxilliary_task_norm_not_norm,bert_with_classifier,
        report=True,model_suffix="",
        debug=False, use_gpu=False, batch_size=2, n_epoch=1, verbose=1):

    if not debug:
        pdb.set_trace = lambda: None

    iter_train = 0
    iter_dev = 0

    model_id, model_location, dict_path, tensorboard_log = setup_repoting_location(model_suffix=model_suffix,verbose=verbose)
    if report:
        printing("CHECKPOINTING : tensorboard logs will be held {}", var=[tensorboard_log], verbose=verbose, verbose_level=1)
        writer = SummaryWriter(log_dir=tensorboard_log)
    else:
        writer = None

    # build or make dictionaries
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              train_path=train_path,
                              dev_path=dev_path,
                              test_path=None,
                              word_embed_dict={},
                              dry_run=False,
                              word_normalization=True,
                              force_new_dic=True,
                              tasks=tasks,
                              add_start_char=1, verbose=1)

    # load , mask, bucket and index data
    readers_train = readers_load(datasets=train_path, tasks=tasks, word_dictionary=word_dictionary,
                                 word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                                 pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                                 type_dictionary=type_dictionary, use_gpu=use_gpu,
                                 norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                                 add_start_char=1, add_end_char=1, symbolic_end=1,
                                 symbolic_root=1, bucket=True, max_char_len=20,
                                 verbose=verbose)
    readers_dev = readers_load(datasets=dev_path, tasks=tasks, word_dictionary=word_dictionary,
                               word_dictionary_norm=word_norm_dictionary, char_dictionary=char_dictionary,
                               pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                               type_dictionary=type_dictionary, use_gpu=use_gpu,
                               norm_not_norm=auxilliary_task_norm_not_norm, word_decoder=True,
                               add_start_char=1, add_end_char=1,
                               symbolic_end=1, symbolic_root=1, bucket=True, max_char_len=20,
                               verbose=verbose)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(voc_tokenizer)

    for epoch in range(n_epoch):
        # build iterator on the loaded data
        batchIter_train = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                             word_dictionary=word_dictionary,
                                                             char_dictionary=char_dictionary,
                                                             pos_dictionary=pos_dictionary,
                                                             word_dictionary_norm=word_norm_dictionary,
                                                             get_batch_mode=False,
                                                             extend_n_batch=1,
                                                             dropout_input=0.0,
                                                             verbose=verbose)
        batchIter_dev = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_train, batch_size=batch_size,
                                                           word_dictionary=word_dictionary,
                                                           char_dictionary=char_dictionary,
                                                           pos_dictionary=pos_dictionary,
                                                           word_dictionary_norm=word_norm_dictionary,
                                                           get_batch_mode=False,
                                                           extend_n_batch=1,
                                                           dropout_input=0.0,
                                                           verbose=verbose)
        # TODO add optimizer (if not : devv loss)
        optimizer = dptx.get_optimizer(bert_with_classifier.parameters(), lr=0.0001)
        bert_with_classifier.train()
        loss_train, iter_train = epoch_run(batchIter_train, tokenizer,
                               bert_with_classifier=bert_with_classifier, writer=writer,iter=iter_train,
                               optimizer=optimizer,
                               n_iter_max=n_iter_max_per_epoch, verbose=verbose)

        bert_with_classifier.eval()
        loss_dev, iter_dev = epoch_run(batchIter_dev, tokenizer,iter=iter_dev,
                             bert_with_classifier=bert_with_classifier, writer=writer,
                             n_iter_max=n_iter_max_per_epoch, verbose=verbose)

        printing("TRAINING : loss train:{} dev:{} for epoch {}  out of {}", var=[loss_train, loss_dev, epoch, n_epoch],
                 verbose=1, verbose_level=1)
        checkpoint_dir = os.path.join(model_location, "{}-ep{}-checkpoint.pt".format(model_id, epoch))
        printing("CHECKPOINT : saving model {} ",var=[checkpoint_dir], verbose=verbose, verbose_level=1)
        saving_every_epoch = 10
        if epoch % saving_every_epoch == 0:
            torch.save(model.state_dict(), checkpoint_dir)
    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1, verbose=verbose)

train_path = [LIU_TRAIN]
dev_path = [TEST]
tasks = ["normalize"]
use_gpu = False
dict_path = "../dictionaries"

if True:
    voc_tokenizer = "/Users/bemuller/Documents/Work/INRIA/dev/representation/lm/bert_models/bert-base-cased-vocab.txt"

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 32001
    NULL_TOKEN_INDEX = 32000
    initialize_bpe_layer = False
    model = BertForTokenClassification(config, num_labels)
    if initialize_bpe_layer:
        output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))), dim=0)
        model.classifier.weight = nn.Parameter(output_layer)

    run(bert_with_classifier=model,
        voc_tokenizer=voc_tokenizer, tasks=tasks, train_path=train_path, dev_path=dev_path, use_gpu=use_gpu,
        auxilliary_task_norm_not_norm=True,
        batch_size=1, n_iter_max_per_epoch=2, n_epoch=50,
        model_suffix="init",
        debug=False, report=True,
        verbose=1)


    # WE ADD THE NULL TOKEN THAT WILL CORRESPOND TO the bpe_embedding_layer.size(0) index
    #NULL_TOKEN_INDEX = bpe_embedding_layer.size(0)


    #pdb.set_trace()
#logits = model(input_ids, token_type_ids, input_mask, labels=torch.LongTensor([[0, 1, 1]]))
#model.bert.embeddings.word_embeddings.weight
# DEFINE ITERATOR

# TRAIN
## - iterate on data
## - tokenize for BERT
## - index for BERT
## - FEED INTO BERT
## FREEZE BERT FIRST
## - COMPTUTE LOSS AND BACKPROP
## REPORT IN TENSORBOARD
## DONE


# THEN
## ADD EARLY STOPPING
## GRADUAL UNFREEZING