
import sys
from env.importing import *
from env.project_variables import *
from env.models_dir import *
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from io_.info_print import printing
import toolbox.deep_learning_toolbox as dptx
from tracking.reporting_google_sheet import append_reporting_sheet, update_status

from toolbox.gpu_related import use_gpu_
from toolbox import git_related as gr
from model.bert_tools_from_core_code.tokenization import BertTokenizer

TOKEN_BPE_BERT_START = "[CLS]"
TOKEN_BPE_BERT_SEP = "[SEP]"
PAD_ID_BERT = 0
PAD_BERT = "[PAD]"
NULL_TOKEN_INDEX = 32000
NULL_STR = "[SPACE]"
NULL_STR_TO_SHOW = "_"

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


def get_indexes(list_pretokenized_str, tokenizer, verbose, use_gpu):

    all_tokenized_ls = [tokenizer.tokenize(inp) for inp in list_pretokenized_str]
    tokenized_ls = [tup[0] for tup in all_tokenized_ls]
    aligned_index = [tup[1] for tup in all_tokenized_ls]
    segments_ids = [[0 for _ in range(len(tokenized))] for tokenized in tokenized_ls]

    printing("DATA : bpe tokenized {}", var=[tokenized_ls], verbose=verbose, verbose_level=2)

    ids_ls = [tokenizer.convert_tokens_to_ids(inp) for inp in tokenized_ls]
    max_sent_len = max([len(inp) for inp in tokenized_ls])
    ids_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in ids_ls]
    aligned_index_padded = [[e for e in inp] + [1000 for _ in range(max_sent_len - len(inp))] for inp in aligned_index]
    segments_padded = [inp + [PAD_ID_BERT for _ in range(max_sent_len - len(inp))] for inp in segments_ids]
    mask = [[1 for _ in inp]+[0 for _ in range(max_sent_len - len(inp))] for inp in segments_ids]

    mask = torch.LongTensor(mask)
    tokens_tensor = torch.LongTensor(ids_padded)
    segments_tensors = torch.LongTensor(segments_padded)
    if use_gpu:
        mask = mask.cuda()
        tokens_tensor = tokens_tensor.cuda()
        segments_tensors = segments_tensors.cuda()

    printing("DATA {}", var=[tokens_tensor], verbose=verbose, verbose_level=2)

    sanity_check_data_len(tokens_tensor, segments_tensors, tokenized_ls, aligned_index, raising_error=True)


    return tokens_tensor, segments_tensors, tokenized_ls, aligned_index_padded, mask


def aligned_output(input_tokens_tensor, output_tokens_tensor, input_alignement_with_raw, output_alignement_with_raw,
                   verbose=1):

    output_tokens_tensor_aligned = torch.empty_like(input_tokens_tensor)

    for ind_sent, (_input_alignement_with_raw, _output_alignement_with_raw) in enumerate(zip(input_alignement_with_raw,
                                                                                             output_alignement_with_raw)):
        _i_input = 0
        _i_output = 0
        _1_to_n_token = False
        not_the_end_of_input = True
        output_tokens_tensor_aligned_sent = []

        padded_reached_ind = 0
        while not_the_end_of_input:

            padded_reach = _input_alignement_with_raw[_i_input] == 1000
            if not (padded_reach and len(_output_alignement_with_raw)==_i_output):
                # usual case
                n_to_1_token = _input_alignement_with_raw[_i_input] < _output_alignement_with_raw[_i_output]
                _1_to_n_token = _input_alignement_with_raw[_i_input] > _output_alignement_with_raw[_i_output]
                end_output_with_padded_reach = 0
            else:
                # we reach padding on input and the end on the output
                end_output_with_padded_reach = 1
                n_to_1_token, _1_to_n_token = 0, 0
            # if the otuput token don't change we have to shift the input of one
            if _1_to_n_token:
                print("WARNING : _1_to_n_token --> next batch ")
                break
            if padded_reach and not n_to_1_token:
                # we assert we also reached padding in the output
                # if we are in n_to_1_token it's different maybe not true # same if we reached the end we handle the case with end_output_with_padded_reach
                if len(_output_alignement_with_raw) != _i_output:
                    assert _output_alignement_with_raw[_i_output] == 1000
                padded_reached_ind = 1
            if n_to_1_token:
                appending = NULL_TOKEN_INDEX
                output_tokens_tensor_aligned_sent.append(appending)
            elif not end_output_with_padded_reach:
                appending = output_tokens_tensor[ind_sent, _i_output]
                output_tokens_tensor_aligned_sent.append(appending)
            else:
                output_tokens_tensor_aligned_sent.append(0)
            _i_input += 1
            # padded_reached_ind is to make sure we 're not facing problem in the output
            _i_output += (1 - n_to_1_token - padded_reached_ind )

            if _i_input == len(_input_alignement_with_raw):
                not_the_end_of_input = False

        if _1_to_n_token:
            break
        printing("TO FILL output {} index {}", var=[output_tokens_tensor_aligned_sent, ind_sent], verbose=verbose, verbose_level=3)
        output_tokens_tensor_aligned[ind_sent] = torch.Tensor(output_tokens_tensor_aligned_sent)


    if input_tokens_tensor.is_cuda:
        output_tokens_tensor_aligned = output_tokens_tensor_aligned.cuda()
    return output_tokens_tensor_aligned, _1_to_n_token


def from_bpe_token_to_str(bpe_tensor, topk, pred_mode, tokenizer):
    """
    pred_mode allow to handle gold data also (which only have 2 dim and not three)
    :param bpe_tensor:
    :param topk: int : number of top prediction : will arrange them with all the top1 all the 2nd all the third...
    :param pred_mode: book
    :return:
    """
    predictions_topk_ls = [[[bpe_tensor[sent, word, top].item() if pred_mode else bpe_tensor[sent, word].item()
                             for word in range(bpe_tensor.size(1))] for sent in range(bpe_tensor.size(0))] for top in
                           range(topk)]
    sent_ls_top = [[tokenizer.convert_ids_to_tokens(sent_bpe, special_extra_token=NULL_TOKEN_INDEX,
                                                    special_token_string=NULL_STR)
                    for sent_bpe in predictions_topk] for predictions_topk in predictions_topk_ls]
    if not pred_mode:
        sent_ls_top = sent_ls_top[0]
    return sent_ls_top


def realigne(ls_sent_str, input_alignement_with_raw, remove_null_str=True, remove_extra_predicted_token=False):
    """
    - detokenization of ls_sent_str based on input_alignement_with_raw index
    - we remove paddding and end detokenization at symbol [SEP] that we take as the end of sentence signal
    """

    assert len(ls_sent_str) == len(input_alignement_with_raw), \
        "ls_sent_str {} input_alignement_with_raw {} ".format(len(ls_sent_str), len(input_alignement_with_raw))
    new_sent_ls = []
    for sent, index_ls in zip(ls_sent_str, input_alignement_with_raw):
        assert len(sent) == len(index_ls)
        former_index = -1
        new_sent = []
        former_token = ""
        for _i, (token, index) in enumerate(zip(sent, index_ls)):
            trigger_end_sent = False
            if remove_extra_predicted_token:
                if index == 1000:
                    # we reach the end according to gold data
                    # (this means we just stop looking at the prediciton of the model (we can do that because we assumed word alignement))
                    trigger_end_sent = True
            if token == NULL_STR:
                token = NULL_STR_TO_SHOW if not remove_null_str else ""
            if index == former_index:
                if token.startswith("##"):
                    former_token += token[2:]
                else:
                    former_token += token
            if index != former_index or _i + 1 == len(index_ls):
                new_sent.append(former_token)
                former_token = token
                if trigger_end_sent:
                    break
            if former_token == TOKEN_BPE_BERT_SEP or _i+1 == len(index_ls) :
                new_sent.append(token)
                break
            former_index = index
        new_sent_ls.append(new_sent[1:])
    return new_sent_ls


def word_level_scoring(metric, gold, topk_pred, topk):

    assert metric in ["exact_match"]
    if topk > 1:
        assert metric == "exact_match", "ERROR : only exact_match allows for looking into topk prediction "
    assert len(topk_pred) == topk, "ERROR : inconsinstent provided topk and what I got "
    if metric == "exact_match":
        for pred in topk_pred:
            if gold == pred:
                return 1
        return 0


def agg_func_batch_score(overall_ls_sent_score, agg_func):

    sum_ = sum([score for score_ls in overall_ls_sent_score for score in score_ls])
    n_tokens = sum([1 for score_ls in overall_ls_sent_score for _ in score_ls])
    n_sents = len(overall_ls_sent_score)

    if agg_func == "sum":
        return sum_
    if agg_func == "n_tokens":
        return n_tokens
    if agg_func == "n_sents":
        return n_sents
    if agg_func == "mean":
        return sum_/n_tokens

    if agg_func == "mean_mean_per_sent":
        sum_per_sent = [sum(score_ls) for score_ls in overall_ls_sent_score]
        token_per_sent = [len(score_ls) for score_ls in overall_ls_sent_score]
        sum_mean_per_sent_score = sum([sum_/token_len for sum_, token_len in zip(sum_per_sent, token_per_sent)])
        return sum_mean_per_sent_score/n_sent


def overall_word_level_metric_measure(gold_sent_ls, pred_sent_ls_topk, topk,
                                      metric="exact_match", agg_func_ls=["sum"]):
    """
    'metric' based on a word level comparison of (pred,gold) : e.g : exact_match , edit
    'agg_func' based on a aggrefaiton func to get the overall batch score : e.g : sum
    :param metric:
    :param agg_func:
    :return batch : score, number of token measured
    """
    assert isinstance(agg_func_ls, list)
    assert len(pred_sent_ls_topk) == topk, "ERROR topk not consistent with prediction list " \
        .format(len(pred_sent_ls_topk), topk)
    overall_score_ls_sent = []
    for gold_ind_sent, gold_sent in enumerate(gold_sent_ls):
        # TODO test for all topk
        assert len(gold_sent) == len(pred_sent_ls_topk[0][gold_ind_sent])
        score_sent = []
        for ind_word in range(len(gold_sent)):
            gold_token = gold_sent[ind_word]
            topk_word_pred = [pred_sent_ls_topk[top][gold_ind_sent][ind_word] for top in range(topk)]
            score_sent.append(word_level_scoring(metric=metric, gold=gold_token, topk_pred=topk_word_pred, topk=topk))
        overall_score_ls_sent.append(score_sent)

    result = []
    for agg_func in agg_func_ls:
        result.append({"score": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent, agg_func=agg_func),
                       "agg_func": agg_func,
                       "metric": "exact_match",
                       "n_tokens": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                        agg_func="n_token"),
                       "n_sents": agg_func_batch_score(overall_ls_sent_score=overall_score_ls_sent,
                                                       agg_func="n_sents")})
    return result


def epoch_run(batchIter, tokenizer,
              iter, n_iter_max, bert_with_classifier,
              use_gpu, data_label,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False,
              verbose=0):

    if predict_mode :
        if topk is None:
            topk = 1
            printing("PREDICITON MODE : setting topk to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = True
        if metric is None:
            printing("PREDICITON MODE : setting metric to default 'exact_match' ", verbose_level=1, verbose=verbose)


    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0
    score = 0
    n_tokens = 0
    n_sents = 0

    while True:

        try:
            batch_i += 1

            batch = batchIter.__next__()
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)
            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(batch.raw_input,
                                                                                                                                tokenizer,
                                                                                                                                verbose,
                                                                                                                                use_gpu)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = get_indexes(batch.raw_output,
                                                                                                                                    tokenizer,
                                                                                                                                    verbose,
                                                                                                                                    use_gpu)

            printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                     verbose_level=1, verbose=verbose)

            _verbose = verbose if isinstance(verbose, int) else 0

            if input_tokens_tensor.size(1) != output_tokens_tensor.size(1):
                printing("-------------- Alignement broken", verbose=verbose, verbose_level=2)
                if input_tokens_tensor.size(1) > output_tokens_tensor.size(1):
                    printing("N to 1 like : NOISY splitted MORE than standard", verbose=verbose, verbose_level=2)
                    noisy_over_splitted += 1
                elif input_tokens_tensor.size(1) < output_tokens_tensor.size(1):
                    printing("1 to N : NOISY splitted LESS than standard", verbose=verbose, verbose_level=2)
                    noisy_under_splitted += 1
                    if skip_1_t_n:
                        printing("WE SKIP IT ", verbose=verbose, verbose_level=2)
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
            # aligning output BPE with input (we are rejecting batch with at least one 1 to n case
            # (that we don't want to handle)
            output_tokens_tensor_aligned, _1_to_n_token = aligned_output(input_tokens_tensor, output_tokens_tensor,
                                                                         input_alignement_with_raw,
                                                                         output_alignement_with_raw)

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
            if input_tokens_tensor.is_cuda:
                token_type_ids = token_type_ids.cuda()
            printing("CUDA SANITY CHECK input_tokens:{}  type:{} input_mask:{}  label:{}",
                     var=[input_tokens_tensor.is_cuda, token_type_ids.is_cuda, input_mask.is_cuda,
                          output_tokens_tensor_aligned.is_cuda],
                     verbose=verbose, verbose_level="cuda")
            try:
                _loss = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask, labels=output_tokens_tensor_aligned)
                if predict_mode:
                    logits = bert_with_classifier(input_tokens_tensor, token_type_ids, input_mask)
                    predictions_topk = torch.argsort(logits, dim=-1, descending=True)[:, :, :topk]
                    # from bpe index to string
                    sent_ls_top = from_bpe_token_to_str(predictions_topk, topk, tokenizer=tokenizer, pred_mode=True)
                    gold = from_bpe_token_to_str(output_tokens_tensor_aligned, topk, tokenizer=tokenizer, pred_mode=False)
                    source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer, pred_mode=False)

                    # de BPE-tokenize
                    src_detokenized = realigne(source_preprocessed, input_alignement_with_raw)
                    gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True)
                    pred_detokenized_topk = []
                    for sent_ls in sent_ls_top:
                        pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw, remove_null_str=True,
                                                              remove_extra_predicted_token=True))
                    pdb.set_trace()

                    perf_prediction = overall_word_level_metric_measure(gold_detokenized, pred_detokenized_topk, topk, metric=metric, agg_func_ls=["sum"])
                    score += perf_prediction[0]["score"]
                    n_tokens += perf_prediction[0]["n_tokens"]
                    n_sents += perf_prediction[0]["n_sents"]
                    pdb.set_trace()

                    if print_pred:
                        printing("TRAINING : eval gold {}", var=[gold], verbose=verbose, verbose_level=1)
                        printing("TRAINING : eval pred {}", var=[sent_ls_top], verbose=verbose, verbose_level=1)
                        printing("TRAINING : eval src {}", var=[source_preprocessed], verbose=verbose, verbose_level=1)
                        # TODO : detokenize
                        #  write to conll
                        #  compute prediction score
            except RuntimeError as e:
                print(e)
                pdb.set_trace()

            loss += _loss
            _loss.backward()
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
                mode = "train"
            else:
                mode = "dev"

            if writer is not None:
                writer.add_scalars("loss",
                                    {"loss-{}-bpe".format(mode):
                                     _loss.clone().cpu().data.numpy()}, iter+batch_i)
        except StopIteration:
            break
    printing("Out of {} batch of {} sentences each : {} batch aligned ; {} with at least 1 sentence noisy MORE SPLITTED "
             "; {} with  LESS SPLITTED  (skipped {} ) (CORRUPTED BATCH {}) ",
             var=[batch_i, batch.input_seq.size(0), aligned, noisy_over_splitted, noisy_under_splitted, skip_1_t_n,
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    if predict_mode:
        report = {"score": score/n_tokens, "agg_func":"mean",
                  "subsample": "all", "data": data_label,
                  "metric": metric, "n_tokens": n_tokens,"n_sents": n_sents}
        if writer is not None:
            writer.add_scalars("prediction_score",
                               {"exact_match-all".format(mode):
                                    report["score"]}, iter)

    else:
        report = None
    iter += batch_i
    return loss, iter, report


def setup_repoting_location(model_suffix="", verbose=1):
    model_local_id = str(uuid4())[:5]
    if model_suffix != "":
        model_local_id += "-"+model_suffix
    model_location = os.path.join(CHECKPOINT_BERT_DIR, model_local_id)
    dictionaries = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "dictionaries")
    tensorboard_log = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "tensorboard")
    os.mkdir(model_location)
    printing("CHECKPOINTING model ID:{}", var=[model_local_id], verbose=verbose, verbose_level=1)
    os.mkdir(dictionaries)
    os.mkdir(tensorboard_log)
    printing("CHECKPOINTING {} for checkpoints {} for dictionaries created", var=[model_location, dictionaries], verbose_level=1, verbose=verbose)
    return model_local_id, model_location, dictionaries, tensorboard_log


def run(tasks, train_path, dev_path, n_iter_max_per_epoch,
        voc_tokenizer, auxilliary_task_norm_not_norm, bert_with_classifier,
        report=True, model_suffix="",description="",
        saving_every_epoch=10, lr=0.0001, fine_tuning_strategy="standart",
        debug=False,  batch_size=2, n_epoch=1, verbose=1):

    printing("CHECKPOINTING info : saving model every {}", var=saving_every_epoch, verbose=verbose, verbose_level=1)
    use_gpu = use_gpu_(use_gpu=None, verbose=verbose)

    if use_gpu:
        bert_with_classifier.to("cuda")
    
    if not debug:
        pdb.set_trace = lambda: None

    iter_train = 0
    iter_dev = 0

    model_id, model_location, dict_path, tensorboard_log = setup_repoting_location(model_suffix=model_suffix,verbose=verbose)
    try:
        row, col = append_reporting_sheet(git_id=gr.get_commit_id(), tasks="BERT NORMALIZE",
                                          rioc_job=os.environ.get("OAR_JOB_ID", "no"), description=description,
                                          log_dir=tensorboard_log, target_dir=model_location,
                                          env="?", status="running {}".format("by hand"),
                                          verbose=1)
    except Exception as e:
        print("REPORTING TO GOOGLE SHEET FAILED")
        print(e)
        row = None

    if report:
        writer = SummaryWriter(log_dir=tensorboard_log)
        printing("CHECKPOINTING : starting writing log \ntensorboard --logdir={} host=localhost --port=1234 ",
                 var=[tensorboard_log], verbose_level=1,
                 verbose=verbose)
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
        batchIter_dev = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers_dev, batch_size=batch_size,
                                                           word_dictionary=word_dictionary,
                                                           char_dictionary=char_dictionary,
                                                           pos_dictionary=pos_dictionary,
                                                           word_dictionary_norm=word_norm_dictionary,
                                                           get_batch_mode=False,
                                                           extend_n_batch=1,
                                                           dropout_input=0.0,
                                                           verbose=verbose)
        # TODO add optimizer (if not : devv loss)
        optimizer = dptx.get_optimizer(bert_with_classifier.parameters(), lr=lr)
        bert_with_classifier.train()
        loss_train, iter_train, _ = epoch_run(batchIter_train, tokenizer, data_label=REPO_DATASET[train_path],
                                              bert_with_classifier=bert_with_classifier, writer=writer,iter=iter_train,
                                              optimizer=optimizer, use_gpu=use_gpu,
                                              n_iter_max=n_iter_max_per_epoch, verbose=verbose)

        bert_with_classifier.eval()
        loss_dev, iter_dev, perf_report = epoch_run(batchIter_dev, tokenizer,iter=iter_dev, use_gpu=use_gpu,
                                                    bert_with_classifier=bert_with_classifier, writer=writer,
                                                    predict_mode=True, data_label=REPO_DATASET[dev_path],
                                                    n_iter_max=n_iter_max_per_epoch, verbose=verbose)

        printing("TRAINING : loss train:{} dev:{} for epoch {}  out of {}", var=[loss_train, loss_dev, epoch, n_epoch],
                 verbose=1, verbose_level=1)
        checkpoint_dir = os.path.join(model_location, "{}-ep{}-checkpoint.pt".format(model_id, epoch))

        if epoch % saving_every_epoch == 0 or epoch == (n_epoch-1):
            last_model = ""
            if epoch == (n_epoch-1):
                last_model = "last"
            printing("CHECKPOINT : saving {} model {} ", var=[last_model,checkpoint_dir], verbose=verbose, verbose_level=1)
            torch.save(bert_with_classifier.state_dict(), checkpoint_dir)

    if writer is not None:
        writer.close()
        printing("tensorboard --logdir={} host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1, verbose=verbose)

    if row is not None:
        update_status(row=row, new_status="done ", verbose=1)

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