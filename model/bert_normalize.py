from env.importing import *
from env.project_variables import *
from env.models_dir import *
from io_.data_iterator import readers_load, conllu_data, data_gen_multi_task_sampling_batch
from io_.info_print import printing
import toolbox.deep_learning_toolbox as dptx
from tracking.reporting_google_sheet import append_reporting_sheet, update_status

from toolbox.gpu_related import use_gpu_
from toolbox import git_related as gr
from toolbox.sanity_check import sanity_check_data_len

from model.bert_tools_from_core_code.tokenization import BertTokenizer

from io_.dat.normalized_writer import write_conll
from io_.dat.constants import TOKEN_BPE_BERT_SEP, TOKEN_BPE_BERT_START, PAD_ID_BERT, PAD_BERT
from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes
from io_.bert_iterators_tools.alignement import aligned_output, realigne
from evaluate.scoring.report import overall_word_level_metric_measure

def epoch_run(batchIter, tokenizer,
              iter, n_iter_max, bert_with_classifier, epoch,
              use_gpu, data_label, null_token_index, null_str,
              skip_1_t_n=True,
              writer=None, optimizer=None,
              predict_mode=False, topk=None, metric=None,
              print_pred=False,
              writing_pred=False, dir_end_pred=None,
              verbose=0):

    if predict_mode:
        if topk is None:
            topk = 1
            printing("PREDICITON MODE : setting topk to default 1 ", verbose_level=1, verbose=verbose)
        print_pred = True
        if metric is None:
            metric = "exact_match"
            printing("PREDICITON MODE : setting metric to default 'exact_match' ", verbose_level=1, verbose=verbose)

    if writing_pred:
        assert dir_end_pred is not None
        dir_normalized = os.path.join(dir_end_pred, "{}_ep-prediction.conll".format(epoch))
        dir_normalized_original_only = os.path.join(dir_end_pred, "{}_ep-prediction_src.conll".format(epoch))
        dir_gold = os.path.join(dir_end_pred, "{}_ep-gold.conll".format(epoch))
        dir_gold_original_only = os.path.join(dir_end_pred, "{}_ep-gold_src.conll".format(epoch))

    batch_i = 0
    noisy_over_splitted = 0
    noisy_under_splitted = 0
    aligned = 0
    skipping_batch_n_to_1 = 0

    loss = 0
    score = 0
    n_tokens = 0
    n_sents = 0
    skipping_evaluated_batch = 0
    mode = "?"
    new_file = True
    while True:

        try:
            batch_i += 1

            batch = batchIter.__next__()
            batch.raw_input = preprocess_batch_string_for_bert(batch.raw_input)
            batch.raw_output = preprocess_batch_string_for_bert(batch.raw_output)

            pdb.set_trace()

            input_tokens_tensor, input_segments_tensors, inp_bpe_tokenized, input_alignement_with_raw, input_mask = get_indexes(batch.raw_input, tokenizer, verbose, use_gpu)
            output_tokens_tensor, output_segments_tensors, out_bpe_tokenized, output_alignement_with_raw, output_mask = get_indexes(batch.raw_output,
                                                                                                                                    tokenizer,
                                                                                                                                    verbose,
                                                                                                                                    use_gpu)

            printing("DATA dim : {} input {} output ", var=[input_tokens_tensor.size(), output_tokens_tensor.size()],
                     verbose_level=2, verbose=verbose)

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
                                                                         output_alignement_with_raw,
                                                                         null_token_index=null_token_index)

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

                    sent_ls_top = from_bpe_token_to_str(predictions_topk, topk, tokenizer=tokenizer, pred_mode=True,
                                                        null_token_index=null_token_index, null_str=null_str)
                    gold = from_bpe_token_to_str(output_tokens_tensor_aligned, topk, tokenizer=tokenizer,
                                                 pred_mode=False, null_token_index=null_token_index,null_str=null_str)
                    source_preprocessed = from_bpe_token_to_str(input_tokens_tensor, topk, tokenizer=tokenizer,
                                                                pred_mode=False, null_token_index=null_token_index,null_str=null_str)

                    # de-BPE-tokenize
                    src_detokenized = realigne(source_preprocessed, input_alignement_with_raw, null_str=null_str)
                    gold_detokenized = realigne(gold, input_alignement_with_raw, remove_null_str=True,null_str=null_str)
                    pred_detokenized_topk = []
                    for sent_ls in sent_ls_top:
                        pred_detokenized_topk.append(realigne(sent_ls, input_alignement_with_raw, remove_null_str=True,
                                                              remove_extra_predicted_token=True,null_str=null_str))

                    if writing_pred:

                        write_conll(format="conll", dir_normalized=dir_normalized,
                                    dir_original=dir_normalized_original_only,
                                    src_text_ls=src_detokenized,
                                    text_decoded_ls=pred_detokenized_topk[0], pred_pos_ls=None, src_text_pos=None,
                                    tasks=["normalize"], ind_batch=iter+batch_i, new_file=new_file,
                                    verbose=verbose)
                        write_conll(format="conll", dir_normalized=dir_gold, dir_original=dir_gold_original_only,
                                    src_text_ls=src_detokenized,
                                    text_decoded_ls=gold_detokenized, pred_pos_ls=None, src_text_pos=None,
                                    tasks=["normalize"], ind_batch=iter + batch_i, new_file=new_file, verbose=verbose)
                        new_file = False

                    perf_prediction, skipping = overall_word_level_metric_measure(gold_detokenized, pred_detokenized_topk, topk, metric=metric, agg_func_ls=["sum"])
                    skipping_evaluated_batch += skipping
                    score += perf_prediction[0]["score"]
                    n_tokens += perf_prediction[0]["n_tokens"]
                    n_sents += perf_prediction[0]["n_sents"]

                    if print_pred:
                        printing("TRAINING : Score : {} / {} tokens / {} sentences", var=[perf_prediction[0]["score"],
                                                                                          perf_prediction[0]["n_tokens"],
                                                                                          perf_prediction[0]["n_sents"]],
                                 verbose=verbose, verbose_level=1)
                        printing("TRAINING : eval gold {}-{} {}", var=[iter, batch_i, gold_detokenized], verbose=verbose,
                                 verbose_level=1)
                        printing("TRAINING : eval pred {}-{} {}", var=[iter, batch_i, pred_detokenized_topk], verbose=verbose,
                                 verbose_level=1)
                        printing("TRAINING : eval src {}-{} {}", var=[iter, batch_i, src_detokenized],
                                 verbose=verbose, verbose_level=1)
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
                print("MODE data {} optimizing".format(data_label))
            else:
                mode = "dev"
                print("MODE data {} optimizing".format(data_label))

            if writer is not None:
                writer.add_scalars("loss",
                                    {"loss-{}-bpe".format(mode):
                                     _loss.clone().cpu().data.numpy()}, iter+batch_i)
        except StopIteration:
            break

    printing("WARNING on {} : Out of {} batch of {} sentences each : {} batch aligned ; {} with at least 1 sentence "
             "noisy MORE SPLITTED "
             "; {} with  LESS SPLITTED {} + BATCH with skipped_1_to_n : {} ",
             var=[data_label, batch_i, batch.input_seq.size(0), aligned, noisy_over_splitted, noisy_under_splitted,
                  "SKIPPED" if skip_1_t_n else "",
                  skipping_batch_n_to_1],
             verbose=verbose, verbose_level=0)
    printing("WARNING on {} ON THE EVALUATION SIDE we skipped extra {} batch ", var=[data_label,skipping_evaluated_batch], verbose_level=1, verbose=1)
    if predict_mode:
        try:
            report = {"score": score/n_tokens, "agg_func": "mean",
                      "subsample": "all", "data": data_label,
                      "metric": metric, "n_tokens": n_tokens,"n_sents": n_sents}
        except Exception as e:
            print(e)
            report = []
        if writer is not None:
            writer.add_scalars("prediction_score",
                               {"exact_match-all-{}".format(mode):
                                    report["score"]}, epoch)

    else:
        report = None
    iter += batch_i
    return loss, iter, report


def setup_repoting_location(model_suffix="", verbose=1):
    """
    create an id for a model and locations for checkpoints, dictionaries, tensorboard logs, data
    :param model_suffix:
    :param verbose:
    :return:
    """
    model_local_id = str(uuid4())[:5]
    if model_suffix != "":
        model_local_id += "-"+model_suffix
    model_location = os.path.join(CHECKPOINT_BERT_DIR, model_local_id)
    dictionaries = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "dictionaries")
    tensorboard_log = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "tensorboard")
    end_predictions = os.path.join(CHECKPOINT_BERT_DIR, model_local_id, "end_predictions")
    os.mkdir(model_location)
    printing("CHECKPOINTING model ID:{}", var=[model_local_id], verbose=verbose, verbose_level=1)
    os.mkdir(dictionaries)
    os.mkdir(tensorboard_log)
    os.mkdir(end_predictions)
    printing("CHECKPOINTING \n- {} for checkpoints \n- {} for dictionaries created \n- {} predictions",
             var=[model_location, dictionaries, end_predictions], verbose_level=1, verbose=verbose)
    return model_local_id, model_location, dictionaries, tensorboard_log, end_predictions


def run(tasks, train_path, dev_path, n_iter_max_per_epoch,
        voc_tokenizer, auxilliary_task_norm_not_norm, bert_with_classifier,
        null_token_index, null_str,
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

    model_id, model_location, dict_path, tensorboard_log, end_predictions = setup_repoting_location(model_suffix=model_suffix,verbose=verbose)
    try:
        row, col = append_reporting_sheet(git_id=gr.get_commit_id(), tasks="BERT NORMALIZE",
                                          rioc_job=os.environ.get("OAR_JOB_ID", "no"), description=description,
                                          log_dir=tensorboard_log, target_dir=model_location,
                                          env=os.environ.get("ENV", "local"), status="running",
                                          verbose=1)
    except Exception as e:
        print("REPORTING TO GOOGLE SHEET FAILED")
        print(e)
        row = None

    if report:
        writer = SummaryWriter(log_dir=tensorboard_log)
        printing("CHECKPOINTING : starting writing log \ntensorboard --logdir={} --host=localhost --port=1234 ",
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

    try:
        for epoch in range(n_epoch):

            checkpointing_model_data = (epoch % saving_every_epoch == 0 or epoch == (n_epoch - 1))
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
            train_data_label = "|".join([REPO_DATASET[_train_path] for _train_path in train_path])
            dev_data_label = "|".join([REPO_DATASET[_dev_path] for _dev_path in dev_path])

            loss_train, iter_train, perf_report_train = epoch_run(batchIter_train, tokenizer,
                                                                  data_label=train_data_label,
                                                                  bert_with_classifier=bert_with_classifier, writer=writer,
                                                                  iter=iter_train, epoch=epoch,
                                                                  writing_pred=checkpointing_model_data, dir_end_pred=end_predictions,
                                                                  optimizer=optimizer, use_gpu=use_gpu, predict_mode=True,
                                                                  null_token_index=null_token_index,null_str=null_str,
                                                                  n_iter_max=n_iter_max_per_epoch, verbose=verbose)

            bert_with_classifier.eval()
            loss_dev, iter_dev, perf_report_dev = epoch_run(batchIter_dev, tokenizer,
                                                            iter=iter_dev, use_gpu=use_gpu,
                                                            bert_with_classifier=bert_with_classifier, writer=writer,
                                                            writing_pred=checkpointing_model_data, dir_end_pred=end_predictions,
                                                            predict_mode=True, data_label=dev_data_label, epoch=epoch,
                                                            null_token_index=null_token_index,null_str=null_str,
                                                            n_iter_max=n_iter_max_per_epoch, verbose=verbose)

            printing("PERFORMANCE {} TRAIN", var=[epoch, perf_report_train],verbose=verbose, verbose_level=1)
            printing("PERFORMANCE {} DEV", var=[epoch, perf_report_dev], verbose=verbose, verbose_level=1)

            printing("TRAINING : loss train:{} dev:{} for epoch {}  out of {}", var=[loss_train, loss_dev, epoch, n_epoch], verbose=1, verbose_level=1)
            checkpoint_dir = os.path.join(model_location, "{}-ep{}-checkpoint.pt".format(model_id, epoch))

            if checkpointing_model_data :
                last_model = ""
                if epoch == (n_epoch-1):
                    last_model = "last"
                printing("CHECKPOINT : saving {} model {} ", var=[last_model, checkpoint_dir], verbose=verbose,
                         verbose_level=1)
                torch.save(bert_with_classifier.state_dict(), checkpoint_dir)

        if writer is not None:
            writer.close()
            printing("tensorboard --logdir={} --host=localhost --port=1234 ", var=[tensorboard_log], verbose_level=1,
                     verbose=verbose)
        print("PERFORMANCE LAST {} TRAIN".format(epoch), perf_report_train)
        print("PERFORMANCE LAST {} DEV".format(epoch), perf_report_dev)
        if row is not None:
            update_status(row=row, new_status="done ", verbose=1)
        print("DONE")
    except Exception as e:
        if row is not None:
            update_status(row=row, new_status="ERROR", verbose=1)
        raise(e)
    # WE ADD THE NULL TOKEN THAT WILL CORRESPOND TO the bpe_embedding_layer.size(0) index
    #NULL_TOKEN_INDEX = bpe_embedding_layer.size(0)


def get_bert_token_classification(vocab_size,
                                   pretrained_model_dir=None, checkpoint_dir=None,
                                   initialize_bpe_layer=None, verbose=1):
    """
    two use case :
    - initialize bert based on pretrained_model_dir and add a token prediction module based or not on initialize_bpe_layer
    - reload from checkpoint bert+tokenclassification
    :param vocab_size:
    :param pretrained_model_dir:
    :param checkpoint_dir:
    :param initialize_bpe_layer:
    :param verbose:
    :return:
    """
    assert checkpoint_dir is not None or pretrained_model_dir is not None, \
        "Neither checkpoint_dir or pretrained_model_dir was provided"
    assert pretrained_model_dir is None or checkpoint_dir is None, \
        "Only one of checkpoint_dir or pretrained_model_dir should be provided "

    config = BertConfig(vocab_size_or_config_json_file=vocab_size, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    # QUESTION : WHERE IS THE MODEL ACTUALLY BEING LOADED ???
    num_labels = vocab_size + 1
    model = BertForTokenClassification(config, num_labels)

    if pretrained_model_dir is not None:
        assert initialize_bpe_layer is not None, "ERROR initialize_bpe_layer should not be None "
        model = model.from_pretrained(pretrained_model_dir, num_labels=num_labels)
        printing("MODEL : loading pretrained BERT and adding extra module for token classification based on {}",
                 var=[pretrained_model_dir],
                 verbose=verbose,
                 verbose_level=1)
        if initialize_bpe_layer:
            output_layer = torch.cat((model.bert.embeddings.word_embeddings.weight.data, torch.rand((1, 768))),
                                     dim=0)
            model.classifier.weight = nn.Parameter(output_layer)
            printing("MODEL : initializing output layer with embedding layer + extra token ",
                     verbose=verbose,
                     verbose_level=1)
    elif checkpoint_dir is not None:
        assert initialize_bpe_layer is None, "ERROR initialize_bpe_layer should b None as loading from existing checkpoint"
        model.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))
        printing("MODEL : loading model BERT+token classification pretrained from checkpoint {}",
                 var=[checkpoint_dir],
                 verbose=verbose,
                 verbose_level=1)

    return model

