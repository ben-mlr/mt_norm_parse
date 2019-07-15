from io_.info_print import printing
from env.importing import edit_distance, np, pdb, time
from io_.dat.constants import SPECIAL_TOKEN_LS
from env.project_variables import HEURISTICS


def get_letter_indexes(list_ordered_words, new_letter=None):
    if new_letter is None:
        new_letter = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", 'q', 'r', "s",
                      "t", "u", "v", "w", "x", "y", "z", "-"]
    dic_ind = {}
    ind_letter = 0
    ind_former_1 = 0
    len_max = len(list_ordered_words)
    letter_former = "?"
    for ind, word in enumerate(list_ordered_words):
        if word.startswith(new_letter[ind_letter]):
            dic_ind[letter_former] = [ind_former_1, ind]
            # for managing big list
            # if letter_former == "j" and False:
            #    dic_ind["y"] = [ind_former_1, ind]
            ind_former_1 = ind
            letter_former = new_letter[ind_letter]
            ind_letter += 1
    dic_ind[letter_former] = [ind_former_1, len_max]
    dic_ind.pop("?")
    return dic_ind


def predict_with_heuristic(src_detokenized, pred_detokenized_topk, heuristic_ls,
                           gold_detokenized=None, slang_dic=None,
                           list_reference=None, index_alphabetical_order=None, list_candidates=None,
                           threshold_edit=None, edit_module_pred_need_norm_only=True,
                           verbose=1):
    """
    applies heuristic based on src data to correct prediction
    if gold_detection assumed : then it uses gold data also
    #TODO : shoul dbe made more efficient
    :param src_detokenized:
    :param pred_detokenized_topk:
    :param heuristic_ls:
    :param gold_detokenized:
    :param verbose:
    :return:
    """

    printing("TRAINING : postprocessing predictions with extra heuristics {} ", var=[heuristic_ls], verbose=verbose, verbose_level=2)
    assert len(list(set(heuristic_ls))) == len(heuristic_ls), "ERROR redudancies in {}".format(heuristic_ls)
    assert len(list(set(heuristic_ls) & set(HEURISTICS)) ) == len(heuristic_ls), \
        "ERROR heuristic {} does not match {}".format(HEURISTICS, heuristic_ls)
    if "gold_detection" in heuristic_ls:
        assert gold_detokenized is not None
    else:
        assert gold_detokenized is None, "ERROR : e don't want gold data here if no need"
    for ind_top, pred_sent in enumerate(pred_detokenized_topk):
        if "gold_detection" in heuristic_ls:
            assert len(pred_sent) == len(gold_detokenized)
        assert len(src_detokenized) == len(pred_sent)
        for sent in range(len(pred_sent)):
            len_sent = len(pred_sent[sent])
            for ind_token in range(len_sent):
                for heuristic in heuristic_ls:
                    if heuristic == "gold_detection" and src_detokenized[sent][ind_token] == gold_detokenized[sent][ind_token]:
                        printing("replace {} {} ", var=[pred_detokenized_topk[ind_top][sent][ind_token],
                                                        gold_detokenized[sent][ind_token]],
                                 verbose=verbose, verbose_level=5)
                        pred_detokenized_topk[ind_top][sent][ind_token] = src_detokenized[sent][ind_token]

                    if heuristic.startswith("edit_check"):
                        assert list_reference is not None and index_alphabetical_order is not None, "ERROR list_reference should not be None as heuristic is {}".format(heuristic)
                        src = src_detokenized[sent][ind_token]
                        pred = pred_detokenized_topk[ind_top][sent][ind_token]

                        def edit_check(list_reference,  src, pred, list_candidates, threshold_edit,
                                       pred_need_norm_only=True):
                            assert list_candidates is not None, "ERROR list_candidates {} ".format(list_candidates)

                            if pred not in list_reference and pred not in SPECIAL_TOKEN_LS \
                                and src not in SPECIAL_TOKEN_LS and len(pred) > 0 \
                                and pred[0] in "abcdefghijklmnopqrstuvwxyz":

                                if pred_need_norm_only and src == pred:
                                    print("EDIT CHECK (pred_need_norm_only is TRUE) "
                                          "on src {} : predicted a NORMED so not editing ".format(src))
                                    return pred
                                # NB : THIS edit check only involve tokens
                                #      that are normalized by the model and not  # , @ , numbers won't
                                if src in list_reference:
                                    return src
                                else:
                                    start = time.time()
                                    try:
                                        index_letter_start = index_alphabetical_order[src[0]][0]
                                        index_letter_end = index_alphabetical_order[src[0]][1]
                                        _list_reference = list_candidates[index_letter_start:index_letter_end]
                                        if len(_list_reference) == 0:
                                            _list_reference = list_candidates
                                        pred_to_ref = np.array([edit_distance(pred, ref) for ref in _list_reference])
                                        src_to_ref = np.array([edit_distance(src, ref) for ref in _list_reference])

                                        index_min_pred_to_ref = np.argmin(pred_to_ref)
                                        inde_min_src_to_ref = np.argmin(src_to_ref)

                                        min_pred = _list_reference[index_min_pred_to_ref]
                                        min_src = _list_reference[inde_min_src_to_ref]
                                        pdb.set_trace()
                                        if pred_to_ref[index_min_pred_to_ref] >= threshold_edit \
                                                and src_to_ref[inde_min_src_to_ref] >= threshold_edit:
                                            print("EDIT CHECK : edit module did not find close enough candidate "
                                                  "so not modifying prediciton for {} : pred {}".format(src, pred))
                                            return pred
                                        norm_edit_pred = pred_to_ref[index_min_pred_to_ref]/len(pred)
                                        norm_edit_src = src_to_ref[inde_min_src_to_ref]/len(src)

                                        pdb.set_trace()
                                        # TODO : if distance is too high should return pred anyway
                                        predict = min_pred if norm_edit_pred < norm_edit_src else min_src
                                        print("EDIT CHECK on {} mode  src {} "
                                              "-> {} , pred {} -> {} : {} ".format(heuristic, src, min_src, pred, min_pred, predict))
                                        print("EDIT CHECK done in {:0.2f}s".format(time.time()-start))
                                    except Exception as e:
                                        print("ERROR (heuristics) : {} src was {}".format(e, src))
                                        return pred
                                    return predict
                                # find the edit the closest in
                            else:
                                if len(pred) == 0:
                                    print("WARNING : pred {} is empty".format(pred))
                                return pred
                        pred_detokenized_topk[ind_top][sent][ind_token] = edit_check(list_reference, src, pred,
                                                                                     list_candidates=list_candidates,
                                                                                     threshold_edit=threshold_edit,
                                                                                     pred_need_norm_only=edit_module_pred_need_norm_only)

                    if heuristic == "@" and src_detokenized[sent][ind_token].startswith("@"):

                        pred_detokenized_topk[ind_top][sent][ind_token] = src_detokenized[sent][ind_token]
                    if heuristic == "#" and src_detokenized[sent][ind_token].startswith("#"):
                        pred_detokenized_topk[ind_top][sent][ind_token] = src_detokenized[sent][ind_token]
                    if heuristic == "url" and (src_detokenized[sent][ind_token].startswith("http") or src_detokenized[sent][ind_token].startswith(".com")):
                        pred_detokenized_topk[ind_top][sent][ind_token] = src_detokenized[sent][ind_token]

                    if heuristic == "slang_translate":
                        assert slang_dic is not None, "ERROR slang_dic  required as heuristic {}".format(heuristic)
                        if src_detokenized[sent][ind_token] in slang_dic:
                            pred_detokenized_topk[ind_top][sent][ind_token] = slang_dic[src_detokenized[sent][ind_token]]


    return pred_detokenized_topk
