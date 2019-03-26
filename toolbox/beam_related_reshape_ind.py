import pdb


def get_beam_ind_token_ind(ind_flatted_ls, first_dim_in_view):
    first_ind = ind_flatted_ls/first_dim_in_view
    second_ind = ind_flatted_ls - (ind_flatted_ls / first_dim_in_view)*first_dim_in_view
    return first_ind, second_ind

def DEPRE_get_beam_ind_token_ind(ind_flatted_ls, first_dim_in_view):
    first_ind = ind_flatted_ls / first_dim_in_view
    second_ind = ind_flatted_ls - (ind_flatted_ls / first_dim_in_view) * first_dim_in_view
    return first_ind, second_ind
# get the predictions and update the output_seq foe each beam


def update_output_seq(output_seq_, token_pred_id_cand, beam_id_cand,log_scores_ranked_former_all_seq, char_decode_step,log_score_best):
        output_seq_1 = output_seq_.clone()
        log_scores_ranked_former_all_seq_1 = log_scores_ranked_former_all_seq.clone()
        for sent in range(output_seq_.size(0)):
            for word in range(output_seq_.size(1)):
                for ind_new_beam in range(output_seq_.size(3)):
                    #beam_id_cand[sent, word, beam]
                    #pdb.set_trace()
                    beam = beam_id_cand[sent, word, ind_new_beam]
                    # we set the new token prediction
                    if int(beam) != ind_new_beam:
                        pdb.set_trace()
                        pdb.set_trace()
                        output_seq_1[sent, word, char_decode_step - 2, ind_new_beam] = output_seq_1[sent, word, char_decode_step - 2, beam]
                        log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 2, ind_new_beam] = log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 2, beam]
                        #pdb.set_trace()
                    #pdb.set_trace()
                    # We update the former step of the new beam ind_new_beam with the ones of the beam we decode
                    output_seq_1[sent, word, char_decode_step - 1, ind_new_beam] = token_pred_id_cand[sent, word, ind_new_beam]
                    log_scores_ranked_former_all_seq_1[sent, word, char_decode_step - 1, ind_new_beam] = log_score_best[sent, word, ind_new_beam]

        return output_seq_1, log_scores_ranked_former_all_seq_1

