
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_cumulated_list(sent_len):
    sent_len_cumulated = [0]
    cumu = 0
    for len_sent in sent_len:
        cumu += int(len_sent)
        sent_len_cumulated.append(cumu)
    return sent_len_cumulated