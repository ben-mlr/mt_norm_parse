from predict.prediction_batch import  greedy_decode_batch
from predict.prediction_string import decode_seq_str, decode_interacively
import pdb
from model.loss import LossCompute
import os
from io_.info_print import printing
from model.generator import Generator
MAX_LEN = 20


def interact(dic_path, model_full_name,
             dir_model, debug=False, model_specific_dictionary=True,
             beam_size=2,
             word_decoding=False,
             extra_arg_specific_label="",
             save_attention=False, show_attention=False, beam_decode=False,
             verbose=2):
    from model.seq2seq import LexNormalizer

    assert model_specific_dictionary
    char_dictionary = None
    voc_size = None

    if not debug:
        pdb.set_trace = lambda: 1
    model = LexNormalizer(generator=Generator,
                          voc_size=voc_size,
                          load=True, model_full_name=model_full_name,
                          model_specific_dictionary=model_specific_dictionary,
                          dict_path=dic_path,
                          dir_model=dir_model,
                          extra_arg_specific_label=extra_arg_specific_label,
                          word_decoding=word_decoding, char_decoding=not word_decoding,
                          verbose=verbose)
    model.eval()
    if show_attention or save_attention:
        assert model.decoder.attn_layer is not None, "ERROR : no attention to plot "
    if save_attention:
        dir_attention = os.path.join(dir_model, "attention_plot")
        if os.path.isdir(dir_attention):
            info = "existing"
        else:
            os.mkdir(dir_attention)
            info = "created"
        printing("Saving to {} {}", var=[info, dir_attention], verbose_level=1, verbose=verbose)
    else:
        dir_attention = None
    decode_interacively(max_len=MAX_LEN, model=model, char_dictionary=char_dictionary, sent_mode=True,
                        dir_attention=dir_attention, save_attention=save_attention, show_attention=show_attention,
                        beam_decode=beam_decode, beam_size=beam_size,
                        verbose=verbose)
    # /1eeb9-WARMUP-unrolling-False0-model_1-model_1_fd8c-folder/dictionaries/
    #1eeb9-WARMUP-unrolling-False0-model_1-model_1_fd8c-folder-50-False_get_batchNEW-repo-folder

import io
import torchvision
from PIL import Image
import socket
import visdom
import torch

import matplotlib.pyplot as plt


def show_plot_visdom():
    vis = visdom.Visdom()
    buf = io.BytesIO()
    print("buf", buf)
    plt.savefig(buf)
    buf.seek(0)
    hostname = socket.gethostname()
    print("host : ", hostname)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), win=attn_win, opts={'title': attn_win})



if __name__ == "__main__":
    
    debug = False
    model_specific_dictionary = True
    script_dir = os.path.dirname(os.path.realpath(__file__))
    list_all_dir = os.listdir("../checkpoints/")
    #f178-DROPOUT_EVEN_INCREASE-0.1-to_sent+word+bridge_out-model_3_046c-folder
    #for ablation_id in ["aaad","bd55","0153","f178"]:
    #ablation_id="f2f2"#"aaad-DROPOUT_word-vs-sent-0.2"#"aaad-DROPOUT_word-vs-sent-0.2-to_all-model_3_2b03-folder"
    # local test
    #ablation_id = "42a20"
    ablation_id = "a5c77"
    #ablation_id = "8ce6b-extend_ep-get_True-attention_simplifiedXauxXdropout0.1_scale_aux-True_aux-0.1do_char_dec-True_char_src_atten-model_14_ad6c"
    ablation_id = "4e128-WARMUP-unrolling-False0-model_1-model_1_1660-folder"
    ablation_id = "/1eeb9-WARMUP-unrolling-False0-model_1-model_1_fd8c-folder"
    #for data in [LIU, DEV]:
    list_ = [dir_ for dir_ in list_all_dir if dir_.startswith(ablation_id) and not dir_.endswith("log") and not dir_.endswith("summary")]
    print("FOLDERS : ", list_)
    #list_ = []
    list_ = ["1eeb9-WARMUP-unrolling-False0-model_1-model_1_fd8c-folder"]
    # word decode with word embed
    list_ = ["1f86c-WARMUP-unrolling-False0-model_1-model_1_57b7-folder"]
    # char decode
    list_ = ["aa1bc-B0-model_1-model_1_1cb2-folder"]
    for folder_name in list_:
        model_full_name = folder_name[:-7]
        print("Interatcing with new model : ", model_full_name)
        print("0Evaluating {} ".format(model_full_name))
        dic_path = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder", "dictionaries")
        model_dir = os.path.join(script_dir, "..", "checkpoints", model_full_name + "-folder")
        interact(dic_path=dic_path, dir_model=model_dir, model_full_name=model_full_name,
                 beam_decode=False, beam_size=1,
                 word_decoding=False,
                 save_attention=False, show_attention=True,
                 debug=False, verbose=1)
        #break
    #show_attention("[lekfezlfkh efj ", ["se", "mjfsemkfj"], torch.tensor([[0,.4], [1,0.6]]))