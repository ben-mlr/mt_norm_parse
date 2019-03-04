
from env.project_variables import DEMO, LIU_TRAIN, LIU_DEV, LEX_TEST, TEST, DEV
from io_.dat.conllu_get_normalization import get_normalized_token
from io_.info_print import printing


def reframe_conll_to_sentConll(src_dir, target_dir, n_hashtag=1,verbose=1):

    with open(src_dir,"r") as f:
        with open(target_dir, "w") as g:
            line = "0"
            sent = []

            while len(line) > 0:
                line = f.readline()
                if line.startswith("#") :
                    g.write(line)
                    # if new_sent == n_hashtag:
                    sent = []
                    output_sent = 0
                elif line != "\n" and len(line) > 0:
                    splitted = line.split('\t')
                    if "-" in splitted[0]:
                        continue
                    sent.append(splitted)
                    src_sent = ""
                    target_sent = ""
                if line == "\n":
                    output_sent = 1
                if output_sent == 1:
                    space = ""
                    for row in sent:
                        src_sent += space+row[1]
                        target_sent += space + get_normalized_token(norm_field=row[9], n_exception=0, verbose=1)[0]
                        space = " "
                    g.write("1\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(src_sent, target_sent))
                    g.write("\n")


    printing("WRITTEN {} src and {} target directory".format(src_dir, target_dir), verbose=verbose, verbose_level=1)


if __name__=="__main__":
    reframe_conll_to_sentConll(LIU_TRAIN, "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/.././data/LiLiu/2577_tweets-li-sent-train_2009.conll")
    reframe_conll_to_sentConll(LIU_DEV,
                               "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/.././data/LiLiu/2577_tweets-li-sent-dev_500.conll")
    print(TEST)
    reframe_conll_to_sentConll(TEST,"/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/lexnorm-sent.conll")
    reframe_conll_to_sentConll(DEV,
                               "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/owoputi-sent.conll")
