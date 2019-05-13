
from io_.info_print import printing


def reframe_tsv_to_sentConll(src_dir, target_dir, verbose=1):

    with open(src_dir,"r") as f:
        with open(target_dir, "w") as g:
            line = "0"
            while len(line) > 0:
                line = f.readline()
                line = line.strip()
                sent = line.split("\t")
                print(sent)
                if len(line)>0:

                    if len(sent)<3:
                        g.write("#sent_id {}\n".format("XXX"))
                        g.write("1\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(sent[0], sent[1]))
                    else:
                        g.write("#sent_id {}\n".format(sent[0]))
                        g.write("1\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(sent[1], sent[2]))
                    g.write("\n")

    printing("WRITTEN {} src and {} target directory".format(src_dir, target_dir), verbose=verbose, verbose_level=1)


def reframe_tsv_monolingual_to_sentConll(src_dir,src_dir_2, target_dir, verbose=1):

    with open(src_dir,"r") as f:
        with open(src_dir_2, "r") as f2:
            with open(target_dir, "w") as g:
                line = "0"
                line_2 = "0"
                i = 0
                while len(line) > 0:
                    while len(line_2)>0:
                        line = f.readline()
                        line_2 = f2.readline()
                        line = line.strip()
                        line_2 = line_2.strip()
                        i+=1
                        print(line, line_2)
                        if len(line)>0:
                            g.write("#sent_id {}\n".format("i"))
                            g.write("1\t{}\t_\t_\t_\t_\t1\t_\t_\tNorm={}|\n".format(line, line_2))
                            g.write("\n")

    printing("WRITTEN {} src and {} target directory".format(src_dir, target_dir), verbose=verbose, verbose_level=1)


if __name__ == "__main__":

    reframe_MT, reframe_tok = False, False

    if reframe_MT:
        for type in ["train","valid", "test"]:
            print(""
                  "", type)
            reframe_tsv_to_sentConll("/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/MTNT/{}/{}.en-fr.tsv".format(type, type),
                                     "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/MTNT/{}/{}.en-fr.conll".format(type, type))
    if reframe_tok:
        for type in ["dev"]:
            reframe_tsv_monolingual_to_sentConll("/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/MTNT/monolingual/{}.en".format(type),
                                                 "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/MTNT/monolingual/{}.tok.en".format(type),
                                                 "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/MTNT/monolingual/{}.en.raw2tok.conll".format(type))
