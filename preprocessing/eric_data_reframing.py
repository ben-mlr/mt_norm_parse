from env.importing import *
from env.project_variables import ERIC_ORIGINAL, ERIC_ORIGINAL_DEMO
from io_.info_print import printing


def conll_eric_to_conll_lexnorm(src_dr, target, mode="conll",skipping_mwe_sent=False, verbose=1):
    assert mode in ["conll","3cols"]
    target_file = open(target, "w")
    line = "#"
    ind = 0
    mapping_counter = 0
    mapping_counter_max = 0
    normalization = ""
    lemma = ""
    tag = ""
    row_sent_to_write = []
    skip_sentence = False
    with open(src_dr, "r", encoding="ISO-8859-1") as src:
        while len(line) > 0:
            printing("LINE {}", var=[line], verbose=verbose, verbose_level=2)
            # if line is empty or comment
            if len(line.strip()) == 0:# or line.strip()[0] == '#':
                # writing all line that starts with #
                # writing memorizing sentences
                if row_sent_to_write is not None and not skip_sentence:
                    for row in row_sent_to_write:
                        target_file.write(row)
                row_sent_to_write = []
                skip_sentence = False
                row_sent_to_write.append(line)
                adjust = 0
            elif line.strip()[0] == '#':
                if mode == "conll":
                    row_sent_to_write.append(line)
                adjust = 0
            elif len(line.strip()) > 0:
                line = line.strip()
                line = line.split('\t')
                index = line[0]
                match_index = re.match("([0-9]+)", index)
                match_1_to_n = re.match("([0-9]+)-([0-9]+)", index)
                # if it's a double index
                if match_1_to_n is not None:
                    printing("DOUBLE {}", var=[line], verbose=verbose, verbose_level=2)
                    first_ind = match_1_to_n.group(1)
                    last_ind = match_1_to_n.group(2)
                    mapping_counter_max = int(last_ind)-int(first_ind)+1
                    if mapping_counter_max > 1 and skipping_mwe_sent:
                        skip_sentence = True
                    adjust += mapping_counter_max-1

                    token_form = line[1]
                # if it's a simple index
                else:
                    printing("SINGLE {}", var=[line], verbose=verbose, verbose_level=2)
                    assert match_index is not None, "ERROR no index {} was found in line {}".format(index, line)
                    if mapping_counter_max == 0:
                        printing("STANDARD {}", var=[line], verbose=verbose, verbose_level=2)
                        normalization = line[1]
                        lemma = line[2]
                        tag = line[3]
                        if mode == "conll":
                            row_to_write = "{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(int(match_index.group(1))-adjust,
                                                                                                 normalization,
                                                                                                 lemma, tag, normalization)
                        elif mode == "3cols":
                            row_to_write = "{}\t{}\t{}\n".format(normalization,
                                                                 "NORMED",
                                                                 normalization)
                        row_sent_to_write.append(row_to_write)
                        #target_file.write(row_to_write)
                        normalization = ""
                        lemma = ""
                        tag = ""
                    else:
                        printing("1 to N", var=[line], verbose=verbose, verbose_level=2)
                        if mapping_counter <= mapping_counter_max:
                            normalization += "_"+line[1] if len(normalization)>0 else line[1]
                            lemma += "_"+line[2] if len(lemma) > 0 else line[2]
                            tag += "_" + line[3] if len(tag) > 0 else line[3]
                            printing("1 to N ACCUMULATE {}", var=[line], verbose=verbose, verbose_level=2)
                            if mapping_counter == (mapping_counter_max-1):
                                printing("1 to N WRITE {}", var=[line], verbose=verbose, verbose_level=2)
                                mapping_counter = -1
                                mapping_counter_max = 0
                                if mode == "conll":
                                    row_to_write = "{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(int(first_ind),
                                                                                                         token_form,
                                                                                                         lemma, tag,
                                                                                                         normalization)
                                elif mode == "3cols":
                                    row_to_write = "{}\t{}\t{}\n".format(token_form,
                                                                         "NORMED" if token_form == normalization
                                                                         else "NEED_NORM",
                                                                         normalization)
                                #target_file.write(row_to_write)
                                row_sent_to_write.append(row_to_write)
                                normalization = ""
                                lemma = ""
                                tag = ""
                                # next token with : WE ASSUME NOT FOLLOWING MWE
                                #target_file.write("{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(int(match_index.group(1))-adjust, line[1],
                                #                                                                        line[2], line[3],
                                #                                                                        line[1]))
                        else:
                            raise(Exception("should not be having {} mapping counter and mapping counter max {}"
                                            "".format(mapping_counter, mapping_counter_max)))
                        mapping_counter += 1
            line = src.readline()
            printing("NEW LINE", var=[line], verbose_level=2, verbose=verbose)
            if len(line) == 0 and ind > 0:
                printing("ENDING {} ", var=[target], verbose_level=1, verbose=verbose)
                break

            ind += 1


if __name__ == "__main__":

    #conll_eric_to_conll_lexnorm(ERIC_ORIGINAL_DEMO, ERIC_ORIGINAL_DEMO+"-ttt_TEST-skped.conllu",
    # skipping_mwe_sent=True)
    conll_eric_to_conll_lexnorm(ERIC_ORIGINAL, ERIC_ORIGINAL[:-7] + "-only_1to1.txt",
                                skipping_mwe_sent=True, mode="3cols")
