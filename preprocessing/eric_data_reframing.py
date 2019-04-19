from env.importing import *
from env.project_variables import ERIC_ORIGINAL, ERIC_ORIGINAL_DEMO


def conll_eric_to_conll_lexnorm(src_dr,target):
    target_file = open(target, "w")
    line = "#"
    ind = 0
    mapping_counter = 0
    mapping_counter_max = 0
    normalization = ""
    lemma = ""
    tag = ""
    with open(src_dr, "r", encoding="ISO-8859-1") as src:
        while len(line) > 0:
            print("LINE", line)
            if len(line.strip()) == 0 or line.strip()[0] == '#':
                # writing all line that starts with #
                target_file.write(line)
                adjust = 0
            elif len(line.strip()) > 0:
                line = line.strip()
                line = line.split('\t')
                index = line[0]
                match_index = re.match("([0-9]+)", index)
                match_1_to_n = re.match("([0-9]+)-([0-9]+)", index)

                print("LINE REAL", line)
                if match_1_to_n is not None:
                    print("DOUBLE", line)
                    first_ind = match_1_to_n.group(1)
                    last_ind = match_1_to_n.group(2)
                    mapping_counter_max = int(last_ind)-int(first_ind)+1
                    adjust += mapping_counter_max-1
                    token_form = line[1]

                else:
                    print("SINGLE", line)
                    assert match_index is not None, "ERROR no index {} was found in line {}".format(index, line)
                    if mapping_counter_max == 0:
                        print("STANDARD", line)
                        normalization = line[1]
                        lemma = line[2]
                        tag = line[3]
                        target_file.write("{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(int(match_index.group(1))-adjust, normalization,
                                                                                          lemma, tag, normalization))
                        normalization = ""
                        lemma = ""
                        tag = ""
                    else:
                        print("1 to N", line)
                        if mapping_counter <= mapping_counter_max:
                            normalization += "_"+line[1] if len(normalization)>0 else line[1]
                            lemma += "_"+line[2] if len(lemma)>0 else line[2]
                            tag += "_" + line[3] if len(tag)>0 else line[3]
                            print("1 to N ACCUMULATE", line)
                            if mapping_counter == (mapping_counter_max-1):
                                print("1 to N WRITE", line)
                                mapping_counter = -1
                                mapping_counter_max = 0
                                target_file.write(
                                    "{}\t{}\t{}\t{}\t_\t_\t_\t_\t_\t_\tNorm={}|\n".format(int(first_ind), token_form,
                                                                                          lemma, tag,
                                                                                          normalization))
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
            print("NEW LINE", line)
            if len(line) == 0 and ind > 0:
                print("ENDING check out {}".format(target))
                break

            ind += 1

conll_eric_to_conll_lexnorm(ERIC_ORIGINAL_DEMO, ERIC_ORIGINAL_DEMO+"ttt.conllu")