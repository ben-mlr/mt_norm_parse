import re
import argparse


def process(dir_src, dir_target):
    count_sent = 1
    count_word = 1
    with open(dir_src,"r") as f:
        with open(dir_target, "w") as out:
            for line in f:
                match_line = re.match("^(.*)\t(NORMED|NEED_NORM)\t(.*)", line)
                if match_line is not None:
                    origin = match_line.group(1)
                    tag = match_line.group(2)
                    norm = match_line.group(3)
                    if count_word==1:
                        out.write("# sent_id = {} \n".format(count_sent))
                    out.write("{count}\t{word}\t_\t_\t_\t_\t_\t_\t_\tNorm={norm}\n".format(count=count_word,
                                                                                     word=origin,norm=norm))
                    count_word += 1
                elif re.match("\n", line) is not None:
                    count_sent += 1
                    count_word = 1
                    out.write("\n")
                else:
                    raise Exception("LINE line {} not match".format(line))

    print("n_sent prcocessed {}".format(count_sent))


if __name__== "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--src", required=True)
    args.add_argument("--target", required=True)
    args = args.parse_args()
    process(args.src, args.target)


