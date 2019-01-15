import re
import argparse
import json
import numpy as np

def process_liu_data(dir_src, dir_target):
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
                    out.write("{count}\t{word}\t_\t_\t_\t_\t0\t_\t_\tNorm={norm}|\n".format(count=count_word,
                                                                                     word=origin,norm=norm))
                    count_word += 1
                elif re.match("\n", line) is not None:
                    count_sent += 1
                    count_word = 1
                    out.write("\n")
                else:
                    raise Exception("LINE line {} not match".format(line))

    print("n_sent prcocessed {}".format(count_sent))


def process_lexnorm2015_data(dir_src, dir_target, log=False):
    data = json.load(open(dir_src,"r"))
    with open(dir_target,"w") as f:
        count_tweets = 0
        len_tweets =  []
        noisy_per_tweets = []
        clean_per_tweets = []

        for tweet in data:
            if len(tweet["input"])!=len(tweet["output"]):
                print("TWEET id {} has n to 1 mapping {} ".format(tweet["id"],tweet))
                continue
            f.write("\n#tweet id {} \n".format(tweet["tid"]))
            f.write("#index  {} \n".format(tweet["index"]))
            count_tweets+=1
            count = 1
            
            clean_tok = 0
            noisy_tok = 0
            for noisy, normed in zip(tweet["input"], tweet["output"]):
                f.write("{count}\t{word}\t_\t_\t_\t_\t0\t_\t_\tNorm={norm}|\n".format(count=count, word=noisy,norm=normed))
                count+=1
                if noisy!=normed:              
                    noisy_tok+=1
                else:
                    clean_tok+=1
            len_tweets.append(len(tweet["input"]))
            noisy_per_tweets.append(noisy_tok)
            clean_per_tweets.append(clean_tok)

    print("n_sent prcocessed {}".format(count_tweets))
    print("conll written  {} based on  {} ".format(dir_target, dir_src))
    
    mean_tweet_len = np.mean(len_tweets)
    mean_clean_tok_per_tweet = np.mean([n_clean/len_tweet*100 for n_clean, len_tweet in zip(clean_per_tweets, len_tweets)])
    mean_noisy_tok_per_tweet = np.mean([n_noisy/len_tweet*100 for n_noisy, len_tweet in zip(noisy_per_tweets, len_tweets)])
    n_tok =  np.sum(len_tweets)
    n_noisy_tok =  np.sum(noisy_per_tweets)
    n_cleaned_tok =  np.sum(clean_per_tweets)
    print(n_cleaned_tok, n_noisy_tok, n_tok)
    if log:
        info = "Lexnorm {dir_src} dataset was transform to {dir_target} with process_lexnorm2015_data in ./toolbox/tweets_norm_to_conllu.py \n \
        Statistics : {count_tweets} tweets or  {n_tok} total tokens, {n_noisy_tok}({percent_noisy:0.2f}%) noisy tokens and {n_cleaned_tok}({percent_clean:0.2f}%) clean tokens \n \
        Each token is in average {mean_tweet_len:0.2f} tokens with {mean_noisy_tok_per_tweet:0.2f}% noisy token per tweet in average and {mean_clean_tok_per_tweet:0.2f}% clean token per tweet in average \
        ".format(dir_src=dir_src, dir_target=dir_target, count_tweets=count_tweets,n_tok=n_tok,n_noisy_tok= n_noisy_tok, 
                    percent_noisy=n_noisy_tok/n_tok*100, n_cleaned_tok=n_cleaned_tok,
                     percent_clean=n_cleaned_tok/n_tok*100, mean_tweet_len=mean_tweet_len, mean_noisy_tok_per_tweet=mean_noisy_tok_per_tweet,
                     mean_clean_tok_per_tweet=mean_clean_tok_per_tweet)

        with open(dir_target+".log","w") as f:
            f.write(info)

        print("log writted to {} ".format(dir_target+".log"))



if __name__== "__main__":
    liu_process = False
    lexnorm_process = True
    args = argparse.ArgumentParser()
    args.add_argument("--src", required=True)
    args.add_argument("--target", required=True)
    args.add_argument('--log', help='',action="store_true")


    args = args.parse_args()
    if liu_process:
        process_liu_data(args.src, args.target)
    elif lexnorm_process:
        process_lexnorm2015_data(args.src, args.target, args.log)


