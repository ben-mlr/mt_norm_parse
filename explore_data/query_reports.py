


import json 

def get_field(field, key_dic, report_dir):
	#print("LOOKING for keys ", key_dic)
	reports = json.load(open(report_dir,"r"))
	n_report_find = 0
	for rep in reports:
		match = 0
		for key,val in key_dic.items():
			if rep[key] == val:
				match+=1
			if match==len(key_dic):
				answer = rep 
				n_report_find+=1

	#print("LAST MATCH", answer)
	assert n_report_find>0, "ERROR no report found with all keys {} ".format(key_dic)
	print("FIELD {} is {} for keys {} ({} reports found)".format(field, answer[field],key_dic, n_report_find))
	return  answer[field], answer, n_report_find





if __name__=="__main__":

	import sys
	sys.path.insert(0,"..")
	sys.path.insert(0,".")
	import os 
	import argparse
	from env.project_variables import *



	#from env.importing import argparse, os
    #parser = argparse.ArgumentParser()
	#parser.add_argument("--report_dir", required=True, type=str, help="display a square of a given number")
	dir_report = "/home/bemuller/projects/mt_norm_parse/env/../checkpoints/bert/"

	# MODEL with 1 batch (to have the least amount of skipped sample )+train on LIU TRAIN + OWOPUTI
	#model_name = "9297344-B-bad2b-9297344-B-model_11"
	model_name ="9332867-B-8f5b6-9332867-B-model_0"
	# MODEL with 1 batch (to have the least amount of skipped sample )+train on LIU ALL and that's it 
	#model_name = "9297044-B-7da2f-9297044-B-model_1"
	#model_name = "9297044-B-91dad-9297044-B-model_10"
	# NB : 9297044-B-91dad-9297044-B-model_10 vs 9297344-B-bad2b-9297344-B-model_11 
	# TEST OOV : from 29% to 32.57% on (model with owoputi, model with liu dev as train)
	# OWOPUTI : from 20% to 33.3%
	# liu dev = from 26% to 17% 
	# the 3% difference in overall oov rate might be impacting a lot the need_norms token on which the models are progressing (seen in F1) !! 

	data_path = LEX_TEST
	data_label_suff = "normalize"
	_all=get_field(field="n_tokens_score", key_dic={"data":REPO_DATASET[data_path]+"-"+data_label_suff, "subsample":"all", "model_full_name":model_name}, report_dir=os.path.join(dir_report, model_name,model_name+"-report.json"))
	oov=get_field(field="n_tokens_score", key_dic={"data": REPO_DATASET[data_path]+"-"+data_label_suff, "subsample":"OOV", "model_full_name":model_name}, report_dir=os.path.join(dir_report, model_name,model_name+"-report.json"))
	print(oov/_all*100)

	# ALSO : n_tokens with space  grep  'Norm=[a-z]* [a-z]*' ./data/wnut-2015-ressources/lexnorm2015/train_data.conll
