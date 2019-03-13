from training.args_tool import args_train

from training.train_eval import train_eval

import os 
if __name__ == "__main__":

    args = args_train()
    params = vars(args)
    print("DISTRIBUTED : train_eval starting thread on GPU {} ".format(os.environ.get("CUDA_VISIBLE_DEVICES","NO GPU FOUND")))
    model_full_name, model_dir = train_eval(args=params,
                                           model_id_pref=args.model_id_pref,
                                           train_path=args.train_path,
                                           n_epochs=args.epochs,
                                           dev_path=args.dev_path,
                                           test_path=args.test_paths,
                                           expand_vocab_dev_test=args.expand_vocab_dev_test,
                                           pos_specific_path=args.pos_specific_path,
                                           overall_label=args.overall_label, overall_report_dir=args.overall_report_dir,
                                           extend_n_batch=2,
                                           get_batch_mode_all=True, compute_mean_score_per_sent=False, bucketing_train=True, freq_checkpointing=1,
                                           symbolic_end=True, symbolic_root=True, freq_writer=1, compute_scoring_curve=False,
                                           gpu=args.gpu,
                                           verbose=args.verbose,
                                           warmup=args.warmup,
                                           debug=args.debug)
    print("DISTRIBUTED : train_eval  thread on done of model {} dir {} ".format(model_full_name, model_dir))
