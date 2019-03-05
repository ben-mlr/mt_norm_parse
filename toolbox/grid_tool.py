from io_.info_print import printing

DEFAULT_BATCH_SIZE = 25
DEFAULT_SCALE = 2
DEFAULT_AUX_NORM_NOT_NORM = False


def grid_param_label_generate(param, batch_size_ls=None, lr_ls=None, scale_ls =None,
                              #auxilliary_task_norm_not_norm_ls=None,
                              shared_context_ls=None,
                              word_embed_init_ls=None, dir_word_encoder_ls=None, char_src_attention_ls=None, dir_sent_encoder_ls=None,
                              clipping_ls=None, unrolling_word_ls=None, teacher_force_ls=None,
                              word_decoding_ls=None,
                              #auxilliary_task_pos_ls=None,
                              stable_decoding_state_ls=None, warmup=False,
                              word_embedding_projected_dim_ls=None, n_layers_sent_cell_ls=None, word_embed_ls=None,proportion_pred_train_ls=None,
                              tasks_ls=None,
                              grid_label=""):

  params = []
  labels = []
  default = []
  info_default = []

  if batch_size_ls is None:
    batch_size_ls = [DEFAULT_BATCH_SIZE]
    default.append(("batch_size",batch_size_ls[0]))
  if lr_ls is None:
    lr_ls = [0.001]
  if scale_ls is None:
    scale_ls = [DEFAULT_SCALE]
    #default.append(scale_ls[0])
  #if auxilliary_task_norm_not_norm_ls is None:
  #  auxilliary_task_norm_not_norm_ls = [DEFAULT_AUX_NORM_NOT_NORM]
  #  default.append(("auxilliary_task_norm_not_norm",auxilliary_task_norm_not_norm_ls[0]))
  if shared_context_ls is None:
    shared_context_ls = ["all"]
    default.append(("shared_context",shared_context_ls[0]))
  if word_embed_init_ls is None:
    word_embed_init_ls = [None]
  if dir_word_encoder_ls is None:
    dir_word_encoder_ls = [2]
    default.append(("dir_word_encoder", dir_word_encoder_ls[0]))
  if char_src_attention_ls is None:
    char_src_attention_ls = [True]
    default.append(("char_src_attention", char_src_attention_ls[0]))
  if dir_sent_encoder_ls is None:
    dir_sent_encoder_ls = [2]
    default.append(("dir_sent_encoder", dir_sent_encoder_ls[0]))
  if clipping_ls is None:
    clipping_ls = [1]
    default.append(("gradient_clipping", clipping_ls[0]))
  if unrolling_word_ls is None:
    unrolling_word_ls = [True]
    default.append(("unrolling_word", unrolling_word_ls[0]))
  if teacher_force_ls is None:
    teacher_force_ls = [True]
    default.append(("teacher_force", teacher_force_ls[0]))
  if word_decoding_ls is None:
    word_decoding_ls = [False]
    default.append(("word_decoding", word_decoding_ls[0]))
  #if auxilliary_task_pos_ls is None:
  ##  auxilliary_task_pos_ls = [False]
  # default.append(("auxilliary_task_pos", auxilliary_task_pos_ls[0]))
  if stable_decoding_state_ls is None:
    stable_decoding_state_ls = [False]
    default.append(("stable_decoding_state", stable_decoding_state_ls[0]))
  if word_embedding_projected_dim_ls is None:
    word_embedding_projected_dim_ls = [None]
    default.append(("word_embedding_projected_dim", word_embedding_projected_dim_ls[0]))
  if n_layers_sent_cell_ls is None:
    n_layers_sent_cell_ls = [1]
  if word_embed_ls is None:
    word_embed_ls = [True]
    default.append(("word_embed",word_embed_ls[0]))
  if proportion_pred_train_ls is None:
    proportion_pred_train_ls = [None]
    default.append(("proportion_pred_train",proportion_pred_train_ls[0]))
  if tasks_ls is None or len(tasks_ls) == 0:
    tasks_ls = [["normalize"]]
    default.append(("task", tasks_ls[0]))
  for def_ in default:
    info_default.append((def_[0],def_[1])) #" "+str(def_[0])+","+str(def_[0])
    printing("GRID : {} argument defaulted to {} ", var=[str(def_)[:-6], def_], verbose=0, verbose_level=0)

  dic_grid = {"batch_size": batch_size_ls,# "auxilliary_task_norm_not_norm": auxilliary_task_norm_not_norm_ls,
              "shared_context": shared_context_ls,
              "lr": lr_ls, "word_embed_init": word_embed_init_ls, "dir_word_encoder": dir_word_encoder_ls,
              "char_src_attention":char_src_attention_ls,
              "dir_sent_encoder": dir_sent_encoder_ls, "gradient_clipping":clipping_ls, "unrolling_word": unrolling_word_ls,
              "word_decoding": word_decoding_ls, #"auxilliary_task_pos": auxilliary_task_pos_ls,
              "stable_decoding_state": stable_decoding_state_ls,
              "word_embedding_projected_dim": word_embedding_projected_dim_ls,
              "n_layers_sent_cell": n_layers_sent_cell_ls,
              "teacher_force": teacher_force_ls, "proportion_pred_train": proportion_pred_train_ls,
              "tasks":tasks_ls,
              "word_embed": word_embed_ls}
  ind_model = 0
  for batch in batch_size_ls:
    #for aux in auxilliary_task_norm_not_norm_ls:
    for shared_context in shared_context_ls:
      for lr in lr_ls:
        for word_embed_init in word_embed_init_ls:
          for scale in scale_ls:
            if shared_context == "sent":
              scale_sent_context = 1
              scale_word = 1
              scaled_output_dim = 1
            else:
              scale_sent_context, scale_word, scaled_output_dim = 1, 1, 1
            for dir_word_encoder in dir_word_encoder_ls:
              for char_src_attention in char_src_attention_ls:
                for dir_sent_encoder in dir_sent_encoder_ls:
                  for clipping in clipping_ls :
                    for unrolling_word in unrolling_word_ls:
                      for word_decoding in word_decoding_ls:
                      #for auxilliary_task_pos in auxilliary_task_pos_ls:
                        for stable_decoding_state in stable_decoding_state_ls:
                          for word_embedding_projected_dim in word_embedding_projected_dim_ls:
                            for word_embed in word_embed_ls:
                              for n_layers_sent_cell in n_layers_sent_cell_ls:
                                for proportion_pred_train in proportion_pred_train_ls:
                                  for tasks in tasks_ls:
                                    for teacher_force in teacher_force_ls:
                                      param0 = param.copy()
                                      ind_model += 1
                                      param0["batch_size"] = batch
                                      #param0["auxilliary_task_norm_not_norm"] = aux
                                      param0["shared_context"] = shared_context
                                      param0["lr"] = lr
                                      param0["word_embed_init"] = word_embed_init
                                      param0["hidden_size_encoder"] = int(param0["hidden_size_encoder"] * scale *
                                                                          scale_word)
                                      param0["hidden_size_sent_encoder"] = int(param0["hidden_size_sent_encoder"] *
                                                                               scale * scale_sent_context)
                                      param0["hidden_size_decoder"] = int(param0["hidden_size_decoder"] * scale)
                                      param0["output_dim"] *= int(scale * scaled_output_dim) + 1
                                      param0["dir_word_encoder"] = dir_word_encoder
                                      param0["char_src_attention"] = char_src_attention
                                      param0["unrolling_word"] = unrolling_word
                                      param0["dir_sent_encoder"] = dir_sent_encoder
                                      param0["n_layers_sent_cell"] = n_layers_sent_cell
                                      param0["gradient_clipping"] = clipping
                                      param0["teacher_force"] = teacher_force
                                      param0["word_decoding"] = word_decoding
                                      param0["char_decoding"] = not word_decoding
                                      #param0["auxilliary_task_pos"] = auxilliary_task_pos
                                      param0["dense_dim_auxilliary_pos"] = 100 #if not "pos" in tasks else 0
                                      param0["dense_dim_auxilliary_pos_2"] = 100 #if not "pos" in tasks else 100

                                      param0["stable_decoding_state"] = stable_decoding_state
                                      param0["init_context_decoder"] = not param0["stable_decoding_state"]
                                      param0["activation_char_decoder"] = "nn.LeakyReLU"
                                      param0["activation_word_decoder"] = "nn.LeakyReLU"

                                      param0["tasks"] = tasks
                                      # default
                                      param0["dropout_bridge"] = 0.1
                                      param0["word_embed"] = word_embed
                                      if word_embed_init is not None and word_embed:
                                        param0["word_embedding_dim"] = 400
                                      elif word_embed:
                                        param0["word_embedding_dim"] = 50
                                      else:
                                        param0["word_embedding_dim"] = 0
                                      param0["dense_dim_word_pred"] = 300 if word_decoding else None
                                      param0["dense_dim_word_pred_2"] = 300 if word_decoding else None
                                      param0["dense_dim_word_pred_3"] = 100 if word_decoding else None
                                      param0["word_embedding_projected_dim"] = word_embedding_projected_dim if param0["word_embed"] else None
                                      param0["proportion_pred_train"] = proportion_pred_train
                                      params.append(param0)
                                      labels.append("{}-model_{}".format(grid_label,ind_model))


  studied_vars = []
  fixed_vars = []
  for var, vals in dic_grid.items():
    if var == "proportion_pred_train":
      if None in vals:
        vals[vals.index(None)] = 0
    if len(vals) > 1:
      print("STUDIES", var, vals)
      studied_vars.append(var)
    else:
      print("FIXED", var, vals)
      fixed_vars.append((var, vals[0]))
  

  return params,labels, info_default, studied_vars, fixed_vars