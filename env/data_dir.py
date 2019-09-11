from env.importing import os, re

# NB : lots of the datasets directory are in project_variables
# this files aims to group all of those at some point


DATA_UD = os.environ.get("DATA_UD", "/Users/bemuller/Documents/Work/INRIA/dev/parsing/data/Universal-Dependencies-2.4")


DATASET_CODE_LS = ['af_afribooms', 'ar_nyuad', 'ar_padt', 'be_hse', 'bg_btb', 'bxr_bdt', 'ca_ancora', 'cop_scriptorium',
                   'cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt', 'cu_proiel', 'da_ddt', 'de_gsd', 'el_gdt', 'en_esl',
                   'en_ewt', 'en_gum', 'en_lines', 'en_partut', 'es_ancora', 'es_gsd', 'et_edt', 'et_ewt', 'eu_bdt',
                   'fa_seraji', 'fi_ftb', 'fi_tdt', 'fr_ftb', 'fr_gsd', 'fr_partut', 'fr_sequoia', 'fr_spoken', 'fro_srcmf',
                   'ga_idt', 'gl_ctg', 'gl_treegal', 'got_proiel', 'grc_perseus', 'grc_proiel', 'he_htb', 'hi_hdtb', 'hr_set',
                   'hsb_ufal', 'hu_szeged', 'hy_armtdp', 'id_gsd', 'it_isdt', 'it_partut', 'it_postwita', 'it_vit', 'ja_bccwj',
                   'ja_gsd', 'kk_ktb', 'kmr_mg', 'ko_gsd', 'ko_kaist', 'la_ittb', 'la_perseus', 'la_proiel', 'lt_alksnis', 'lt_hse',
                   'lv_lvtb', 'lzh_kyoto', 'mr_ufal', 'mt_mudt', 'nl_alpino', 'nl_lassysmall', 'no_bokmaal', 'no_nynorsk', 'no_nynorsklia',
                   'orv_torot', 'pl_lfg', 'pl_pdb', 'pt_bosque', 'pt_gsd', 'qhe_hiencs', 'ro_nonstandard', 'ro_rrt', 'ru_gsd', 'ru_syntagrus',
                   'ru_taiga', 'sk_snk', 'sl_ssj', 'sl_sst', 'sme_giella', 'sr_set', 'sv_lines', 'sv_talbanken', 'swl_sslc', 'ta_ttb',
                   'te_mtg', 'tr_imst', 'ug_udt', 'uk_iu', 'ur_udtb', 'vi_vtb', 'wo_wtb', 'zh_gsd']


def get_dir_data(set, data_code, demo=False):
    assert set in ["train", "dev", "test"]
    assert data_code in DATASET_CODE_LS, "ERROR {}".format(data_code)
    demo_str = "-demo" if demo else ""

    file_dir = os.path.join(DATA_UD, "{}-ud-{}{}.conllu".format(data_code, set, demo_str))

    assert os.path.isfile(file_dir), "{} not found".format(file_dir)

    return file_dir


def get_code_data(dir):
    matching = re.match(".*\/([^\/]+).*.conllu", dir)
    if matching is not None:
        return matching.group(1)
    return "training_set-not-found"

# 1 define list of dataset code and code 2 dir dictionary
# 2 : from grid_run : call dictionary and iterate on data set code

