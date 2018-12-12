

def sanity_check_info_checkpoint(info_checkpoint, template):
    for key in template.keys():
        # git id is added on the fly as updated
        if key not in  ["git_id", "other"]:
            assert key in info_checkpoint, "ERROR {} key is not in info_checkpoint".format(key)