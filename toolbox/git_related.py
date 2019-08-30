from env.importing import *


def get_commit_id():
    repo = git.Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
    git_commit_id = str(repo.head.commit)#object.hexsha
    return git_commit_id
