
import git


def get_commit_id():
    repo = git.Repo(search_parent_directories=True)
    git_commit_id = repo.head.object.hexsha
    return git_commit_id