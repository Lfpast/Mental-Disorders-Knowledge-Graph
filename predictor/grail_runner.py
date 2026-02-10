import subprocess
from typing import List, Optional


def run_grail_train(
    grail_repo: str,
    dataset_name: str,
    experiment_name: str,
    python_exe: str = "python",
    extra_args: Optional[List[str]] = None,
) -> None:
    cmd = [python_exe, "train.py", "-d", dataset_name, "-e", experiment_name]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=grail_repo, check=True)


def run_grail_test_auc(grail_repo: str, dataset_name: str, experiment_name: str, python_exe: str = "python") -> None:
    cmd = [python_exe, "test_auc.py", "-d", dataset_name, "-e", experiment_name]
    subprocess.run(cmd, cwd=grail_repo, check=True)


def run_grail_test_ranking(grail_repo: str, dataset_name: str, experiment_name: str, python_exe: str = "python") -> None:
    cmd = [python_exe, "test_ranking.py", "-d", dataset_name, "-e", experiment_name]
    subprocess.run(cmd, cwd=grail_repo, check=True)
