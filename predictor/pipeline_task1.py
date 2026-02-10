import argparse
import os
from .grail_runner import run_grail_test_auc, run_grail_test_ranking, run_grail_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1: run GraIL training/testing")
    parser.add_argument("--grail-repo", required=True, help="Path to cloned GraIL repository")
    parser.add_argument("--dataset-name", required=True, help="Dataset name under grail/data")
    parser.add_argument("--experiment-name", required=True, help="Experiment name for GraIL logs")
    parser.add_argument("--python", default="python", help="Python executable")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--grail-args", nargs=argparse.REMAINDER, default=[], help="Extra args passed to GraIL train.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.grail_repo):
        raise ValueError("GraIL repo path does not exist")

    extra_args = args.grail_args if args.grail_args else None
    run_grail_train(
        args.grail_repo,
        args.dataset_name,
        args.experiment_name,
        python_exe=args.python,
        extra_args=extra_args,
    )

    if args.run_tests:
        run_grail_test_auc(args.grail_repo, args.dataset_name, args.experiment_name, python_exe=args.python)
        run_grail_test_ranking(args.grail_repo, args.dataset_name, args.experiment_name, python_exe=args.python)


if __name__ == "__main__":
    main()
