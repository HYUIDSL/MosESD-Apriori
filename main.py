import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from src.utils.experiment import run_experiment
import config

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def get_arg_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description="Run anomaly detection experiments")
    parser.add_argument("--dataset", type=str, default="semes", 
                       help="Dataset name (semes, psm, smap, msl, swat, wadi, smd)")
    parser.add_argument("--log_num", type=int, default=None, 
                       help="Log number (-1 for all files)")
    parser.add_argument("--step", type=int, default=None,
                       help="Step filter for SEMES data")
    parser.add_argument("--metric", type=str, default="pa_f1", 
                       choices=["f1", "pa_f1"],
                       help="Metric for evaluation")
    parser.add_argument("--num_workers", type=int, default=-1,
                        help="Number of workers")
    return parser.parse_args()


def merge_config_to_args(args):
    """Merge config.py constants into args namespace (uppercase keys only)"""
    for key in dir(config):
        if key.isupper():  # Only uppercase constants
            value = getattr(config, key)
            key_lower = key.lower()
            if not hasattr(args, key_lower):
                setattr(args, key_lower, value)
    return args

def main():
    args = get_arg_parser()
    args = merge_config_to_args(args)
    run_experiment(args)

if __name__ == "__main__":
    main()    
