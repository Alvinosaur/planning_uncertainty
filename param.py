import argparse
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Planning Uncertainty Parameters')
    # General
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--replay_results', action="store_true")
    parser.add_argument('--replay_dir', action="store", type=str, default="")
    parser.add_argument('--redirect_stdout', action="store_true")

    # Type of planner
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--avg', action="store_true")

    # Planner Agnostic
    parser.add_argument('--max_time', action="store", type=int, default="30",
                        help="Single planner planning time limit (sec), mulitplied by N for average planner.")
    parser.add_argument('--dx', action="store", type=float, default="0.1")
    parser.add_argument('--dy', action="store", type=float, default="0.1")
    parser.add_argument('--dz', action="store", type=float, default="0.1")
    parser.add_argument('--goal_thresh', action="store", type=float, default="0.1")
    parser.add_argument('--dtheta', action="store", type=int, default="8")
    parser.add_argument('--eps', action="store", type=float, default="5")
    parser.add_argument('--load_params', action="store_true")

    # Avg Planner
    parser.add_argument('--fall_thresh', action="store", type=float, default="0.2")
    parser.add_argument('--n_sims', action="store", type=int, default="10")
    # Avg Planner Sampling Strategy
    parser.add_argument('--bimodal', action="store_true")
    parser.add_argument('--high_fric', action="store_true")

    # Single Planner
    parser.add_argument('--single_low_fric', action="store_true",
                        help="Single planner to use manually-specified low friction")
    parser.add_argument('--single_high_fric', action="store_true",
                        help="Single planner to use manually-specified high friction")

    args = parser.parse_args()

    return args
