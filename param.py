import argparse
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Planning Uncertainty Parameters')
    # General
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--replay_results', action="store_true")
    parser.add_argument('--replay_dir', action="store", type=str)
    parser.add_argument('--redirect_stdout', action="store_true")

    # Type of planner
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--avg', action="store_true")

    # Replanning after execution
    parser.add_argument('--use_replan', action="store_true")

    # Planner Agnostic  simulate_prev
    parser.add_argument('--use_ee_trans_cost', action="store", type=bool, default="true")
    parser.add_argument('--max_time', action="store", type=int, default="60",
                        help="Single planner planning time limit (sec), mulitplied by N for average planner.")
    parser.add_argument('--dx', action="store", type=float, default="0.1")
    parser.add_argument('--dy', action="store", type=float, default="0.1")
    parser.add_argument('--dz', action="store", type=float, default="0.1")
    parser.add_argument('--goal_thresh', action="store", type=float, default="0.13")
    parser.add_argument('--dtheta', action="store", type=int, default="8")
    parser.add_argument('--eps', action="store", type=float, default="7")

    # Optionally specify specific start-goal pairs or even index of a solved path
    parser.add_argument('--start_goal', action="store", type=int, default="-1")
    parser.add_argument('--start_goal_range', action="store", type=str)
    parser.add_argument('--solved_index', action="store", type=int)

    # Optional load planning params from a file
    parser.add_argument('--load_params', action="store_true")
    parser.add_argument('--params_path', action="store", type=str, default="")

    # Optionally specify specific execution parameters
    parser.add_argument('--exec_param', action="store", type=int, default="-1")
    parser.add_argument('--exec_low_fric', action="store_true")
    parser.add_argument('--exec_high_fric', action="store_true")
    parser.add_argument('--exec_med_fric', action="store_true")

    # Avg Planner
    parser.add_argument('--fall_thresh', action="store", type=float, default="0")
    parser.add_argument('--n_sims', action="store", type=int, default="10")
    # Avg Planner Sampling Strategy
    parser.add_argument('--bimodal', action="store_true")
    parser.add_argument('--high_fric', action="store_true")

    # Single Planner
    parser.add_argument('--single_low_fric', action="store_true",
                        help="Single planner to use manually-specified low friction")
    parser.add_argument('--single_high_fric', action="store_true",
                        help="Single planner to use manually-specified high friction")
    parser.add_argument('--single_med_fric', action="store_true",
                        help="Single planner to use manually-specified medium friction")

    # Experimental
    # possible options: always_N, always_1, far_N, close_N
    parser.add_argument('--sim_type', action="store", type=str, default="always_N")
    parser.add_argument('--sim_dist_thresh', action="store", type=float, default="0.25")

    args = parser.parse_args()

    return args
