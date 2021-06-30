import argparse
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Planning Uncertainty Parameters')
    # General
    parser.add_argument('--visualize', action="store_true")
    parser.add_argument('--replay_results', action="store_true")
    parser.add_argument('--replay_dir', action="store", type=str)
    parser.add_argument('--redirect_stdout', action="store_true")
    parser.add_argument('--load_params', action="store_true")
    parser.add_argument('--save_global_params', action="store_true")
    parser.add_argument('--params_path', action="store", type=str, default="")
    parser.add_argument('--save_edge_betas', action="store_true")  # save data for learned alpha, beta priors

    # Type of planner
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--full', action="store_true")
    parser.add_argument('--lazy', action="store_true")
    parser.add_argument('--beta', action="store_true")

    # Replanning after execution
    parser.add_argument('--use_replan', action="store_true")

    # Planning Details
    parser.add_argument('--fall_thresh', action="store", type=float, default="0.1")
    parser.add_argument('--beta_var_thresh', action="store", type=float, default="0.02")

    # Simulation Parameters
    parser.add_argument('--num_exec_sims', action="store", type=int, default="20")
    parser.add_argument('--num_plan_sims', action="store", type=int, default="10")
    parser.add_argument('--single_low_fric', action="store_true",
                        help="Single planner to use manually-specified low friction")
    parser.add_argument('--single_high_fric', action="store_true",
                        help="Single planner to use manually-specified high friction")
    parser.add_argument('--single_med_fric', action="store_true",
                        help="Single planner to use manually-specified medium friction")

    # DO NOT CHANGE:
    parser.add_argument('--use_ee_trans_cost', action="store", type=bool, default="true")
    parser.add_argument('--max_time', action="store", type=int, default="720",
                        help="Single planner planning time limit (sec), mulitplied by N for average planner.")
    parser.add_argument('--dx', action="store", type=float, default="0.1")
    parser.add_argument('--dy', action="store", type=float, default="0.1")
    parser.add_argument('--dz', action="store", type=float, default="0.1")
    parser.add_argument('--goal_thresh', action="store", type=float, default="0.13")
    parser.add_argument('--dtheta', action="store", type=int, default="8")
    parser.add_argument('--eps', action="store", type=float, default="7")

    # Generate random start-goal files
    parser.add_argument('--start_goal_fname', action="store", type=str)

    # DEBUG:
    # Optionally specify specific start-goal pairs or even index of a solved path
    parser.add_argument('--start_goal', action="store", type=int, default="-1")
    parser.add_argument('--start_goal_range', action="store", type=str)
    parser.add_argument('--solved_path_index', action="store", type=int)

    # LATER:
    # Optionally specify specific execution parameters
    parser.add_argument('--exec_param', action="store", type=int, default="-1")
    parser.add_argument('--exec_low_fric', action="store_true")
    parser.add_argument('--exec_high_fric', action="store_true")
    parser.add_argument('--exec_med_fric', action="store_true")
    parser.add_argument('--exec_all_fric', action="store_true")

    args = parser.parse_args()

    return args
