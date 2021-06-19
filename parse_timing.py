import re
import argparse
import os
import sys
import numpy as np

"""
sim_reset: 0.02229
command: 0.00317
setup_objects: 0.00058
step_sim: 0.01598
collision_time: 0.00065
check_bottle_state: 0.00380
remove_time: 0.00001
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description='Planning Uncertainty Parse Parameters')
    parser.add_argument('--results_dir', action="store", type=str, default="")
    parser.add_argument('--redirect_stdout', action="store_true")
    args = parser.parse_args()

    return args


def get_sum_avg(text, name):
    findings = re.findall(f"{name}: (\d+.\d+)", text)
    times = [float(v) for v in findings]
    return sum(times), sum(times) / len(times)


args = parse_arguments()
with open(
        os.path.join(args.results_dir, "plan_output.txt"),
        "r") as f:
    text = f.read()
    sim_reset, sim_reset_avg = get_sum_avg(text, "sim_reset")
    run_sim, run_sim_avg = get_sum_avg(text, "run_sim")
    command, command_avg = get_sum_avg(text, "command")
    setup_objects, setup_objects_avg = get_sum_avg(text, "setup_objects")
    step_sim, step_sim_avg = get_sum_avg(text, "step_sim")
    collision_time, collision_time_avg = get_sum_avg(text, "collision_time")
    check_bottle_state, check_bottle_state_avg = get_sum_avg(text, "check_bottle_state")
    remove_time, remove_time_avg = get_sum_avg(text, "remove_time")
    sim_time, sim_time_avg = get_sum_avg(text, "expansion_sim_time")

    findings = re.findall(f"Total time: (\d+.\d+)", text)
    total_times = np.array([float(v) for v in findings])
    print(total_times)

    # findings = re.findall(f"States Expanded: (\d+)", text)
    # states_expanded = np.array([int(v) for v in findings])
    # print(states_expanded)

    print("Num full evaluations:")
    findings = re.findall(f"Num full evaluations: (\d+)", text)
    num_full_evals = np.array([int(v) for v in findings])
    print(num_full_evals)
    print(np.average(num_full_evals))

    # NOTE: for non-lazy implementation, all evaluations are counted here
    print("Num lazy evaluations:")
    findings = re.findall(f"Num lazy evaluations: (\d+)", text)
    num_lazy_evals = np.array([int(v) for v in findings])
    print(num_lazy_evals)
    print(np.average(num_lazy_evals))

    print("Num total evaluations:")
    total_evals = num_full_evals + num_lazy_evals
    print(np.average(total_evals))

    print("Average planning time: %.2f" % np.average(total_times))
    print("Average time per state expanded * 1000: %.2f" % np.average((total_times / total_evals) * 1000))

    print("run_sim: %.3f" % run_sim_avg)
    print("sim_reset: %.3f" % sim_reset_avg)
    print("command: %.3f" % command_avg)
    print("setup_objects: %.3f" % setup_objects_avg)
    print("step_sim: %.3f" % step_sim_avg)
    print("collision_time: %.3f" % collision_time_avg)
    print("check_bottle_state: %.3f" % check_bottle_state_avg)
    print("remove_time: %.3f" % remove_time_avg)
    print("expansion_sim_time: %.3f" % sim_time_avg)

    count = 0
    num_results = 11
    avg_path_length = 0
    for i in range(num_results):
        fname = os.path.join(args.results_dir, "results_%d.npz" % i)
        try:
            results = np.load("%s" % fname, allow_pickle=True)
        except:
            print("results %d doesn't exist, skipping..." % i)
            continue

        count += 1
        state_path = results["state_path"]
        avg_path_length += len(state_path)

    print("Average path length:")
    print(avg_path_length / count)

# python main.py --avg --n_sims 10 --max_time 720 --sim_type always_N --bimodal --exec_low_fric --single_med_fric --visualize --replay_results --replay_dir /home/alvin/research/planning_uncertainty/avg_bimodal/2021-06-05@11:21:54.758195_fall_thresh_0.20
