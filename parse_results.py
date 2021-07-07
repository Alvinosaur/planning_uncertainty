import re
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Planning Uncertainty Parse Parameters')
    parser.add_argument('--results_dir', action="store", type=str, default="")
    parser.add_argument('--redirect_stdout', action="store_true")
    args = parser.parse_args()

    return args


def parse_exec_output(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "Fall Rate: (\d+.\d+), success rate: (\d+.\d+)", text)
        fall_probs = [float(v[0]) for v in results]
        success_probs = [float(v[1]) for v in results]
        num_failed_plans = len(re.findall("not found!", text))

    return fall_probs, success_probs


def parse_plan_output(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "States Expanded: (\d+).*", text)
        num_states = [float(v) for v in results]

        times = re.findall("time taken: (\d+.\d+)", text)
        times = [float(v) for v in times]

        timeouts = re.findall("time taken: NA", text)

        replans = re.findall("Execution of Plan: failure, replanning...", text)

        knock_downs = re.findall("failure, knocked over", text)

        successes = re.findall("Execution of Plan: success", text)

        total_attempts = len(times) + len(timeouts)
        timeout_rate = len(timeouts) / float(total_attempts)
        replan_rate = len(replans) / float(total_attempts)
        knock_down_rate = len(knock_downs) / float(total_attempts)
        success_rate = len(successes) / float(total_attempts)

    return sum(num_states) / len(num_states), sum(times) / len(
        times), timeout_rate, replan_rate, knock_down_rate, success_rate


if __name__ == "__main__":
    args = parse_arguments()
    if args.redirect_stdout:
        sys.stdout = open(os.path.join(args.results_dir, "parse_results.txt"), "w")

    avg_path_length = 0
    plan_count = 0
    num_plan_timeouts = 0
    avg_total_time_taken = 0
    avg_num_attempts = 0
    avg_planning_time = 0
    avg_num_timeouts = 0
    avg_exec_time = 0
    avg_expansion_count = 0
    avg_full_eval_count = 0
    avg_lazy_eval_count = 0
    avg_invalid_count = 0
    avg_planning_time_per_iter = []
    avg_planning_time_per_iter_counts = []
    success_rate = 0
    fall_rate = 0
    count = 0
    sub_count = 0
    sub_count_exec = 0  # diff because it's possible planning actually terminates without finding goal
    # if replanning starts again in a place where no motions are valid
    time_per_expansion_series = np.zeros(21)

    files = os.listdir(args.results_dir)
    for f in files:
        if f[:8] != "analysis": continue
        start_goal_idx = int(re.findall("analysis_(\d+)", f)[0])
        try:
            results = np.load(os.path.join(args.results_dir, "results_%d.npz" % start_goal_idx))
            state_path = results["state_path"]
            avg_path_length += len(state_path)
            plan_count += 1
        except:
            print("results %d doesn't exist, skipping..." % start_goal_idx)
            continue

        data = np.load(os.path.join(args.results_dir, f), allow_pickle=True)
        total_time_taken = data["total_time_taken"]
        num_attempts = data["num_attempts"]
        planning_times = data["planning_times"]
        avg_expansion_times = data["avg_expansion_times"]
        exec_times = data["exec_times"]
        expansion_counts = data["expansion_counts"]
        full_eval_counts = data["full_eval_counts"]
        lazy_eval_counts = data["lazy_eval_counts"]
        invalid_counts = data["invalid_counts"]
        is_success = data["is_success"]
        is_fallen = data["is_fallen"]

        # if first planning timed out
        if len(planning_times) == 1 and planning_times[0] is None:
            num_plan_timeouts += 1

        else:
            avg_total_time_taken += total_time_taken
            avg_num_attempts += num_attempts
            success_rate += is_success
            fall_rate += is_fallen
            count += 1

            for i in range(len(planning_times)):
                if planning_times[i] is None:
                    avg_num_timeouts += 1
                    continue

                sub_count += 1
                avg_planning_time += planning_times[i]
                time_per_expansion_series[start_goal_idx] += avg_expansion_times[i]
                if i >= len(avg_planning_time_per_iter):
                    avg_planning_time_per_iter.append(planning_times[i])
                    avg_planning_time_per_iter_counts.append(1)
                else:
                    avg_planning_time_per_iter[i] += planning_times[i]
                    avg_planning_time_per_iter_counts[i] += 1

                try:
                    avg_exec_time += exec_times[i]
                    sub_count_exec += 1
                except TypeError:
                    print("Replanning failed here, skipping this exec time")
                    pass

                avg_expansion_count += expansion_counts[i]
                avg_full_eval_count += full_eval_counts[i]
                avg_lazy_eval_count += lazy_eval_counts[i]
                avg_invalid_count += invalid_counts[i]

            time_per_expansion_series[start_goal_idx] /= sub_count

    avg_total_time_taken /= count
    avg_num_attempts /= count
    success_rate /= count
    fall_rate /= count
    avg_planning_time /= sub_count
    avg_exec_time /= sub_count_exec
    avg_expansion_count /= sub_count
    avg_full_eval_count /= sub_count
    avg_lazy_eval_count /= sub_count
    avg_invalid_count /= sub_count
    print("avg timeout rate: %.2f" % (avg_num_timeouts / (avg_num_timeouts + sub_count)))
    print("avg_path_length: %.2f" % avg_path_length)
    print("avg_total_time_taken: %.2f" % avg_total_time_taken)
    print("avg_num_attempts: %.2f" % avg_num_attempts)
    print("success_rate: %.2f" % success_rate)
    print("fall_rate: %.2f" % fall_rate)
    print("avg_planning_time: %.2f" % avg_planning_time)
    print("avg_exec_time: %.2f" % avg_exec_time)
    print("avg_expansion_count: %.2f" % avg_expansion_count)
    print("avg_full_eval_count: %.2f" % avg_full_eval_count)
    print("avg_lazy_eval_count: %.2f" % avg_lazy_eval_count)
    print("avg_invalid_count: %.2f" % avg_invalid_count)
    # plt.plot(time_per_expansion_series)
    # plt.title("Average time per expansion vs start-goal")
    # plt.show()
