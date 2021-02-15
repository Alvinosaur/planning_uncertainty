import re
import argparse
import os
import sys


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

        total_attempts = len(times) + len(timeouts)
        timeout_rate = len(timeouts) / float(total_attempts)
        replan_rate = len(replans) / float(total_attempts)
        knock_down_rate = len(knock_downs) / float(total_attempts)

    return sum(num_states) / len(num_states), sum(times) / len(times), timeout_rate, replan_rate, knock_down_rate


if __name__ == "__main__":
    args = parse_arguments()
    if args.redirect_stdout:
        sys.stdout = open(os.path.join(args.results_dir, "parse_results.txt"), "w")

    print("Plan Statistics:")
    avg_num_states, avg_plan_time, avg_timeout_rate, avg_replan_rate, avg_knock_down_rate = parse_plan_output(
        os.path.join(args.results_dir, "plan_output.txt"))
    print("Avg num states expanded: %.1f" % avg_num_states)
    print("Avg plan time: %.3f" % avg_plan_time)
    print("Avg timeout rate: %.3f" % avg_timeout_rate)
    print("Avg Replan rate: %.3f" % avg_replan_rate)
    print("Avg knockdown rate: %.3f" % avg_knock_down_rate)

    print("Exec Statistics:")
    fall_probs, success_probs = parse_exec_output(os.path.join(args.results_dir, "exec_output.txt"))
    print("Fall Probs:")
    print(fall_probs)
    print("Avg Fall Prob: %.3f" % (sum(fall_probs) / len(fall_probs)))
    print("Success Probs:")
    print(success_probs)
    print("Avg Success Prob: %.3f" % (sum(success_probs) / len(success_probs)))
