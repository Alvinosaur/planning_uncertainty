import re
import numpy as np
import matplotlib.pyplot as plt


single_results = "results/results.txt"
avg_results_dir = "avg_results_0.30"
avg_results = f"{avg_results_dir}/results.txt"


def parse(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "Fall Rate: (\d+.\d+), success rate: (\d+.\d+)", text)
        fall_probs = [float(v[0]) for v in results]
        success_probs = [float(v[1]) for v in results]
        num_failed_plans = len(re.findall("not found!", text))

    return fall_probs, success_probs


def parse_output(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "States Expanded: (\d+).*", text)
        num_states = [float(v) for v in results]

        times = re.findall("time taken: (\d+.\d+)", text)
        times = [float(v) for v in times]

        timeouts = re.findall("time taken: NA", text)

        total_attempts = len(times) + len(timeouts)
        timeout_rate = len(timeouts) / float(total_attempts)

    return sum(num_states) / len(num_states), sum(times) / len(times), timeout_rate


single_fall_probs, single_success_probs = parse(single_results)
avg_fall_probs, avg_success_probs = parse(avg_results)

print("Fall Probs:")
print("single:")
print(single_fall_probs)
print("avg:")
print(avg_fall_probs)
print("(Avg Fall Prob) Single: %.3f, Avg: %.3f" % (
    sum(single_fall_probs) / len(single_fall_probs),
    sum(avg_fall_probs) / len(avg_fall_probs)
))

print("Success Probs:")
print("single:")
print(single_success_probs)
print("avg:")
print(avg_success_probs)
print("(Avg Success Prob) Single: %.3f, Avg: %.3f" % (
    sum(single_success_probs) / len(single_success_probs),
    sum(avg_success_probs) / len(avg_success_probs)
))

avg_num_states, avg_times, avg_timeout_rate = parse_output(f"{avg_results_dir}/output.txt")
num_states, times, timeout_rate = parse_output("results/output.txt")
print(f"Avg num states: {avg_num_states}, plan time: {avg_times}, avg timeout rate: {avg_timeout_rate}")
print(f"Num states: {num_states}, plan time: {times}, timeout rate: {timeout_rate}")
