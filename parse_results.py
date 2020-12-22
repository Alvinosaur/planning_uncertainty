import re
import numpy as np
import matplotlib.pyplot as plt


single_results = "results/results.txt"
avg_results_dir = "avg_results_0.10_sample_2"
avg_results = f"{avg_results_dir}/results.txt"


def parse(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "Fall Rate: (\d+.\d+), success rate: (\d+.\d+)", text)
        fall_probs = [float(v[0]) for v in results]
        success_probs = [float(v[1]) for v in results]
        num_failed_plans = len(re.findall("not found!", text))

    return fall_probs, success_probs, num_failed_plans


def parse_output(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "States Expanded: (\d+).*", text)
        num_states = [float(v) for v in results]

    return sum(num_states) / len(num_states)


single_fall_probs, single_success_probs, single_num_failed_plans = parse(single_results)
avg_fall_probs, avg_success_probs, avg_num_failed_plans = parse(avg_results)

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

avg_num_states = parse_output(f"{avg_results_dir}/output.txt")
num_states = parse_output("results/output.txt")
print(f"Avg num states: {avg_num_states}, avg num timeouts: {avg_num_failed_plans}")
print(f"Num states: {num_states}, num timeouts: {single_num_failed_plans}")
