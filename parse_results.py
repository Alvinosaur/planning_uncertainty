import re
import numpy as np
import matplotlib.pyplot as plt


single_results = "results/results.txt"
avg_results = "avg_results/results.txt"


def parse(fname):
    with open(fname, "r") as f:
        text = f.read()
        results = re.findall(
            "Fall Rate: (\d+.\d+), success rate: (\d+.\d+)", text)
        fall_probs = [float(v[0]) for v in results]
        success_probs = [float(v[1]) for v in results]

    return fall_probs, success_probs


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
