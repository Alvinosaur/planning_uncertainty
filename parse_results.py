import re


# with open("normal_results.txt", "r") as f:
#     # with open("avg_results.txt", "r") as f:
#     text = f.read()
#     matches = re.findall("States Expanded: (\d+), found goal: 1", text)
#     vals = [int(v) for v in matches]
#     print(sum(vals) / float(len(vals)))

successes = []
falls = []
# with open("normal_exec_results.txt", "r") as f:
with open("avg_exec_results.txt", "r") as f:
    text = f.read()
    lines = text.split("\n")
    for l in lines:
        matches = re.findall(
            "Fall Rate: (\d*\.?\d+), success rate: (\d*\.?\d+)", l)
        if len(matches) > 0:
            matches = matches[0]
            fall_rate = float(matches[0])
            success_rate = float(matches[1])

            falls.append(fall_rate)
            successes.append(success_rate)

print("Avg Fall Rate: %.3f" % (sum(falls) / len(falls)))
print("Avg Success Rate: %.3f" % (sum(successes) / len(successes)))
