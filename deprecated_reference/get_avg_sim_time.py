import re
log = """
Time(s) to run one sim: 0.147
Time(s) to run one sim: 0.154
Time(s) to run one sim: 0.158
Time(s) to run one sim: 0.148
Time(s) to run one sim: 0.140
Time(s) to run one sim: 0.150
Time(s) to run one sim: 0.144
Time(s) to run one sim: 0.145
Time(s) to run one sim: 0.163
Time(s) to run one sim: 0.156
Time(s) to run one sim: 0.158
Time(s) to run one sim: 0.158
Time(s) to run one sim: 0.139
Time(s) to run one sim: 0.134
Time(s) to run one sim: 0.119
Time(s) to run one sim: 0.137
Time(s) to run one sim: 0.136
Time(s) to run one sim: 0.118
Time(s) for one state: 31.255
Percent complete: 0.67
Time(s) to run one sim: 0.168
Time(s) to run one sim: 0.169
Time(s) to run one sim: 0.161
Time(s) to run one sim: 0.136
Time(s) to run one sim: 0.136
Time(s) to run one sim: 0.119
Time(s) to run one sim: 0.154
Time(s) to run one sim: 0.159
Time(s) to run one sim: 0.167
Time(s) to run one sim: 0.153
Time(s) to run one sim: 0.166
Time(s) to run one sim: 0.159
Time(s) to run one sim: 0.124
Time(s) to run one sim: 0.116
Time(s) to run one sim: 0.121
Time(s) to run one sim: 0.133
Time(s) to run one sim: 0.134
Time(s) to run one sim: 0.122
Time(s) for one state: 31.169
Percent complete: 1.00
Time(s) to run one sim: 0.169
Time(s) to run one sim: 0.160
Time(s) to run one sim: 0.162
Time(s) to run one sim: 0.155
Time(s) to run one sim: 0.144
Time(s) to run one sim: 0.103
Time(s) to run one sim: 0.137
Time(s) to run one sim: 0.127
Time(s) to run one sim: 0.105
Time(s) to run one sim: 0.152
Time(s) to run one sim: 0.156
Time(s) to run one sim: 0.159
Time(s) to run one sim: 0.127
Time(s) to run one sim: 0.135
Time(s) to run one sim: 0.128
Time(s) to run one sim: 0.134
Time(s) to run one sim: 0.127
Time(s) to run one sim: 0.119
"""

lines = log.split("\n")
pattern = r"Time\(s\) to run one sim: (\d+\.\d+)"
avg = 0
count = 0
for line in lines:
    matches = re.findall(pattern, line)
    if len(matches) > 0:
        avg += float(matches[0])
        count += 1

print("Avg: %.3f" % (avg/count))
