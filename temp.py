import time
import math
from contextlib import contextmanager
import signal
import sys

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Enforces max allotted time for planning, execution, and replanning
    Args:
        seconds ([type]): [description]
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def f(n):
    for i in range(n):
        print(i)
        time.sleep(1)

# for j in [3, 7]:
#     try:
#         with time_limit(5):
#             f(j)
#     except:
#         continue

if __name__=="__main__":
    print("python " + " ".join(sys.argv))
