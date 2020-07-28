import numpy as np


class State(object):
    def __init__(self, x, v, t):
        self.x = x
        self.v = v
        self.t = t


def gen_trapezoidal_velocity_profile(start: State, final: State, dt,
                                     duty_cycle):
    """Source:
    https://drive.google.com/file/d/1OIw3erlI6zIOfEqbsA0W1lyYh6k5vS3y/view?usp=sharing

    returns velocity profile of length N-1 for N waypoints

    Args:
        start (State): [intial angle, initial angular vel, init time]
        final (State): [final angle, final angular vel, final time]
        dqmax (float): max angular velocity
    """
    q0, dq0, t0 = start.x, start.v, start.t
    qf, dqf, tf = final.x, final.v, final.t
    assert(0 < duty_cycle <= 0.5)
    tr = duty_cycle * (tf - t0)
    vm = (qf - q0) / (tf - t0 - tr)
    ta = t0 + tr
    tb = tf - tr

    ts = np.linspace(start=t0, stop=ta,
                     num=int((ta - t0) / dt)) - t0
    # calculate ramp-up velocity profile
    ramp_up_profile = (vm / tr) * ts

    ts = tf - np.linspace(start=tb, stop=tf,
                          num=int((tf - tb) / dt))
    # calculate ramp-down velocity profile
    ramp_down_profile = (vm / tr) * ts

    # constant velocity profile
    assert (tb >= ta)
    const_profile = vm * np.ones(int((tb - ta) / dt))

    return np.concatenate([ramp_up_profile, const_profile, ramp_down_profile])


def gen_trapezoidal_velocity_profile_unbounded_time(start: State, final: State, dt,
                                                    max_a, vm):
    """Source:
    https://drive.google.com/file/d/1OIw3erlI6zIOfEqbsA0W1lyYh6k5vS3y/view?usp=sharing

    returns velocity profile of length N-1 for N waypoints

    Args:
        start (State): [intial angle, initial angular vel, init time]
        final (State): [final angle, final angular vel, final time]
        dqmax (float): max angular velocity
    """
    q0, _, t0 = start.x, start.v, start.t
    # unspecified final time
    qf, _ = final.x, final.v

    # need negative max velocity and acceleration to get negative change
    if (qf < q0):
        vm *= - 1
        max_a *= -1
    tr = vm / max_a

    tf = (qf - q0) / vm + t0 + tr
    ta = t0 + tr
    tb = tf - tr

    ts = np.linspace(start=t0, stop=ta,
                     num=int((ta - t0) / dt)) - t0
    # calculate ramp-up velocity profile
    ramp_up_profile = (vm / tr) * ts

    ts = tf - np.linspace(start=tb, stop=tf,
                          num=int((tf - tb) / dt))
    # calculate ramp-down velocity profile
    ramp_down_profile = (vm / tr) * ts

    # constant velocity profile
    assert (tb >= ta)
    const_profile = vm * np.ones(int((tb - ta) / dt))

    return np.concatenate([ramp_up_profile, const_profile, ramp_down_profile])


def test_trap_vel_profile():
    t0, tf = 4, 12
    x0, xf = 2, 9
    v0, vf = 0, 0
    dt = 0.5
    start = State(x=x0, v=v0, t=t0)
    final = State(x=xf, v=vf, t=tf)
    vmax = 10
    duty_cycle = 0.2
    # returns velocity profile of length N-1 for N waypoints
    vel_profile = gen_trapezoidal_velocity_profile(
        start, final, dt, duty_cycle=duty_cycle)

    N = int((tf - t0) / float(dt))
    ts = np.linspace(start=t0, stop=tf, num=N)
    xs = np.zeros(N)
    xs[0] = x0
    x = x0
    for i in range(N - 1):
        x += vel_profile[i] * dt
        xs[i + 1] = x
    plt.plot(ts[:-1], vel_profile, label="vel")
    plt.plot(ts, xs, label="pos")
    plt.legend(loc="upper right")
    plt.show()
