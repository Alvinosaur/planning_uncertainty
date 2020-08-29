import pybullet as p
import pybullet_data
import math
import numpy as np
import time
import signal
from contextlib import contextmanager
import sys
import typing as t


from sim_objects import Bottle, Arm
from environment import Environment, EnvParams
from naive_joint_space_planner import NaivePlanner, SINGLE, AVG, MODE


class PlanExecMetrics(object):
    def __init__(self):
        self.is_success = False
        self.is_fallen = False
        self.found_plan = False
        self.num_replan_attempts = 0
        self.initial_plan_time = 0  # first plan's runtime
        self.total_plan_time = 0  # includes first plan and any replans
        self.start_plan_time = 0  # used to measure plan time if times out
        self.end_plan_time = 0


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def replan_exec_loop(env: Environment, planner: NaivePlanner,
                     bottle_pos, bottle_ori, start, goal,
                     exec_params: EnvParams,
                     metrics: PlanExecMetrics):
    policy = None
    while (not metrics.is_success and not metrics.is_fallen):
        # don't consider first plan as a "replan"
        is_first_plan = (policy is None)
        if not is_first_plan:
            metrics.num_replan_attempts += 1

        # planner stores its own set of simulation parameters whether it is one
        # or multiple
        print("Planning @ start %s" % planner.state_to_str(start))
        metrics.start_plan_time = time.time()
        _, policy = planner.plan(start=start, goal=goal)
        metrics.end_plan_time = time.time()

        # add time of planning
        metrics.total_plan_time += (
            metrics.end_plan_time - metrics.start_plan_time)
        if is_first_plan:
            metrics.initial_plan_time = (
                metrics.end_plan_time - metrics.start_plan_time)

        print("Found plan!")
        metrics.found_plan = True

        # reset arm to original position to execute new policy
        env.arm.resetEE()

        print("Exec params: fill: %.2f, fric: %.2f" %
              (exec_params.bottle_fill, exec_params.bottle_fric))

        # When executing plan, use different environment parameters to examine
        # how robust planner is
        for dq in policy:
            trans_cost, bottle_pos, bottle_ori, joint_pos = env.run_sim(
                action=dq, sim_params=exec_params, bottle_pos=bottle_pos, bottle_ori=bottle_ori)
            print("Execution: %s" % planner.state_to_str(bottle_pos))

            # immediately terminate if bottle fell during execution
            if trans_cost == env.FALL_COST:
                metrics.is_fallen = True
                break

        final_state = np.concatenate([bottle_pos, joint_pos])
        # final state in x, y might be goal even if bottle fell, so need
        # is_fallen
        metrics.is_success = planner.reached_goal(
            final_state) and not metrics.is_fallen
        print(final_state)
        print("Execution successful: %d, bottle fell: %d" %
              (metrics.is_success, metrics.is_fallen))

        # set new start as current state if need to replan
        start = final_state


def simulate_planner_random_env(env: Environment, planner: NaivePlanner,
                                start, goal,
                                exec_params_set: t.List[EnvParams],
                                plan_params_set_per_iter: t.List[t.List[EnvParams]],
                                iters, max_time_s=30):
    """Over iters number of iterations, randomly sample various environment
    parameters from normal distributions and execute the original planned
    policy in that environment. If the plan doesn't fail, but also doesn't
    reach the desired goal, have the planner replan and re-execute until alotted
    amount of time has passed.

    Args:
        env (Environment): [description]
        planner (NaivePlanner): [description]
        policy (list): [description]

        iters (int, optional): [description]
    """
    assert(len(plan_params_set_per_iter) == iters)
    assert(len(exec_params_set) == iters)

    success_count = 0
    fall_count = 0
    failed_plan_count = 0
    replan_count = 0
    total_initial_plan_time = 0
    total_plan_time = 0

    # initial bottle position and arm state reset each iteration
    init_bottle_pos = planner.bottle_pos_from_state(start)
    init_bottle_ori = np.array([0, 0, 0, 1])

    for i in range(iters):
        # set new planning parameters
        # SINGLE just uses the first param in the list
        plan_params_set = plan_params_set_per_iter[i]
        planner.sim_params_set = plan_params_set

        exec_params = exec_params_set[i]

        # store info about success of planning and execution
        metrics = PlanExecMetrics()

        # reset arm state
        env.arm.resetEE()

        # set hard time limit for replan and execution
        try:
            with time_limit(max_time_s):
                replan_exec_loop(
                    env=env, planner=planner, bottle_pos=init_bottle_pos, bottle_ori=init_bottle_ori, start=start, goal=goal,
                    exec_params=exec_params,
                    metrics=metrics)
        except TimeoutException:
            print("Plan/Exec Loop Timed out!")

            # if timed out during planning, then we know end_time < start_time
            # since new end_time is either 0 or outdated
            if metrics.end_plan_time < metrics.start_plan_time:
                untracked_plan_time = time.time() - metrics.start_plan_time
                metrics.total_plan_time += untracked_plan_time

                # if initial plan time is 0, then first initial plan didn't complete
                if metrics.initial_plan_time == 0:
                    metrics.initial_plan_time = untracked_plan_time

        failed_plan_count += (not metrics.found_plan)
        if metrics.found_plan:
            success_count += metrics.is_success
            fall_count += metrics.is_fallen
            replan_count += metrics.num_replan_attempts
            total_initial_plan_time += metrics.initial_plan_time
            total_plan_time += metrics.total_plan_time

        else:
            print("Planning failed!")

    # combine all metrics
    fail_plan_rate = failed_plan_count / float(iters)

    # only consider success, fall, and replan for successful initial plans
    found_plan_count = iters - failed_plan_count
    success_rate = success_count / float(found_plan_count)
    fall_rate = fall_count / float(found_plan_count)
    avg_num_replan_attempts = replan_count / float(found_plan_count)
    avg_initial_plan_time = total_initial_plan_time / float(found_plan_count)
    avg_total_plan_time = total_plan_time / float(found_plan_count)

    print("Results for %d iters and max time(s): %d" % (iters, max_time_s))
    printout = "Success rate: %.2f, " % success_rate
    printout += "fall rate: %.2f, " % fall_rate
    printout += "fail plan rate: %.2f, " % fail_plan_rate
    printout += "avg num replans: %.2f, " % avg_num_replan_attempts
    printout += "avg_initial_plan_time: %.2f, " % avg_initial_plan_time
    printout += "avg_total_plan_time: %.2f, " % avg_total_plan_time
    print(printout)


def direct_plan_execution(start, goal, planner: NaivePlanner, env: Environment,
                          exec_params=None,
                          replay_saved=False, visualize=False, sim_mode=SINGLE,
                          replay_random=False):
    if sim_mode == SINGLE:
        filename = "results"
    elif sim_mode == AVG:
        filename = "results_avg"
    else:
        filename = "results_mode"

    if not replay_saved:
        state_path, policy = planner.plan(start=start, goal=goal)
        np.savez(filename, state_path=state_path, policy=policy)

    else:
        results = np.load("%s.npz" % filename)
        policy = results["policy"]
        print(len(policy))
        state_path = results["state_path"]

    if exec_params is None:
        exec_params = planner.sim_params_set[0]

    # set random bottle parameters
    if replay_random:
        # rand_fill = np.random.normal(
        #     loc=env.mean_fillp, scale=env.std_fillp)
        # rand_fill = np.clip(rand_fill, env.min_fill, env.max_fill)
        # rand_fric = np.random.normal(
        #     loc=env.mean_friction, scale=env.std_friction)
        # rand_fric = np.clip(rand_fric, env.min_fric, env.max_fric)
        rand_fill = env.min_fill
        rand_fric = env.min_fric

        # set random parameters
        env.bottle.set_fill_proportion(rand_fill)
        env.bottle.lat_fric = rand_fric
        env.bottle.create_sim_bottle()

    if visualize:
        # print(policy)
        bottle_pos = planner.bottle_pos_from_state(start)
        init_joints = planner.joint_pose_from_state(start)
        env.arm.reset(init_joints)
        bottle_ori = np.array([0, 0, 0, 1])
        for dq in policy:
            # run deterministic simulation for now
            # init_joints not passed-in because current joint state
            # maintained by simulator
            # print(bottle_pos)
            # print(bottle_ori)
            trans_cost, bottle_pos, bottle_ori, _ = env.run_sim(
                action=dq, bottle_pos=bottle_pos, bottle_ori=bottle_ori, sim_params=exec_params)
            print("Action: %s" % planner.state_to_str(dq * 180 / math.pi))
            print("Pos: %.2f,%.2f" %
                  tuple(bottle_pos[:2]))
            # print("Pos: %.2f,%.2f" %
            #       tuple(bottle_pos)[:2])
            print(np.linalg.norm(bottle_pos[:2] -
                                 planner.bottle_pos_from_state(goal)[:2]))

    elif not visualize and replay_saved:
        print("Trying to playback plan without visualizing!")
        exit()


def main():
    VISUALIZE = False
    REPLAY_RESULTS = False
    SAVE_STDOUT_TO_FILE = False
    sim_mode = SINGLE
    replay_random = False  # replay with  random bottle parameters chosen
    LOGGING = False
    GRAVITY = -9.81
    if VISUALIZE:
        p.connect(p.GUI)  # or p.DIRECT for nongraphical version
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, GRAVITY)
    planeId = p.loadURDF(Environment.plane_urdf_filepath,
                         basePosition=[0, 0, 0])
    kukaId = p.loadURDF(Environment.arm_filepath, basePosition=[0, 0, 0])
    if LOGGING and VISUALIZE:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, "temp.mp4")

    # bottle
    # bottle_start_pos = np.array(
    #     [-0, -0.6, Bottle.INIT_PLANE_OFFSET]).astype(float)
    # bottle_goal_pos = np.array([-0.6, -0.2, 0]).astype(float)
    bottle_start_pos = np.array(
        [0.5, 0.5, Bottle.INIT_PLANE_OFFSET]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # defining euclidean distance dimensionality for heuristic and transition costs
    use_3D = True  # use 3D euclidean distance

    if VISUALIZE:
        # visualize a vertical blue line representing goal pos of bottle
        vertical_offset = np.array([0, 0, 0.5])  # just to make line vertical
        Environment.draw_line(lineFrom=bottle_goal_pos,
                              lineTo=bottle_goal_pos + vertical_offset,
                              lineColorRGB=[0, 0, 1], lineWidth=1,
                              lifeTime=0)

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.5, 0.3, 0.2])
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    max_force = 20  # N
    arm = Arm(EE_start_pos=EE_start_pos,
              start_ori=base_start_ori,
              kukaId=kukaId,
              max_force=max_force)
    start_joints = arm.joint_pose

    start = np.concatenate(
        [bottle_start_pos, start_joints])
    # goal joints are arbitrary and populated later in planner
    goal = np.concatenate(
        [bottle_goal_pos, [0] * arm.num_joints])
    xbounds = [-0.4, -0.9]
    ybounds = [-0.1, -0.9]
    dx = dy = dz = 0.05
    da_rad = 15 * math.pi / 180.0
    dist_thresh = np.linalg.norm([dx, dy, dz])
    state_disc = np.concatenate([[dx, dy, dz], [da_rad] * arm.num_DOF])
    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert(dist_thresh >= dx)
    eps = 40

    # run planner and visualize result
    env = Environment(arm, bottle, state_disc, is_viz=VISUALIZE,
                      use_3D=use_3D, min_iters=50)

    # randomly sampled environment parameters to test robustness of plans
    # each type of planning will use same randomly generated test params
    num_iters = 10
    exec_params_set = env.gen_random_env_param_set(num=num_iters)

    # for each iteration, have a list[list[sim_params]] so each iteration
    # planner has a new set of planning params so the planner doesn't fail every
    # single time with the same set of planning params
    # planner normally has its own auto-generated set, but this is for
    # explicitly comparing performance of different plannners
    num_rand_samples = 2
    plan_params_set_per_iter = []
    for i in range(num_iters):
        new_set = env.gen_random_env_param_set(num=num_rand_samples)
        plan_params_set_per_iter.append(new_set)

    # Create the three types of planners
    single_planner = NaivePlanner(env, xbounds,
                                  ybounds, dist_thresh, eps, da_rad=da_rad,
                                  dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=SINGLE, num_rand_samples=1)
    avg_planner = NaivePlanner(env, xbounds,
                               ybounds, dist_thresh, eps, da_rad=da_rad,
                               dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=AVG,
                               num_rand_samples=num_rand_samples)

    mode_planner = NaivePlanner(env, xbounds,
                                ybounds, dist_thresh, eps, da_rad=da_rad,
                                dx=dx, dy=dy, dz=dz, use_3D=use_3D,
                                sim_mode=MODE,
                                num_rand_samples=num_rand_samples)
    # make sure mode and average use the same set of planning env params to
    # reduce number of independent variables in experiment
    mode_planner.sim_params_set = avg_planner.sim_params_set

    max_time_s = 30
    # planners = [single_planner, avg_planner, mode_planner]
    # planners = [avg_planner]
    # names = ["avg", "avg", "mode"]
    # for pi, planner in enumerate(planners):
    #     name = names[pi]
    #     if SAVE_STDOUT_TO_FILE:
    #         sys.stdout = open('%s_planner_output.txt' % name, 'w')
    #     simulate_planner_random_env(env=env, planner=planner,
    #                                 start=start, goal=goal,
    #                                 exec_params_set=exec_params_set,
    #                                 plan_params_set_per_iter=plan_params_set_per_iter,
    #                                 iters=num_iters, max_time_s=max_time_s)
    start_plan_time = time.time()
    exec_params = exec_params_set[0]
    direct_plan_execution(start, goal, single_planner, env,
                          #   exec_params=exec_params,
                          replay_saved=REPLAY_RESULTS, visualize=VISUALIZE, sim_mode=sim_mode, replay_random=replay_random)
    end_time = time.time()
    # print("Time taken: %.2f" % (end_time - start_plan_time))
    # s1 = np.array([-0.50, -0.50, 0.04, 0.00, 0.00, -0.00, 1.00,
    #                0.51, 2.09, -0.11, 0.45, -0.14, 2.08, -0.91])
    # s2 = np.array([-0.50, -0.50, 0.04, -0.00, 0.00, -0.00, 1.00,
    #                0.51, 2.09, -0.11, 0.46, -0.14, 2.08, -0.93])
    # print(planner.state_to_key(s1))
    # print(planner.state_to_key(s2))


if __name__ == "__main__":
    main()
    # test_quaternion_discretization()
    # test_state_indexing()
