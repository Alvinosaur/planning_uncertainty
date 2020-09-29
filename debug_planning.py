import pybullet as p
import pybullet_data
import math
import numpy as np
import time
import signal
from contextlib import contextmanager
import sys
import typing as t
import os
import pickle


from sim_objects import Bottle, Arm
from environment import Environment, EnvParams
from naive_joint_space_planner import NaivePlanner, SINGLE, AVG, MODE


class PlanExecMetrics(object):
    def __init__(self):
        self.is_success = False  # if executed plan was successful in reaching goal
        self.is_fallen = False  # if executed plan caused bottle to fall over
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


def policy_to_full_traj(init_joints, policy):
    cur_joints = np.copy(init_joints)
    piecewise_trajs = []
    for (dq_vec, num_iters) in policy:
        target_joints = cur_joints + dq_vec
        traj = np.linspace(
            start=cur_joints, stop=target_joints, num=num_iters)
        piecewise_trajs.append(traj)
        cur_joints = target_joints
    full_arm_traj = np.vstack(piecewise_trajs)
    return full_arm_traj


def replan_exec_loop(env: Environment, planner: NaivePlanner,
                     bottle_pos, bottle_ori, start, goal,
                     exec_params: EnvParams,
                     metrics: PlanExecMetrics):
    policy = None

    # keep trying to replan and execute to reach goal
    # stop if bottle fell during execution, immediate failure
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

        # separately track 1st plan time as well as 1st plan + replan time
        metrics.total_plan_time += (
            metrics.end_plan_time - metrics.start_plan_time)
        if is_first_plan:
            metrics.initial_plan_time = (
                metrics.end_plan_time - metrics.start_plan_time)

        # code never reaches here if time limit expires during planning
        print("Found plan!")
        metrics.found_plan = True

        # reset arm to original position to execute new policy
        env.arm.resetEE()

        print("Exec params: fill: %.2f, fric: %.2f" %
              (exec_params.bottle_fill, exec_params.bottle_fric))

        # combine a policy into one smooth trajectory
        init_joints = np.array(planner.joint_pose_from_state(start))
        if len(policy) == 0:
            print("Policy length == 0 but planning completed?")
            raise(Exception("Policy = 0 returned!"))
        full_arm_traj = policy_to_full_traj(init_joints, policy)

        # When executing plan, use different environment parameters to examine
        # how robust planner is

        metrics.is_fallen, is_collision, bottle_pos, bottle_ori, joint_pos = (
            env.simulate_plan(init_pose=init_joints,
                              traj=full_arm_traj, bottle_pos=bottle_pos,
                              bottle_ori=bottle_ori,
                              use_vel_control=planner.use_vel_control,
                              sim_params=exec_params))
        print("Execution: %s" % planner.state_to_str(bottle_pos))

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
        planner.change_param_set(plan_params_set)

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
        except:
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
    if found_plan_count == 0:
        success_rate = fall_rate = avg_num_replan_attempts = avg_initial_plan_time = avg_total_plan_time = 0
    else:
        success_rate = success_count / float(found_plan_count)
        fall_rate = fall_count / float(found_plan_count)
        avg_num_replan_attempts = replan_count / float(found_plan_count)
        avg_initial_plan_time = total_initial_plan_time / \
            float(found_plan_count)
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
        results = dict(state_path=state_path, policy=policy,
                       plan_params=planner.sim_params_set)
        pickle.dump(results, open("%s.obj" % filename, "wb"))

        if exec_params is None:
            exec_params = planner.sim_params_set[0]

    else:
        with open("%s.obj" % filename, "rb") as f:
            results = pickle.load(f)
        policy = results["policy"]
        state_path = results["state_path"]
        plan_params = results["plan_params"]

        if exec_params is None:
            exec_params = plan_params[0]

    if visualize:
        # print(policy)
        bottle_pos = planner.bottle_pos_from_state(start)
        init_joints = planner.joint_pose_from_state(start)
        env.arm.reset(init_joints)
        bottle_ori = np.array([0, 0, 0, 1])

        # combine a policy into one smooth trajectory
        init_joints = np.array(planner.joint_pose_from_state(start))
        full_arm_traj = policy_to_full_traj(init_joints, policy)
        is_fallen, is_collision, bottle_pos, bottle_ori, joint_pos = (
            env.simulate_plan(init_pose=init_joints,
                              traj=full_arm_traj, bottle_pos=bottle_pos,
                              bottle_ori=bottle_ori,
                              use_vel_control=planner.use_vel_control,
                              sim_params=exec_params))

    elif not visualize and replay_saved:
        print("Trying to playback plan without visualizing!")
        exit()


def show_env_param_failure_distrib(env, xbounds, ybounds, dist_thresh, eps,
                                   da_rad, dx, dy, dz, use_3D, start, goal):
    step_fill = (env.max_fill - env.min_fill) / 10.0
    fill_range = np.arange(env.min_fill, env.max_fill + 0.001, step_fill)
    step_fric = (env.max_fric - env.min_fric) / 10.0
    fric_range = np.arange(env.min_fric, env.max_fric + 0.001, step_fric)
    failures = np.zeros(shape=(11, 11))  # fill x fric
    sys.stdout = open('temp.txt', 'w')
    for filli, fill in enumerate(fill_range):
        for frici, fric in enumerate(fric_range):
            single_planner = NaivePlanner(env, xbounds,
                                          ybounds, dist_thresh, eps, da_rad=da_rad,
                                          dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=SINGLE, num_rand_samples=1)
            new_sim_params_set = [EnvParams(
                bottle_fill=fill, bottle_fric=fric, bottle_fill_prob=0, bottle_fric_prob=0)]
            single_planner.change_param_set(new_sim_params_set)
            try:
                with time_limit(20):
                    direct_plan_execution(start, goal, single_planner, env,
                                          #   exec_params=exec_params,
                                          replay_saved=False, visualize=False, sim_mode=SINGLE, replay_random=False)
                    print("%s SUCCEEDED!" % single_planner.sim_params_set)
            except TimeoutException:
                failures[filli, frici] = 1
                print("%s FAILED!" % single_planner.sim_params_set)

    print(fill_range)
    print(fric_range)
    print(failures)


def test_plan_exec(env: Environment, planner: NaivePlanner,
                   start, goal, exec_params):
    # initial bottle position and arm state reset each iteration
    init_bottle_pos = planner.bottle_pos_from_state(start)
    init_bottle_ori = np.array([0, 0, 0, 1])

    metrics = PlanExecMetrics()  # unused

    replan_exec_loop(env=env, planner=planner, bottle_pos=init_bottle_pos,
                     bottle_ori=init_bottle_ori, start=start, goal=goal,
                     exec_params=exec_params,
                     metrics=metrics)


def test_env_params():
    use_3D = True  # use 3D euclidean distance heuristic and transition costs
    VISUALIZE = True
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
        [0.7, 0.1, 0]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.5, 0.3, Arm.EE_min_height])
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    max_force = 200  # N
    arm = Arm(EE_start_pos=EE_start_pos,
              start_ori=base_start_ori,
              kukaId=kukaId,
              max_force=max_force)
    start_joints = arm.joint_pose

    xbounds = [-0.4, -0.9]
    ybounds = [-0.1, -0.9]
    dx = dy = dz = 0.05
    da_rad = 15 * math.pi / 180.0
    dist_thresh = np.linalg.norm([dx, dy]) * 2
    state_disc = np.concatenate([[dx, dy, dz], [da_rad] * arm.num_DOF])
    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert(dist_thresh >= dx)
    eps = 40

    # run planner and visualize result
    env = Environment(arm, bottle, state_disc, is_viz=VISUALIZE,
                      use_3D=use_3D, min_iters=50, max_iters=300)

    # Create the three types of planners
    use_vel_control = False
    # only mark failure if every single sim of action failed
    fall_proportion_thresh = 1.0
    # iters_per_traj_set = [150, 175, 200]
    iters_per_traj_set = [50, 80, 150]
    planner = NaivePlanner(env, xbounds,
                           ybounds, dist_thresh, eps, da_rad=da_rad,
                           dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=SINGLE, num_rand_samples=1,
                           use_vel_control=use_vel_control,
                           fall_proportion_thresh=fall_proportion_thresh,
                           iters_per_traj_set=iters_per_traj_set)

    # Create many different object types
    # scale_x = np.linspace(start=0.8, stop=1.2, num=1, endpoint=True)
    scale_x = [0.8]
    scale_y = np.linspace(start=0.6, stop=0.8, num=3, endpoint=True)
    scale_z = np.linspace(start=1.0, stop=1.2, num=3, endpoint=True)
    for sy in scale_y:
        for sx in scale_x:
            for sz in scale_z:
                mesh_scale = [sx, sy, sz]
                print(mesh_scale)
                col_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=os.path.join(
                                                    bottle.folder, "cup.obj"),
                                                meshScale=mesh_scale)
                fill_props = [0.1, 0.4, 0.7, 1.0]
                # frics = [0.02, 0.03, 0.04]
                # frics = [0.1]  Too high, cap should be < this
                frics = [0.03, 0.09]  # this is a good cap
                combos = [(fric, fill)
                          for fric in frics for fill in fill_props]
                for fric, fill in combos:
                    env.bottle.set_fill_proportion(fill)
                    env.bottle.lat_fric = fric
                    print("Fill: %.2f, fric: %.2f" %
                          (fill, fric))

                    for ai in [0, planner.A.actions_mat.shape[0], 2 * planner.A.actions_mat.shape[0]]:
                        print(ai)
                        env.arm.resetEE()
                        env.bottle.create_sim_bottle(
                            new_shape=(col_id, mesh_scale))

                        # action defined as an offset of joint angles of arm
                        action = planner.A.get_action(ai)
                        print("Num Iters: %d" % action[1])

                        planner.sim_func(
                            action=action, init_joints=start_joints,
                            bottle_pos=bottle_start_pos,
                            bottle_ori=bottle_start_ori,
                            sim_params=planner.sim_params_set[0],
                            use_vel_control=False)


def main():
    use_3D = True  # use 3D euclidean distance heuristic and transition costs
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
        [0.5, 0.5, 0]).astype(float)
    bottle_goal_pos = np.array([0.2, 0.6, 0]).astype(float)
    bottle_start_ori = np.array([0, 0, 0, 1]).astype(float)
    bottle = Bottle(start_pos=bottle_start_pos, start_ori=bottle_start_ori)

    # starting end-effector pos, not base pos
    EE_start_pos = np.array([0.5, 0.3, Arm.EE_min_height])
    base_start_ori = np.array([0, 0, 0, 1]).astype(float)
    max_force = 200  # N
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
    dist_thresh = np.linalg.norm([dx, dy]) * 2
    state_disc = np.concatenate([[dx, dy, dz], [da_rad] * arm.num_DOF])
    # if  the below isn't true, you're expecting bottle to fall in exactly
    # the same state bin as the goal
    assert(dist_thresh >= dx)
    eps = 40

    # run planner and visualize result
    env = Environment(arm, bottle, state_disc, is_viz=VISUALIZE,
                      use_3D=use_3D, min_iters=50, max_iters=300)

    # randomly sampled environment parameters to test robustness of plans
    # each type of planning will use same randomly generated test params
    num_iters = 5
    exec_params_set = env.gen_random_env_param_set(num=num_iters)

    # for each iteration, have a list[list[sim_params]] so each iteration
    # planner has a new set of planning params so the planner doesn't fail every
    # single time with the same set of planning params
    # planner normally has its own auto-generated set, but this is for
    # explicitly comparing performance of different plannners
    num_rand_samples = 5
    save_sim_params = False
    if save_sim_params:
        plan_params_sets = []
        for i in range(num_iters):
            new_set = env.gen_random_env_param_set(num=num_rand_samples)
            plan_params_sets.append(new_set)
        with open("exec_params_set.obj", "wb") as f:
            exec_plan_params = dict(exec_params_set=exec_params_set,
                                    plan_params_sets=plan_params_sets)
            pickle.dump(exec_plan_params, f)
    else:
        with open("exec_params_set.obj", "rb") as f:
            exec_plan_params = pickle.load(f)
            exec_params_set = exec_plan_params["exec_params_set"]
            plan_params_sets = exec_plan_params["plan_params_sets"]

    target = 19  # in total configs 0 - 19 have been tested and validated as feasible
    with open("feasible_configs.obj", "rb") as f:
        feasible_configs = pickle.load(f)
        start, goal = feasible_configs["filtered_start_goal_pairs"][target]
        # start[0] += 0.1
        # start[1] += 0.2
        # goal[0] *= -1
        # goal[0] -= 0.2
        # goal[1] -= 0.1
        # print(goal)

        feasible_configs["filtered_start_goal_pairs"][target] = (start, goal)
        # del feasible_configs["filtered_start_goal_pairs"][target]

    # with open("feasible_configs.obj", "wb") as f:
    #     pickle.dump(feasible_configs, f)

    start_goals = [feasible_configs["filtered_start_goal_pairs"][target]]

    # Create the three types of planners
    use_vel_control = False
    # only mark failure if every single sim of action failed
    fall_proportion_thresh = 1.0
    # iters_per_traj_set = [150, 175, 200]
    iters_per_traj_set = [200]
    single_planner = NaivePlanner(env, xbounds,
                                  ybounds, dist_thresh, eps, da_rad=da_rad,
                                  dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=SINGLE, num_rand_samples=1,
                                  use_vel_control=use_vel_control,
                                  fall_proportion_thresh=fall_proportion_thresh,
                                  iters_per_traj_set=iters_per_traj_set)
    avg_planner = NaivePlanner(env, xbounds,
                               ybounds, dist_thresh, eps, da_rad=da_rad,
                               dx=dx, dy=dy, dz=dz, use_3D=use_3D, sim_mode=AVG,
                               num_rand_samples=num_rand_samples,
                               use_vel_control=use_vel_control,
                               fall_proportion_thresh=fall_proportion_thresh,
                               iters_per_traj_set=iters_per_traj_set)
    # make sure both planners have same set of simulation parameters to compare
    single_planner.sim_params_set = plan_params_sets[0]
    avg_planner.sim_params_set = plan_params_sets[0]

    # each sim takes ~0.02s, for the avg/mode, simulating one action takes
    # t = num_rand_samples * 0.02 if there is a collision, else just 0.02
    # num actions to try out = ~max_time_s / t
    max_time_s = 200  # for single planner, 10 * num_rand_samples time for avg

    # Create many different object types
    # scale_x = np.linspace(start=0.8, stop=1.2, num=1, endpoint=True)
    scale_x = [1.0]
    scale_y = np.linspace(start=0.8, stop=1.2, num=3, endpoint=True)
    scale_z = np.linspace(start=0.8, stop=1.2, num=3, endpoint=True)
    all_cup_shapes = []  # x, y, z
    for sx in scale_x:
        for sy in scale_y:
            for sz in scale_z:
                mesh_scale = [sx, sy, sz]
                col_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName=os.path.join(
                                                    bottle.folder, "cup.obj"),
                                                meshScale=mesh_scale)

                all_cup_shapes.append((col_id, mesh_scale))

    planners = [single_planner, avg_planner]
    names = ["single", "avg", "mode"]
    new_shape = all_cup_shapes[0]
    col_id, mesh_scale = new_shape

    # iterate through different object types and (start, goal) permuatations
    for (start, goal) in start_goals:
        print("Start(%s) and goal(%s)" % (str(start), str(goal)), flush=True)
        print("Mesh scale (x, y, z): (%.2f, %.2f, %.2f)" % (
            mesh_scale[0], mesh_scale[1], mesh_scale[2]), flush=True)
        # change bottle shape and center of mass based on new height
        env.bottle.create_sim_bottle(new_shape=new_shape)

        if VISUALIZE:
            # visualize a vertical blue line representing goal pos of bottle
            # just to make line vertical
            vertical_offset = np.array([0, 0, 0.5])
            bottle_goal_pos = single_planner.bottle_pos_from_state(goal)
            Environment.draw_line(lineFrom=bottle_goal_pos,
                                  lineTo=bottle_goal_pos + vertical_offset,
                                  lineColorRGB=[0, 0, 1], lineWidth=1,
                                  lifeTime=0)

        # reset arm state
        env.arm.resetEE()

        for pi, planner in enumerate(planners):
            name = names[pi]
            if name == "single":
                timeout = max_time_s
            else:
                timeout = num_rand_samples * max_time_s

            if SAVE_STDOUT_TO_FILE:
                sys.stdout = open('%s_planner_output.txt' % name, 'w+')

            print("%s planner with allotted time %.2f!" %
                  (name, timeout), flush=True)

            _, policy = planner.plan(start=start, goal=goal)

            print("%s Found plan!" % name)
            # # set hard time limit for replan and execution
            # try:
            #     with time_limit(timeout):
            #         _, policy = planner.plan(start=start, goal=goal)
            #         # replan_exec_loop(
            #         #     env=env, planner=planner, bottle_pos=init_bottle_pos, bottle_ori=init_bottle_ori, start=start, goal=goal,
            #         #     exec_params=exec_params,
            #         #     metrics=metrics)
            # except Exception as e:
            # print("Failure to plan due to: %s" % e, flush=True)

    """For now, go one-step at a time, take a set of  10 (start, goal) pairs,
verify that single planner works on them for different objects. Then verify
visually that average plannner also works on them. For avg, don't average the
next state, only average the fall probability. Next  state should be identical
to the single planner's generated next state just to be consistent, so that
means avg planner shouuld uuse single planner's sim param to generate this. Also
make it so that there are no infeasible edges for average  planner except 100%
fall probability so we  don't end up with no plan, but may end up with a very
unsafe, high cost plan. 
    """


if __name__ == "__main__":
    test_env_params()
