#########################################################
# DESCRIPTION: Shows that the mpc controller can adapt to diffrent goals by changing the cost function
#########################################################
import logging
import math
import time
import sys

import gym
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
from mpc_local import mpc

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 10  # T
    N_BATCH = 1
    LQR_ITER = 5
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    CASE = 0


    class PendulumDynamics(torch.nn.Module):
        def forward(self, state, action):
            th = state[:, 0].view(-1, 1)
            thdot = state[:, 1].view(-1, 1)

            g = 10
            m = 1
            l = 1
            dt = 0.05

            u = action
            u = torch.clamp(u, -2, 2)

            newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = torch.clamp(newthdot, -8, 8)

            state = torch.cat((angle_normalize(newth), newthdot), dim=1)
            return state


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    downward_start = True
    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    env = wrappers.Monitor(env, '/tmp/box_ddp_pendulum/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    nx = 2
    nu = 1

    u_init = None
    render = True
    retrain_after_iter = 50
    run_iter = 5000

    # swingup goal (observe theta and theta_dt)
    goal_weights = torch.tensor((1., 0.1))  # nx
    # goal_state = torch.tensor((-7*np.pi/8, 0.))  # nx
    goal_state = torch.tensor((-0*np.pi/8, 0.))  # nx
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)
    total_reward = 0

    for i in range(run_iter):
        state = env.state.copy()
        state = torch.tensor(state).view(1, -1)
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=1, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)
        # print("mpc initialized")

        # compute action based on current state, dynamics, and cost
        # print("calling ctrl...")
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, PendulumDynamics())
        # logger.debug("nominal_states: %s", nominal_states)
        # logger.debug("nominal_actions: %s", nominal_actions)
        # logger.debug("nominal_objs: %s", nominal_objs)
        # print("ctrl done")

        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)
        
        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.detach().numpy())
        total_reward += r
        # logger.debug("states: %s",s)
        logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()
        
        ######################### STATE MACHINE FOR THE COST FUNCTION #########################
        theta = nominal_states[0][0][0]
        logger.debug("theta: %.4f, goal_state[0]: %.4f", angle_normalize(theta), angle_normalize(goal_state[0]))
        if CASE == 0:
            if abs(angle_normalize(theta) - angle_normalize(goal_state[0])) < 1e-2:
                print(" --------------------- Goal 1 reached ---------------------")
                time.sleep(1)  # Wait for 1 second
                goal_state = torch.tensor((1*np.pi/8, 0.))
                # px = -torch.sqrt(goal_weights) * goal_state
                px = -goal_weights * goal_state
                p = torch.cat((px, torch.zeros(nu)))
                p = p.repeat(TIMESTEPS, N_BATCH, 1)
                cost = mpc.QuadCost(Q, p)
                CASE = 1
        elif CASE == 1:
            if abs(angle_normalize(theta) - angle_normalize(goal_state[0])) < 1e-2:
                print(" --------------------- Goal 2 reached ---------------------")
                time.sleep(1)
                goal_state = torch.tensor((-1*np.pi/8, 0.))
                px = -torch.sqrt(goal_weights) * goal_state
                p = torch.cat((px, torch.zeros(nu)))
                p = p.repeat(TIMESTEPS, N_BATCH, 1)
                cost = mpc.QuadCost(Q, p)
                CASE = 2
        ######################### STATE MACHINE FOR THE COST FUNCTION #########################

    logger.info("Total reward %f", total_reward)
