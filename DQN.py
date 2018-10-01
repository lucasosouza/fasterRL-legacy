# coding: utf-8

# imports
import gym
import gym.spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import time

import itertools
from pprint import pprint

from collections import defaultdict
import pickle
from termcolor import colored

# start logging everything
from tensorboardX import SummaryWriter
from datetime import datetime
import os
from functools import reduce

# modularization
from wrappers import make_env
from buffers import Experience, ExperienceBuffer

class DQN(nn.Module):
    
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()


        self.network = nn.Sequential(
            nn.Linear(input_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions)
        )
        
    
    def forward(self, x):
        """ Main forward function """      
        
        x = x.type(torch.FloatTensor)
        return self.network(x)

        
class Agent:
    
    def __init__(self, alias, env, exp_buffer, net, tgt_net, gamma, epsilon, tau, trial, log_dir):
        
        # agent's name
        self.alias = alias
        # creates an environment
        self.env = env
        # creates an experience buffer
        self.exp_buffer = exp_buffer
        # reset
        self.reset()
        # learning net
        self.net = net
        # target net
        self.tgt_net = tgt_net
        # epsilon
        self.epsilon = epsilon
        # defines ratio of update of online network and target network
        self.tau = tau
        # define discount rate
        self.gamma = gamma

        # remaining variables to be initialized later
        self.optimizer = None
        self.total_rewards = []
        self.total_steps = []
        self.print_color = 'white'
        self.completed = False
        self.transfer_batch_size = 1

        ## book keeping variables
        # frames and speed
        self.frame_idx = 0
        self.ts_frame = 0
        self.speed = 0

        # intermediate variables
        self.latest_qvals = None        

        # rewards
        self.done_reward = 0
        self.mean_reward = 0
        self.std_reward = 0
        self.best_mean_reward = -np.inf 
        self.test_rewards = []

        # initialize log writer
        run_dir = "".join([alias, "-trial", str(trial)])
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_dir))

    def reset(self):
        """ Reset is an internal method. Only calls itself on initialization and when an episode is finished
        """

        self.state = self.env.reset() # is it a bug I haven't seen yet
        self.steps = 0
        self.total_reward = 0.0
    
    def request_share(self, threshold):
        """ Returns a mask with all states that it wants experience from """

        return self.exp_buffer.grid_occupancy == threshold

    def play_step(self, device="cpu", test=False):
        """ Play a single step """
        
        done_reward = None
        self.steps += 1

        ## action selection
        # play step with e-greedy exploration strategy
        # if not in test fase
        if np.random.random() < self.epsilon and not test:
            # takes a random action
            action = self.env.action_space.sample()
        else:
            # moves state into an array with 1 sample to pass through neural net
            state_a = np.array([self.state], copy=False)
            # creates tensor
            state_v = torch.tensor(state_a).to(device)
            # get q values with feed forward
            q_vals_v = self.net(state_v)
            self.latest_qvals = q_vals_v.detach().numpy()[0] # store for bookkeeping
            # chooses greedy action and get its value
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        # take action
        new_state, reward, is_done, _ = self.env.step(action) # step of the environment is done here
        self.total_reward +=  reward

        # only add to experience buffer if not in test
        if not test:        
            exp = Experience(self.state, action, reward, is_done, new_state)
            self.exp_buffer.append(exp)
        
        # change state to new state
        self.state = new_state
        
        # if complete, accrue total reward and reset
        if is_done:
            done_reward = self.total_reward
            self.done_reward = done_reward # book keeping
            # add totals
            self.total_rewards.append(done_reward)
            self.total_steps.append(self.steps)            
            # reset environment
            self.reset()
            
        return is_done, done_reward

    def record(self, loss=None, track_weights=True):

        self.writer.add_scalar("epsilon", self.epsilon, self.frame_idx)
        # monitor training speed
        self.writer.add_scalar("speed", self.speed, self.frame_idx)
        # monitor average reward on last 100 episodes
        self.writer.add_scalar("reward_100_avg", self.mean_reward, self.frame_idx)
        self.writer.add_scalar("reward_100_std", self.std_reward, self.frame_idx)
        # monitor reward
        self.writer.add_scalar("reward", self.done_reward, self.frame_idx)
        # monitor num_steps
        self.writer.add_scalar("steps", self.steps, self.frame_idx)
        # track loss, if available
        if loss:
            self.writer.add_scalar("loss", loss, self.frame_idx)
        if self.latest_qvals is not None:
            self.writer.add_scalar("min_q_value", max(self.latest_qvals), self.frame_idx)
            self.writer.add_scalar("max_q_value", min(self.latest_qvals), self.frame_idx)

        # track weights
        # if track_weights:
        #     self.writer.add_histogram("net_weights", self.gen_params_debug(self.net))
        #     self.writer.add_histogram("tgt_net_weights", self.gen_params_debug(self.tgt_net))

    @staticmethod
    def gen_params_debug(network):
        weights = []
        for tensor in network.parameters():
            weights.extend(tensor.detach().numpy().ravel())

        return np.array(weights)

    def soft_update_target_network(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """        
        
        # iterate through both together and make a copy one by one
        for target_param, local_param in zip(self.tgt_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)


    def calc_expected_state_action_values(self, batch, device='cpu'):

        # unpack vectors of variables
        states, actions, rewards, dones, next_states = batch
        
        # creates tensors. and push them to device, if GPU is available, then uses GPU
        next_states_v = torch.tensor(next_states).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        # dones is dealt differently. not sure why yet, maybe to specify is just a 1-0 tensor
        done_mask = torch.ByteTensor(dones).to(device)
                
        # apply target network to next_states. get maximum q-value. no need to know the action, just the value
        ## main difference here: use latest values instead of tgt_net when transferring
        next_state_values = self.net(next_states_v).max(1)[0]
        # if is done, value of next state is set to 0. important correction
        next_state_values[done_mask] = 0.0
        # detach values from computation graph, to prevent gradients from flowing into neural net
        # used to calculate Q approximation for next states
        # if we don't this, backpropagation of loss will affect predictions for both current state and next state.
        # and we want only to affect predictions for current state
        next_state_values = next_state_values.detach()
        
        # calculate total value (Bellman approximation value)
        expected_state_action_values = next_state_values * self.gamma + rewards_v
        
        return expected_state_action_values


    def calc_loss(self, batch, device="cpu", expected_state_action_values=None, double=True):
        """ Function optimized to exploit GPU parallelism by processing all batch samples with vector operations """


        # unpack vectors of variables
        states, actions, rewards, dones, next_states = batch
        
        # creates tensors. and push them to device, if GPU is available, then uses GPU
        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        actions_v = torch.tensor(actions).to(device)
        # dones is dealt differently. not sure why yet, maybe to specify is just a 1-0 tensor
        done_mask = torch.ByteTensor(dones).to(device)

        # calculate state-action values
        # gather: select only the values for the actions taken
        # result of gather applied to tensor is differentiable operation, keep all gradients w.r.t to final loss value
        state_action_values = self.net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        
        # if not given expected action values, calculate
        # regular procedure is to calculate, do not calculate only in transfer cases
        if expected_state_action_values is None:
                    
            # apply target network to next_states. get maximum q-value. no need to know the action, just the value
            # if double, get actions from regular network and value from target network
            # avoids overfitting
            if double:
                # get q values with feed forward
                next_q_vals_v = self.net(next_states_v)
                # chooses greedy action from target net
                _, next_state_action_v = torch.max(next_q_vals_v, dim=1)
                # gets actions from 
                next_state_values = \
                    self.tgt_net(next_states_v).gather(1, next_state_action_v.unsqueeze(-1)).squeeze(-1)
            else:
                next_state_values = self.tgt_net(next_states_v).max(1)[0]
            # if is done, value of next state is set to 0. important correction
            next_state_values[done_mask] = 0.0
            # detach values from computation graph, to prevent gradients from flowing into neural net
            # used to calculate Q approximation for next states
            # if we don't this, backpropagation of loss will affect predictions for both current state and next state.
            # and we want only to affect predictions for current state
            next_state_values = next_state_values.detach()
            
            # calculate total value (Bellman approximation value)
            expected_state_action_values = next_state_values * self.gamma + rewards_v
            
        # calculate mean squared error loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        
        return loss


def DQN_experiment(params, log_dir):

    # define device on which to run
    device = torch.device(params["DEVICE"])

    ## initialize global variables
    # initialize local log trackers 
    log_episodes_count = []
    log_ma_steps = []
    log_md_steps = []
    log_ma_rewards = []
    log_md_rewards = []

    colors=['green','red','blue','yellow','cyan','magenta','grey','white']

    # try several times and average results, needs to compensate for stochasticity
    for trial in range(params["NUM_TRIALS"]):

        # initialize environment
        agents = []

        # need to be one env per agent
        env = make_env(params["DEFAULT_ENV_NAME"])

        # initialize agents
        for idx in range(params["NUM_AGENTS"]):

            # initialize agent
            buffer = ExperienceBuffer(params["REPLAY_SIZE"], env)
            net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
            tgt_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
            epsilon = params["EPSILON_START"]
            gamma = params["GAMMA"]
            tau = params["SOFT_UPDATE_TAU"]
            agent = Agent('agent' + str(idx+1), env, buffer, net, tgt_net, gamma, epsilon, tau, trial, log_dir)

            # other variables
            agent.optimizer = optim.Adam(agent.net.parameters(), lr=params["LEARNING_RATE"])
            agent.print_color = colors[idx]

            agents.append(agent)    


        ######### training loop
        ################################

        ts = time.time() # track start time


        ######### 1. Filling replay bugg
        ################################

        # both agents fill their buffer prior to experience
        for agent in agents:
            while True:
            
                # add frame count
                agent.frame_idx+= 1

                # play step
                episode_over, done_reward = agent.play_step(device=device)
                if params["DEBUG"]: agent.record()

                # check if minimum buffer size has been achieved. if not, move on, do not do learning
                if len(agent.exp_buffer) >= params["REPLAY_START_SIZE"]:
                    agent.reset()
                    break    


        ######### 1. They start alternating
        ################################

        episode_start = time.time()        
        ep_count = 0
        # while all agents have not completed:    
        while sum(map(lambda agent:agent.completed, agents)) != len(agents):

            ep_count += 1

            # agents alternate
            for agent in agents:

                ## Before 2 agents perform, act, do one round of experience share
                # given a sharing interval and it is not the first episode
                if params["SHARING"] and ep_count % params["SHARING_INTERVAL"] == 0 and ep_count > 0:

                    # agent 1 requests
                    student, teacher = agents[0], agents[1]
                    transfer_mask = student.request_share(threshold=0)
                    transfer_batch = teacher.exp_buffer.sample_with_mask(student.steps[-1], transfer_mask)
                    student.exp_buffer.extend(transfer_batch)

                    # agent 2 requests
                    student, teacher = agents[1], agents[0]
                    transfer_mask = student.request_share(threshold=0)
                    transfer_batch = teacher.exp_buffer.sample_with_mask(student.steps[1], transfer_mask)
                    student.exp_buffer.extend(transfer_batch)


                # check if agent has not completed the task already
                # if it does, go to the next agent
                if not agent.completed:

                    # play until episode is over
                    episode_over = False
                    while not episode_over:

                        # add frame count
                        agent.frame_idx+= 1

                        # play step
                        episode_over, done_reward = agent.play_step(device=device)

                        if done_reward is not None:

                            # calculate speed
                            agent.speed = (agent.frame_idx - agent.ts_frame) / (time.time() - ts)
                            agent.ts_frame = agent.frame_idx
                            ts = time.time()

                            # get time between episodes

                            ## verify completion and report metrics
                            if params["INDEPENDENT_EVALUATION"]:

                                if len(agent.total_rewards) % params["TRACKING_INTERVAL"] == 0:
                                    agent.test_rewards = []
                                    evaluation_start = time.time()
                                    for _ in range(100):
                                        done_reward = False
                                        while not done_reward:
                                            _, done_reward = agent.play_step(device=device, test=True)
                                        agent.test_rewards.append(done_reward)
                                    evaluation_time = time.time() - evaluation_start

                                    # only report after one episode ends
                                    agent.mean_reward = np.mean(agent.test_rewards)
                                    agent.std_reward = np.std(agent.test_rewards)

                                    # calculate elapsed time
                                    episode_end = time.time()
                                    episode_speed = params["TRACKING_INTERVAL"] / (episode_end - episode_start) 
                                    episode_start = time.time()

                                    # report
                                    print(colored("%s, %d: done %d episodes, mean reward %.2f, std reward %.2f, eps %.2f, speed %d f/s, ep_speed %.2f e/s, eval_time %.2f s" % (
                                        agent.alias, agent.frame_idx, len(agent.total_rewards), agent.mean_reward, agent.std_reward, agent.epsilon, agent.speed, episode_speed, evaluation_time
                                    ), agent.print_color))
                                    
                                    ## check if reward has improved from last iteration
                                    if agent.mean_reward is not None:
                                        if agent.mean_reward > params["MEAN_REWARD_BOUND"]:
                                            print(colored("%s solved in %d episodes!" % (agent.alias, len(agent.total_rewards)), agent.print_color))
                                            # save final version
                                            # save final version
                                            # torch.save(agent.net.state_dict(), "weights/" + params["DEFAULT_ENV_NAME"] + "-" + agent.alias + "-best.dat")
                                            # mark as completed
                                            agent.completed = True
                                            # save local log
                                            log_episodes_count[agent.alias].append(len(agent.total_rewards))
                                            log_steps[agent.alias].append(len(agent.total_rewards))

                            ## approach to track evaluation using moving averages:
                            else:
                                # only report after one episode ends
                                agent.mean_reward = np.mean(agent.total_rewards[-params["NUMBER_EPISODES_MEAN"]:])
                                agent.std_reward = np.std(agent.total_rewards[-params["NUMBER_EPISODES_MEAN"]:])

                                # calculate elapsed time
                                episode_end = time.time()
                                episode_speed = 1 / (episode_end - episode_start)
                                episode_start = time.time()

                                # report
                                if len(agent.total_rewards) % params["TRACKING_INTERVAL"] == 0:
                                    print(colored("%s, %d: done %d episodes, mean reward %.2f, std reward %.2f, eps %.2f, speed %d f/s, ep_speed %.2f e/s" % (
                                        agent.alias, agent.frame_idx, len(agent.total_rewards), agent.mean_reward, agent.std_reward, agent.epsilon, agent.speed, episode_speed
                                    ), agent.print_color))
                                
                                ## check if reward has improved from last iteration
                                if agent.mean_reward is not None:
                                    if agent.mean_reward > params["MEAN_REWARD_BOUND"]:
                                        print(colored("%s solved in %d episodes!" % (agent.alias, len(agent.total_rewards)), agent.print_color))
                                        # save final version
                                        # torch.save(agent.net.state_dict(), "weights/" + params["DEFAULT_ENV_NAME"] + "-" + agent.alias + "-best.dat")
                                        # mark as completed
                                        agent.completed = True
                                        # save local log
                                        log_episodes_count.append(len(agent.total_rewards))
                                        log_ma_rewards.append(np.mean(agent.total_rewards[-params["REPORTING_INTERVAL"]:]))
                                        log_md_rewards.append(np.std(agent.total_rewards[-params["REPORTING_INTERVAL"]:]))
                                        log_ma_steps.append(np.mean(agent.total_steps[-params["REPORTING_INTERVAL"]:]))
                                        log_md_steps.append(np.std(agent.total_steps[-params["REPORTING_INTERVAL"]:]))

                        # if no sign of converging, also break
                        # but don't store the result
                        if len(agent.total_rewards) > params["MAX_GAMES_PLAYED"]:
                            agent.completed = True

                        # decay epsilon after the first episodes that fill the buffer
                        # decay epsilon linearly on frames
                        agent.epsilon = max(params["EPSILON_FINAL"], params["EPSILON_START"] - (agent.frame_idx-params["REPLAY_START_SIZE"]) / params["EPSILON_DECAY_LAST_FRAME"])
                            
                        # update at every frame using soft updates
                        if params["SOFT"]:
                            agent.soft_update_target_network()
                        else:                        
                            if agent.frame_idx % params["SYNC_TARGET_FRAMES"] == 0:
                                agent.tgt_net.load_state_dict(agent.net.state_dict())
                            
                        ## learn
                        # zero gradients
                        agent.optimizer.zero_grad()
                        # sample from buffer
                        batch = agent.exp_buffer.sample(params["BATCH_SIZE"])
                        # calculate loss
                        # decide to leave it on the agent as a static method, instead of floating around
                        loss_t = agent.calc_loss(batch, device=device)
                        # calculate gradients
                        loss_t.backward()
                        # gradient clipping
                        if params["GRADIENT_CLIPPING"]: nn.utils.clip_grad_norm_(net.parameters(), params["GRAD_L2_CLIP"])
                        # optimize
                        agent.optimizer.step()

                        # track agent parameters, including loss function
                        # detach loss before extracting value - not sure if needed, but better safe than sorry
                        if params["DEBUG"]: agent.record(loss_t.detach().item())


    for agent in agents:
        agent.writer.close()

    # return local log with results
    local_log = {
        "episodes_count": log_episodes_count,
        "ma_steps": log_ma_steps,
        "md_steps": log_md_steps,
        "ma_rewards": log_ma_rewards,
        "md_rewards": log_md_rewards
    }
    return local_log







