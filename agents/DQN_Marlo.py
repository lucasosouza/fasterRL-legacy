# coding: utf-8

# imports
import gym
import gym.spaces
# import gym_minecraft
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
from wrappers import make_env, wrap_env_marlo
from buffers import Experience, ExperienceBuffer, ExperienceBufferGridImage
import json
import gc

# new marlo additions
import marlo

COLORS=['green','blue','red','yellow','cyan','magenta','grey','white']
RANDOM_SEED = 42

class DQN(nn.Module):
    
    def __init__(self, input_shape, n_actions, device):

        # set random seed for pytorch
        torch.manual_seed(RANDOM_SEED)

        super(DQN, self).__init__()
        self.device = device

        # defines convolutional layers as defined in DQN paper
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()        
        )
        
        # get size of last layer to pass it to the first fc layer
        # since pytorch has no flatten layer
        conv_out_size = self._get_conv_out(input_shape)
        
        # defines fully connected layers as defined in DQN paper
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        

    def _get_conv_out(self, shape):
        """ 
            Get shape of output of conv layers, to help defining the input shape of next fc layer
            underscore for private methods, why not use @static instead?
        """
        # pass a batch with 1 obs and shape equal input shape through conv layers
        o = self.conv(torch.zeros(1, *shape))
        # get shape of output
        output_shape = int(np.prod(o.size()))

        return output_shape
    
    def forward(self, x):
        """ Main forward function """      
        
        # manually changed from torch.FloatTensor to torch.cuda.FloatTensor to run in GPU
        # need to be dynamic or not do the conversion at all
        if self.device == "cuda":
            x = x.type(torch.cuda.FloatTensor)
        elif self.device == "cpu":
            x = x.type(torch.FloatTensor)
        
        # apply the convolution layer to input and ovtain a 4d tensor on output
        # and result is flattened, by the view function
        # view doesn't create a new memory obect or move data in memort, 
        # just change higher-level shape of tensor
        conv_out = self.conv(x).view(x.size()[0], -1)       

        # pass flattened 2d tensor to fc layer
        return self.fc(conv_out)        

        
class Agent:
    
    def __init__(self, alias, env, exp_buffer, net, tgt_net, gamma, epsilon, tau, 
                trial, log_dir, params):
        

        # set numpy random seed for action selection
        np.random.seed(RANDOM_SEED)

        self.params = params

        # agent's name
        self.alias = alias
        # assigns an environment
        self.env = env
        self.obs_shape = self.env.observation_space.shape
        # assigns an experience buffer
        self.exp_buffer = exp_buffer
        # reset
        self.episode = 0 # it will add 1 after reset and be set to 0
        self.reset(count_episode=False)
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
        self.frame_speed = 0
        self.ep_speed = 0

        # intermediate variables
        self.latest_qvals = None        

        # rewards
        self.step_reward = None
        self.done_reward = 0
        self.mean_reward = 0
        self.std_reward = 0
        self.best_mean_reward = -np.inf 
        self.test_rewards = []

        # initialize log writer
        run_dir = "".join([alias, "-trial", str(trial)])
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_dir))

    def reset(self, count_episode=True):
        """ Reset is an internal method. Only calls itself on initialization and when an episode is finished
        """

        self.env.close()
        self.env = init_environment(self.params["DEFAULT_ENV_NAME"])
        self.state = self.env.reset() # is it a bug I haven't seen yet
        self.steps = 0
        self.total_reward = 0.0
        if count_episode:
            self.episode += 1
    
    def request_share(self, threshold):
        """ Returns a mask with all states that it wants experience from """

        return self.exp_buffer.identify_unexplored(threshold)

    def fill_buffer(self):

        # fill buffer prior to experience
        while len(self.exp_buffer) < self.params["REPLAY_START_SIZE"]:
                    
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            reward *= self.params["REWARD_SCALING_FACTOR"] # scaling reward according to a predefined factor
            exp = Experience(self.state, action, reward, is_done, new_state)
            self.exp_buffer.append(exp)

            # change state to new state
            self.state = new_state

            # if done, needs to reset
            if is_done: self.reset(count_episode=False)
     
        # reset to leave agent to start learning
        self.reset(count_episode=False)


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
            # manually adding .cpu() to run in GPU mode
            self.latest_qvals = q_vals_v.detach().cpu().numpy()[0] # store for bookkeeping
            # chooses greedy action and get its value
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        # take action
        new_state, reward, is_done, _ = self.env.step(action) # step of the environment is done here
        reward *= self.params["REWARD_SCALING_FACTOR"] # scaling reward according to a predefined factor
        # if self.params["RENDER"]:
        #     self.env.render('human') # specific for minecraft
        self.total_reward +=  reward
        self.step_reward = reward # for bookkeeping purposes

        # only add to experience buffer if not in test
        # also do not add if state doesn't match expected size
        if not test:        
            # this is a temporary check to id if this is the problem
            # cannot id the issue at the moment
            # it does seem to be the problem
            if self.state.shape != self.obs_shape:
                print("State shape size is inconsistent")
            elif new_state.shape != self.obs_shape:
                print("New state shape size is inconsistent")
            else:
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
            # track episode
            self.record_episode()
            # reset environment
            self.reset()

        return is_done, done_reward


    @staticmethod
    def gen_params_debug(network):
        weights = []
        for tensor in network.parameters():
            weights.extend(tensor.detach().cpu().numpy().ravel())

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


    def hard_update_target_network(self):
        """ Update every X steps """

        self.tgt_net.load_state_dict(self.net.state_dict())
 

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

    def learn(self, device="cpu"):
        # zero gradients
        self.optimizer.zero_grad()
        # sample from buffer
        batch = self.exp_buffer.sample(self.params["BATCH_SIZE"])
        # calculate loss
        loss_t = self.calc_loss(batch, device=device)
        # calculate gradients
        loss_t.backward()
        # gradient clipping
        if self.params["GRADIENT_CLIPPING"]: 
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params["GRAD_L2_CLIP"])
        # optimize
        self.optimizer.step()

        return loss_t

    def record_frame(self, loss=None):

        # print("recording frame: ", str(self.frame_idx))

        if loss:
            self.writer.add_scalar("loss", loss, self.frame_idx)

        # monitor training speed - not that essential - in frames per second
        self.writer.add_scalar("frame_speed", self.frame_speed, self.frame_idx)

        # episode wide is fine as well
        self.writer.add_scalar("epsilon", self.epsilon, self.frame_idx)

        # monitor reward
        self.writer.add_scalar("step_reward", self.step_reward, self.frame_idx)

        if self.latest_qvals is not None:
            self.writer.add_scalar("q_value/min", min(self.latest_qvals), self.frame_idx)
            self.writer.add_scalar("q_value/max", max(self.latest_qvals), self.frame_idx)


    def record_episode(self, track_weights=True):

        # monitor average reward on last 100 episodes
        self.writer.add_scalar("reward_100/avg", self.mean_reward, self.episode)
        self.writer.add_scalar("reward_100/std", self.std_reward, self.episode)
        # monitor reward
        self.writer.add_scalar("reward", self.done_reward, self.episode)
        # monitor num_steps
        self.writer.add_scalar("steps", self.steps, self.episode)
        # monitor episode speed
        self.writer.add_scalar("ep_speed", self.ep_speed, self.episode)
        # monitor parameter sharing
        if self.params["SHARING"]:
            self.writer.add_scalar("experiences_received", self.exp_buffer.experiences_received, self.episode)

        # too much overhead, remove it for now
        # track weights
        # if track_weights:
        #     self.writer.add_histogram("net_weights", self.gen_params_debug(self.net))
        #     self.writer.add_histogram("tgt_net_weights", self.gen_params_debug(self.tgt_net))

################################## ENVIRONMENT METHODS

def DQN_experiment(params, log_dir, local_log_path, random_seed=None):

    params["MEAN_REWARD_BOUND"] *= params["REWARD_SCALING_FACTOR"]

    # fix replay start sie to be equal to batch size
    params["REPLAY_START_SIZE"] = params["BATCH_SIZE"]

    # initialize local log trackers 
    local_log = {}    

    # try several times and average results, needs to compensate for stochasticity
    for trial in range(params["NUM_TRIALS"]):
        train(params=params, log_dir=log_dir, local_log=local_log,
              random_seed=RANDOM_SEED, trial=trial)
        gc.collect()

    # output local log json
    with open( local_log_path , "w") as f:
        json.dump(local_log, f)

    # Inform experiment is done
    print("Experiment complet. Results found at: " + local_log_path)
    

def init_environment(env_name):

    client_pool = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
    join_tokens = marlo.make(env_name,
                             params={
                                "client_pool": client_pool
                             })

    # As this is a single agent scenario,there will just be a single token
    # assert len(join_tokens) == 1
    join_token = join_tokens[0]

    # initialize environment    
    env = marlo.init(join_token)

    # env.configure(videoResolution=[84,84])
    # I need to figure out where in marlo I can fix the videoresolutions. probably in the params of marlo.make

    # env.configure(stack_frames=4) 
    # there is the frame stacking logic which is in the environment
    # I have to either transpose it here
    # or let go of the DQN 
    # I can just transpose
    # issue is I can't observe 4 times
    # it is turn based so every observation I do is one step I take 

    # create env and add specific conifigurations to Malmo

    # I can probably configure all that in the wrapper
    # I have to understand what in the wrapper I will still use in the new environment
    # but the logic of getting 4 states in a row will need to go to the wrapper
    # only thing wrapper is currently doing is adding the float
    # the black and white transformation is also not happening
    # hence I will still get and extra dimension 3 channels, instead of 4, 
    # and since 4 is not fixed anywhere it should work fine 
    env = wrap_env_marlo(env)

    if RANDOM_SEED:
        env.seed(RANDOM_SEED)

    print("Observation Space: ", env.observation_space)
    print("Action Space: ", env.action_space)

    return env


def train(params, log_dir, local_log, random_seed, trial):

    # define device on which to run
    device = torch.device(params["DEVICE"])

    ## Marlo specifics
    # get join tokens
    env = init_environment(params["DEFAULT_ENV_NAME"])

    agents = []
    for aid in range(params["NUM_AGENTS"]):

        # initialize bufer
        if params["SHARING"] and params["PRIORITIZED_SHARING"]:
            bufer = ExperienceBufferGridImage(params["REPLAY_SIZE"])
        else:
            bufer = ExperienceBuffer(params["REPLAY_SIZE"])            

        # initialize agent        
        net = DQN(env.observation_space.shape, env.action_space.n, params["DEVICE"]).to(device)
        tgt_net = DQN(env.observation_space.shape, env.action_space.n, params["DEVICE"]).to(device)
        epsilon = params["EPSILON_START"]
        gamma = params["GAMMA"]
        tau = params["SOFT_UPDATE_TAU"]
        agent = Agent('agent' + str(aid), env, bufer, net, tgt_net, gamma, epsilon, tau, 
            trial, log_dir, params)

        # other variables
        agent.optimizer = optim.Adam(agent.net.parameters(), lr=params["LEARNING_RATE"])
        agent.print_color = COLORS[aid]

        local_log[agent.alias+"-"+str(trial)] = {"rewards": [],"steps": []}

        # fill buffer with initial size - don't count these episodes
        agent.fill_buffer()

        agents.append(agent)

    # training loop
    ep_count = 0
    while sum(map(lambda agent:agent.completed, agents)) != len(agents):

        # overall count of episodes
        ep_count += 1

        # sharing
        if params["SHARING"] and ep_count % params["SHARING_INTERVAL"] == 0 and ep_count > 0:
            if params["PRIORITIZED_SHARING"]:
                share(agents, params["BATCH_SIZE_TRANSFER"], params["REPLAY_START_SIZE"], params["SHARING_THRESHOLD"])
            else:
                share_no_mask(agents, params["BATCH_SIZE_TRANSFER"], params["REPLAY_START_SIZE"])

        # each agent does one episode
        for agent in agents:

            ## Before 2 agents perform, act, do one round of experience share
            # given a sharing interval and it is not the first episode
            if not agent.completed:

                episode_over = False
                episode_start = time.time()        
                while not episode_over:

                    # play step
                    frame_start = time.time()
                    episode_over, done_reward = agent.play_step(device=device)
                    agent.frame_idx+= 1

                    #### Folllowing methods on episode basis
                    if done_reward is not None:

                        # calculate episode speed
                        agent.ep_speed = 1 / (time.time() - episode_start)
                        # reset trackers
                        episode_start = time.time()

                        # save to local log as well
                        local_log[agent.alias+"-"+str(trial)]["rewards"].append(agent.total_rewards[-1])
                        local_log[agent.alias+"-"+str(trial)]["steps"].append(agent.total_steps[-1])

                        if params["INDEPENDENT_EVALUATION"]:
                            offline_evaluation(params, agent, log_dir)
                        else:
                            online_evaluation(params, agent, log_dir)

                        ## check if problem has been solved
                        # need a minimum number of episodes to evaluate
                        if len(agent.total_rewards) >= params["NUMBER_EPISODES_MEAN"]:                                
                            # and mean reward has to go above boundary
                            if agent.mean_reward >= params["MEAN_REWARD_BOUND"]:
                                    print(colored("%s solved in %d episodes!" % (agent.alias, len(agent.total_rewards)), agent.print_color))
                                    agent.completed = True

                        # if no sign of converging, also break
                        if len(agent.total_rewards) >= params["MAX_GAMES_PLAYED"]:
                            agent.completed = True

                    #### Folllowing methods on frame basis
                    # decay epsilon linearly on frames
                    agent.epsilon = max(params["EPSILON_FINAL"], params["EPSILON_START"] - \
                        agent.frame_idx / params["EPSILON_DECAY_LAST_FRAME"])
                    
                    # update at every frame using soft updates
                    if params["SOFT"]:
                        agent.soft_update_target_network()
                    # or hard updates
                    else:
                        if agent.frame_idx % params["SYNC_TARGET_FRAMES"] == 0:
                            agent.hard_update_target_network()
                    
                    ## learn
                    loss_t = agent.learn(device)

                    # record
                    agent.frame_speed = 1 / (time.time() - frame_start)
                    if params["DEBUG"]: 
                        agent.record_frame(loss_t.detach().item()) # detach required?


    # del bufer to force gc later, occupies too much memory
    del bufer
    for agent in agents:
        del agent.exp_buffer
    # closes tensorboard writer
    agent.writer.close()


def share(agents, batch_size_transfer, replay_start_size, threshold=0):

        # agent 1 requests
        student, teacher = agents[0], agents[1]
        if not student.completed:
            transfer_mask = student.request_share(threshold=threshold)
            transfer_batch = teacher.exp_buffer.sample_with_mask(batch_size_transfer, transfer_mask)
            student.exp_buffer.receive(transfer_batch)

        # agent 2 requests
        student, teacher = agents[1], agents[0]
        if not student.completed:
            transfer_mask = student.request_share(threshold=threshold)
            transfer_batch = teacher.exp_buffer.sample_with_mask(batch_size_transfer, transfer_mask)
            student.exp_buffer.receive(transfer_batch)

        # track experiences transferred
        print("Number of experiences transferred: ")
        print(agents[0].exp_buffer.experiences_received, agents[1].exp_buffer.experiences_received)


def share_no_mask(agents, batch_size_transfer, replay_start_size):

        # agent 1 requests
        student, teacher = agents[0], agents[1]
        if not student.completed:
            transfer_batch = teacher.exp_buffer.sample_no_mask(batch_size_transfer)
            student.exp_buffer.receive(transfer_batch)

        # agent 2 requests
        student, teacher = agents[1], agents[0]
        if not student.completed:
            transfer_batch = teacher.exp_buffer.sample_no_mask(batch_size_transfer)
            student.exp_buffer.receive(transfer_batch)

        # track experiences transferred
        print("Number of experiences transferred: ")
        print(agents[0].exp_buffer.experiences_received, agents[1].exp_buffer.experiences_received)


def offline_evaluation(params, agent, log_dir):
    """ approach to track evaluation using independent runs. more costly, but more accurate. """

    if len(agent.total_rewards) % params["PRINT_INTERVAL"] == 0:
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

        # report
        print(colored("%s, %d: done %d episodes, mean reward %.2f, std reward %.2f, eps %.2f, ep_speed %.2f e/s, eval_time %.2f s" % (
            agent.alias, agent.frame_idx, len(agent.total_rewards), agent.mean_reward, agent.std_reward, agent.epsilon, agent.ep_speed, evaluation_time
        ), agent.print_color))
      
        weights_dir=os.path.join(log_dir, "weights/")
        torch.save(agent.net.state_dict(), weights_dir + params["DEFAULT_ENV_NAME"] + "-" + agent.alias + ".dat")


def online_evaluation(params, agent, log_dir):
    """ approach to track evaluation using moving averages """

    # only report after one episode ends
    agent.mean_reward = np.mean(agent.total_rewards[-params["NUMBER_EPISODES_MEAN"]:])
    agent.std_reward = np.std(agent.total_rewards[-params["NUMBER_EPISODES_MEAN"]:])

    # report
    if len(agent.total_rewards) % params["PRINT_INTERVAL"] == 0:
        print(colored("%s, %d: done %d episodes, mean reward %.2f, std reward %.2f, eps %.2f, ep_speed %.2f e/s, steps %d"  % (
            agent.alias, agent.frame_idx, len(agent.total_rewards), agent.mean_reward, agent.std_reward, agent.epsilon, agent.ep_speed, agent.total_steps[-1]
        ), agent.print_color))
    print(agent.total_rewards[-params["NUMBER_EPISODES_MEAN"]:])

    weights_path= log_dir.replace("runs", "weights")
    torch.save(agent.net.state_dict(), weights_path + "-" + agent.alias + ".dat")
