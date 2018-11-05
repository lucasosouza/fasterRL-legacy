# coding: utf-8

# imports
import gym
import gym.spaces
import marlo
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

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
    
    def __init__(self, alias, env, net, paramas):
        

        self.params = params
        # agent's name
        self.alias = alias
        # assigns an environment
        self.env = env
        # learning net
        self.net = net
        agent.optimizer = optim.Adam(self.net.parameters(), lr=params["LEARNING_RATE"])

        # intermediate variables
        self.latest_qvals = None        

    def play_step(self, device="cpu", test=False):
        """ Play a single step """
        
        ## select action
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

        # no appending to buffer now

        # change state to new state
        self.state = new_state

        return is_done, done_reward
