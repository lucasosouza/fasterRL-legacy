# coding: utf-8

import numpy as np
from collections import namedtuple

Experience = namedtuple('Experience', 
    field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer:
    
    def __init__(self, capacity, env):
        """ Made changes to adapt to any environment. 
            Number of bins could be a hyperparameter to optimize for if the method works 
        """

        # initializes a deque
        self.buffer = []

        # variables to discretize gtid
        self.n_bins = 10 # need to use in a later method
        self.capacity = capacity

        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        # grid shape: discretize all state variables, then add action variable
        grid_shape = [self.n_bins for _ in range(n_states)]
        grid_shape.append(n_actions)

        # store experiences by grid position, to facilitate recovery
        # best way I could find to init an array with empty lists
        m = np.zeros(np.product(grid_shape), dtype=list)
        for v in range(len(m)):
            m[v] = []
        self.grid_experiences = m.reshape(grid_shape)
        # grid occupancy is just a matrix with integers to count experiences
        self.grid_occupancy = np.zeros(grid_shape, dtype=np.int32)

        # calculate state bins
        self.state_bins = []
        for i in range(n_states):
            low, high = env.observation_space.low[i], env.observation_space.high[i]
            bins = np.histogram([low, high], bins=self.n_bins)[1]
            self.state_bins.append(bins[1:])

        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):

        # calculate grid position
        position_new = self.get_position(experience)
        # append to buffer
        self.buffer.append(experience)
        # store in grid
        self.grid_experiences[position_new].append(experience)
        # add to counter
        self.grid_occupancy[position_new] += 1

        # check if a state needs to be removed
        if len(self.buffer) > self.capacity:
            # remove from buffer
            removed_experience = self.buffer.pop(0)
            # calculate position
            position_old = self.get_position(removed_experience)
            # remove from grid - will always be the first to be added
            self.grid_experiences[position_old].pop(0)
            # remove from count
            self.grid_occupancy[position_old] -= 1

    def get_position(self, experience):
        """ Calculate position in grid for a given experience """

        position = []
        state = experience.state
        action = experience.action
        for idx in range(len(state)):
            place = min(self.n_bins-1, int(np.digitize(state[idx], self.state_bins[idx], right=True)))
            position.append(place)
        position.append(action)

        return tuple(position)

    def extend(self, experiences):
        for experience in experiences:
            self.append(experience)
        
    def sample(self, batch_size):
        """ Sample from experience batch based on predetermined rules.
        Main 'meat' from the class is in this method """
        
        # pick random experiences in buffer, with no replacement
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # break down into one tuple per variable of the experience
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        # convert tuples into np arrays
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def sample_with_mask(self, batch_size, mask):
        """ Sample from experience batch based on predetermined rules.
        Main 'meat' from the class is in this method """
        
        # filter only relevant experiences
        selected_experiences = reduce(lambda x,y: x+y, list(self.grid_experiences[mask]))

        # pick random experiences in buffer, with no replacement
        selected_batch_size = min(batch_size, len(selected_experiences))

        # only proceed if batch size is greater than 0
        if selected_batch_size > 0:
            # select indices
            indices = np.random.choice(len(selected_experiences), selected_batch_size, replace=False)
            return [selected_experiences[idx] for idx in indices]

        # else return only empty arrays - maintain the api
        else:
            return []
