import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import gym
import numpy as np
np.random.seed(10)
from gym import spaces
from surrogate_models import coefficient_model
from constraints import t_lower, t_upper

"""
State:
The current vector of chosen values

Action:
choose the next value
"""
class airfoil_surrogate_environment(gym.Env):

    def __init__(self, env_params):
    
        self.num_params = env_params['num_params']
        self.num_obs = env_params['num_obs']
        self.init_guess = env_params['init_guess'] # Needs to be shape=(1,self.num_params)

        # Load dataset
        input_data = np.load(dir_path+'/DOE_2000.npy').astype('float32')
        output_data = np.load(dir_path+'/coeff_data_2000.npy').astype('float32')

        self.model = coefficient_model(input_data,output_data)

        # Restore model for use in RL
        self.model.restore_model()

        print('Action parameter dimension : ', self.num_params)
        print('Observation parameter dimension : ', self.num_obs)


        lbo = -1+np.zeros(shape=2)
        ubo = 1+np.ones(shape=2)
        self.observation_space = spaces.Box(low=lbo,high=ubo,dtype='double')

        lba = np.asarray(t_lower)
        uba = np.asarray(t_upper)

        self.action_space = spaces.Box(low=lba,high=uba,dtype='double')

        # initialization
        self.max_steps = env_params['num_steps']
        self.start_coeffs = self.model.op_scaler.inverse_transform(\
                            self.model.predict(\
                            self.init_guess.reshape(1,self.num_params)))[0,:] # This is the initial observation

        self.coeffs = self.start_coeffs
       
    def reset(self):
        self.shape_vec = self.init_guess
        self.coeffs = self.start_coeffs
        self.iter = 0
        self.path = []
        self.coeffs_path = []
        
        return self.coeffs
        
    def _take_action(self, action):
        self.path.append(self.shape_vec)
        self.shape_vec = action
        self.iter = self.iter + 1
        
    def step(self, action):
    
        self._take_action(action)       
       
        # Need to use surrogate model (NN based) to calculate coefficients
        input_var = self.shape_vec.reshape(1,self.num_params)        
        
        # pred = self.model.predict(input_var)[0,0]**2# + (self.model.predict(input_var)[0,1]-0.3)**2

        obs = self.model.predict(input_var)
        obs = self.model.op_scaler.inverse_transform(obs)[0,:]
        pred = 0.5*(obs[0]**2)

        self.coeffs = obs
        self.coeffs_path.append(obs)
      
        if self.iter < self.max_steps:
            reward = 0.0
            done = False
        else:
            reward = -pred
            done = True
        
        return obs, reward, done , {}
        
    def render(self, mode="human", close=False):
        pass


if __name__ == '__main__':
    # Create an RL based optimization check
    env_params = {}
    env_params['num_params'] = 8
    env_params['num_obs'] = 2
    env_params['init_guess'] = np.random.uniform(size=(27))
    env_params['num_steps'] = 5

    check = airfoil_surrogate_environment(env_params)
    check.reset()
    check.step(np.random.uniform(size=(8)))