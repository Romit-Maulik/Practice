import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
np.random.seed(10)

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env

import gym
from gym.spaces import Box

from surrogate_models import coefficient_model, coefficient_model_adjoint
from surrogate_environments import airfoil_surrogate_environment

tf = try_import_tf()

register_env("Airfoil", lambda env_params: airfoil_surrogate_environment(env_params))

#debugging by following example code 
class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
         name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class rl_optimize():
    def __init__(self,env_params,model_args,num_iters,num_steps):
        # Load dataset
        input_data = np.load(dir_path+'/doe_data.npy').astype('float32')
        output_data = np.load(dir_path+'/coeff_data.npy').astype('float32')
        self.action_shape = np.shape(input_data)[1]
        self.obs_shape = np.shape(output_data)[1]

        # Reload model because of TF2/Ray issues
        if model_args == 'regular':
            self.model=coefficient_model(input_data,output_data)
        elif model_args == 'augmented':
            adjoint_data = np.zeros(shape=(170,8)).astype('float32') # placeholder
            self.model=coefficient_model_adjoint(input_data,output_data,adjoint_data)

        ray.init()

        # ModelCatalog.register_custom_model("my_model", CustomModel)

        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 0
        config["num_workers"] = 4
        config["eager"] = False
        config["env_config"] = env_params

        # Add custom model for policy
        model={}
        # model["custom_model"] = "my_model"
        config["model"] = model

        self.trainer = ppo.PPOTrainer(config=config, env="Airfoil")
        self.num_iters = num_iters
        self.num_steps = num_steps

    def train(self):
        # Can optionally call trainer.restore(path) to load a checkpoint.
        for i in range(self.num_iters):
           # Perform one iteration of training the policy with PPO
           result = self.trainer.train()
           print(pretty_print(result))

        # Final save
        checkpoint = self.trainer.save()
        print("Final checkpoint saved at", checkpoint)

        f = open("rl_checkpoint",'w')
        f.write(checkpoint)
        f.close()

    def optimize_shape(self):
        # action = np.random.uniform(size=(self.action_shape))
        from constraints import t_base
        action = np.asarray(t_base)

        self.path = []
        self.obs_path = []
        for _ in range(self.num_steps):
            obs = self.model.predict(action.reshape(1,self.action_shape))
            obs = self.model.op_scaler.inverse_transform(obs)[0,:]
            action = self.trainer.compute_action(obs)

            self.path.append(action)
            self.obs_path.append(obs)

        return self.path, self.obs_path

    def load_checkpoint(self,path):
        self.trainer.restore(path)


if __name__ == "__main__":
    pass



