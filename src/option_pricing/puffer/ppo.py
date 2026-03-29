import pufferlib.ocean
import pufferlib.vector
from pufferlib import pufferl
import torch
import os
import sys
PROJECT_ROOT = os.path.abspath("..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)        

from algorithms import *






default_params = {
    'n_processes' : 10, 
    'n_rights' : 1,
    'max_steps' : 50,
}


# Simple trainer based on pufferl functions
def ppo_trainer(env_name='puffer_option_pricing', env_params = None):
    args = pufferl.load_config(env_name)

    # You can customize the puffer-provided config
    if env_params is not None:
        for key in env_params:
            args['env'][key] = env_params[key]

    # Or, you can create and use a separate config file
    # args = pufferl.load_config_file(<YOUR_OWN_CONFIG.ini>, fill_in_default=True)
    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()
        vecenv.render()

    trainer.print_dashboard()
    torch.save(policy.state_dict(), "ppo_option_policy.pt")
    print(trainer.evaluate())
    trainer.close()
    return policy

if __name__ == "__main__":
    ppo_trainer(env_name='puffer_pong')






