##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 3

##########################################################################################
# Path Config

import random, os
import sys
import warnings
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer
import torch
import numpy as np


##########################################################################################
# parameters

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
    'distribution_list': ['uniform', 'cluster', 'mixed'],
    'distribution': {
        'data_type': 'uniform',  # cluster, mixed, uniform
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
    },
    'seed': 1234
}
env_params['load_raw']=None
LKH3_optimal = {
    20: [6.156523, 3.05725599999999, 5.439149],
    50: [10.417558, 5.155511, 9.354149],
    100: [15.740834, 7.909336, 14.294179]
}

model_params = {
    'normalization':'instance',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'intermediate_dim' : 128, # MOE
    'num_experts': 8,
    'top_k': 3,
    'used_shared_expert': True,
    'loading_balance_loss': True,
    'is_moe': True,
    'encoder_moe': True,
    'decoder_moe': True,
    'expert_method': 'top_k',    # 'sampling' or 'top_k'
    'type_expert': 'Res',   # origin, Res, Res_wo_shortcut, Res_wo_silu, Res_wo_res
    'router_method': 'instance',    # instance, node
    'CE_Loss': True,
    'CE_Weight': 0.1,
}


optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones':[900],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 5000,
    'train_episodes': 20* 1000,
    'train_batch_size': 256,
    'prev_model_path': None,
    'mix_strategy': True,    # mixed
    'weight_update': True,
    'weight_temperature': 1,
    'aux_weight': 0.01,
    'multi_test': True,
    'multi_test_eval': 1,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_{}_{}.json'.format(env_params['problem_size'],env_params['distribution']['data_type'])
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
        'tb_logger': True
    },
    'grad_accumulation':{
        'enable': False,
        'accumulation_steps': 1,
    },
    'model_load': {
        'enable': False,
    },
    'best': 0
}

if trainer_params['mix_strategy']:
    env_params['distribution_list'] = ['mix_three']
else:
    trainer_params['weight_update'] = False

# Limit of CUDA Memory Choice
# if env_params['problem_size'] == 100:
#     env_params['pomo_size'] = 100
#     trainer_params['train_batch_size'] = 85
#     trainer_params['grad_accumulation']['enable'] = True
#     trainer_params['grad_accumulation']['accumulation_steps'] = 3


if trainer_params['multi_test']:
    # uniform, cluster, mixed
    trainer_params['LKH3_optimal'] = LKH3_optimal[env_params['problem_size']]
    trainer_params['val_distributions'] = ['uniform', 'cluster', 'mixed']
    trainer_params['val_batch_size'] = 1000
    trainer_params['val_dataset_multi'] = {
        'uniform': './data/vrp_uniform{}_1000_seed1234.pkl'.format(env_params['problem_size']),
        'cluster': './data/vrp_cluster{}_1000_seed1234.pkl'.format(env_params['problem_size']),
        'mixed': './data/vrp_mixed{}_1000_seed1234.pkl'.format(env_params['problem_size'])
    }

###################################################
logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n{}_epoch{}_{}_batchsize{}_{}Norm'.format(env_params['problem_size'],trainer_params['epochs'],
                                                      env_params['distribution']['data_type'],trainer_params['train_batch_size'],
                                                                      model_params['normalization']),
        'filename': 'run_log'
    }
}


trainer_params['tb_path'] = logger_params['log_file']['desc']
##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    _set_seed(env_params['seed'])

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 100
    trainer_params['train_episodes'] = 100
    trainer_params['train_batch_size'] = 32
    trainer_params['multi_test_eval'] = 1


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Global seed set to {seed}")

##########################################################################################

if __name__ == "__main__":
    main()
