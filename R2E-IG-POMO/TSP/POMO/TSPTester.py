
import torch
import numpy as np
import os
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        if env_params["load_path"].endswith(".pkl"):
            self.env = Env(**self.env_params)
        else:
            # for solving instances with TSPLIB format
            self.path_list = [os.path.join(env_params["load_path"], f) for f in sorted(os.listdir(env_params["load_path"]))] \
                if os.path.isdir(env_params["load_path"]) else [env_params["load_path"]]
            assert self.path_list[-1].endswith(".tsp")

        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        if '.pt' in model_load['path']:
            checkpoint_fullname = model_load['path']
        else:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        print(checkpoint_fullname)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator_second()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.path_list:
            for path in self.path_list:
                self._solve_tsplib(path)
        else:
            test_num_episode = self.tester_params['test_episodes']
            episode = 0

            while episode < test_num_episode:

                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)

                # infertime_all=[]
                score, aug_score, infertime = self._test_one_batch(batch_size, episode)
                # infertime_all.append(infertime)
                # print(np.mean(infertime_all))

                score_AM.update(score, batch_size)
                aug_score_AM.update(aug_score, batch_size)

                episode += batch_size

                ############################
                # Logs
                ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
                self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

                all_done = (episode == test_num_episode)

                if all_done:
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" NO-AUG SCORE: {} ".format(score_AM.avg))
                    self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_score_AM.avg))

            # print(np.mean(infertime_all))
            # print(torch.cuda.max_memory_reserved())
            return aug_score_AM.avg

    def _test_one_batch(self, batch_size,episode=None):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor,load_path=self.env_params['load_path'],episode=episode)

            import time
            tik = time.time()

            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        torch.cuda.synchronize()
        tok = time.time()
        # Return
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item(), tok-tik
    
    def _test_one_batch_benchmark(self, test_data):
        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        batch_size = test_data.size(0)
        with torch.no_grad():
            self.env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score

    
    def _solve_tsplib(self, path):
        """
        Solving one instance with TSPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n, 2]
        locations = torch.Tensor(original_locations / original_locations.max())  # Scale location coordinates to [0, 1]
        loc_scaler = original_locations.max()

        env_params = {'problem_size': locations.size(1), 'pomo_size': locations.size(1), 'loc_scaler': loc_scaler, 'distribution': {
        'data_type': 'uniform',  # cluster, mixed, uniform, mix_three
        'n_cluster': 3,
        'n_cluster_mix': 1,
        'lower': 0.2,
        'upper': 0.8,
        'std': 0.07,
        'use_LHS': False,
        'centroid_file': None
    }, 'load_raw': None}
        self.env = Env(**env_params)
        _, _, no_aug_score, aug_score = self._test_one_batch_benchmark(locations)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))


def validate(model, env, batch_size, augment = True, load_path = None):

    # Augmentation
    ###############################################
    if augment:
        aug_factor = 8
    else:
        aug_factor = 1
    # Ready
    ###############################################
    model.eval()
    with torch.no_grad():
        env.load_problems(batch_size, aug_factor,load_path = load_path)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

    # POMO Rollout
    ###############################################
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        # shape: (batch, pomo)
        state, reward, done = env.step(selected)

    # Return
    ###############################################
    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    # shape: (augmentation, batch, pomo)

    max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
    # shape: (augmentation, batch)
    no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

    max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
    # shape: (batch,)
    aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

    return no_aug_score.item(), aug_score.item()
