
import torch
import numpy as np
import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
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
        #

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
            # for solving instances with CVRPLIB format
            # self.path_list = [os.path.join(env_params["load_path"], f) for f in sorted(os.listdir(env_params["load_path"]))] \
            #     if os.path.isdir(env_params["load_path"]) else [env_params["load_path"]]
            self.path_list = (
                [
                    os.path.join(env_params["load_path"], f)
                    for f in sorted(os.listdir(env_params["load_path"]))
                    if f.endswith(".vrp")
                    and os.path.isfile(os.path.join(env_params["load_path"], f))
                ]
                if os.path.isdir(env_params["load_path"])
                else ([env_params["load_path"]] if env_params["load_path"].endswith(".vrp") else [])
            )

            assert self.path_list[-1].endswith(".vrp")
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
        self.expert_dist = torch.zeros(self.model_params["num_experts"])

    def run(self):

        ##########################################################################################
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        path_list = getattr(self, "path_list", None)

        if path_list:
            for path in self.path_list:
                self._solve_cvrplib(path)
        else:
            if self.tester_params['test_data_load']['enable']:
                print(self.tester_params['test_data_load']['filename'])
                self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)


            test_num_episode = self.tester_params['test_episodes'] #1000
            episode = 0

            inferTime = []
            graph_embedding_list = []
            
            while episode < test_num_episode:

                remaining = test_num_episode - episode
                batch_size = min(self.tester_params['test_batch_size'], remaining)

                import time
                tik = time.time()
                if self.tester_params['record_tsne']:
                    score, aug_score, graph_embedding_batch = self._test_one_batch(batch_size, episode)
                    graph_embedding_list.append(graph_embedding_batch)
                else:
                    score, aug_score = self._test_one_batch(batch_size, episode)

            
                torch.cuda.synchronize()
                tok =time.time()
                inferTime.append(tok-tik)
                # print(np.mean(inferTime))

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
                    self.expert_dist /= self.expert_dist.sum()
                    self.logger.info(" *** Test Done *** ")
                    self.logger.info(" EXPERT DISTRIBUTION: {} ".format(self.expert_dist))
                    self.logger.info(" NO-AUG SCORE: {} ".format(score_AM.avg))
                    self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_score_AM.avg))
            if self.tester_params['record_tsne']:
                graph_embedding = torch.cat(graph_embedding_list, dim=0).cpu().numpy()
                data_name_tmp = self.env_params['load_path'].split('/')[-1].split('.')[0].split('_')
                data_name = data_name_tmp[0] + '_' + data_name_tmp[1] + '_' + str(test_num_episode) 
                np.save(f"./tsne/{data_name}.npy", graph_embedding)
                # print(graph_embedding.shape)
            return score_AM.avg, aug_score_AM.avg, np.mean(inferTime)

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
            self.env.load_problems(batch_size, aug_factor,load_path=self.env_params['load_path'], episode=episode)
            # self.env.load_problems(batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, attn_type='qk_scaled')
        if self.tester_params['record_tsne']:
            graph_embedding_batch = self.model.decoder.graph_embedding[::self.tester_params['aug_factor'],0,:]

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

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
        # d = expert_distribution(self.model.decoder.histogram_expert[:,0,:].reshape(-1), self.model_params["num_experts"])
        # self.expert_dist.add_(d)

        if self.tester_params['record_tsne']:
            return no_aug_score.item(), aug_score.item(), graph_embedding_batch
        else:
            return no_aug_score.item(), aug_score.item()
    
    def _test_one_batch_benchmark(self, test_data):
        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        batch_size = test_data[-1].size(0)
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
    
    def _solve_cvrplib(self, path):
        """
        Solving one instance with CVRPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        loc_scaler = 1000
        locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]

        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler, 'distribution': {
            'data_type': 'uniform',  # cluster, mixed, uniform
            'n_cluster': 3,
            'n_cluster_mix': 1,
            'lower': 0.2,
            'upper': 0.8,
            'std': 0.07,
        }, 'load_raw': None}
        self.env = Env(**env_params)
        data = (depot_xy, node_xy, node_demand)
        _, _, no_aug_score, aug_score = self._test_one_batch_benchmark(data)
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

    episode = 0
    test_num_episode = 1000
    no_aug_score1=[]
    aug_score1=[]
    while episode < test_num_episode:
        remaining = test_num_episode - episode
        batch_size = min(batch_size, remaining)
        with torch.no_grad():
            env.load_problems(batch_size, aug_factor, load_path=load_path, episode=episode)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
        # POMO Rollout
        ###############################################
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
        ###############################################
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
        no_aug_score1.append(no_aug_score.item())
        aug_score1.append(aug_score.item())
        episode += batch_size

    import numpy as np
    return np.mean(no_aug_score1),np.mean(aug_score1)

def expert_distribution(indices, minlength):
    tmp = torch.bincount(indices, minlength=minlength)
    tmp = tmp / tmp.sum()
    return tmp