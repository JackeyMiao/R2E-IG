import torch
from torch import nn
from logging import getLogger


from CVRPEnv import CVRPEnv as Env
from CVRPTester import validate
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
# from tensorboard_logger import Logger as TbLogger
from tensorboardX import SummaryWriter

from utils.utils import *

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def temperature_softmax(s, T=1.0, eps=1e-12):
    s = s.float()  # 确保 float 计算
    s_max = s.max()  # 标量
    exps = torch.exp((s - s_max) / (T + eps))
    denom = exps.sum() + eps
    return exps / denom

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # distribution params
        self.distributions = env_params['distribution_list']
        self.val_distributions = trainer_params['val_distributions']
        self.sample_weight = [1/3, 1/3, 1/3]
        self.cumulate_weight = [1/3, 2/3, 3/3]
        self.weight_temperature = trainer_params['weight_temperature']


        # moe params
        self.aux_weight = trainer_params['aux_weight']
        self.ce_weight = model_params['CE_Weight']

        # result folder, logger
        self.logger = getLogger(name='trainer')
        if self.trainer_params['logging']['tb_logger']:
            self.writer = SummaryWriter('./log/' + get_start_time() +self.trainer_params['tb_path'])
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()


            # Train
            train_score, train_task_loss, train_aux_loss, train_ce_loss, train_loss, loss_dict = self._train_one_moe_epoch(epoch)

            # Record loss of each distribution
            for key in loss_dict.keys():
                locals()[key] = loss_dict[key]

            # Test in three dataset
            if self.trainer_params['multi_test'] and (epoch % self.trainer_params['multi_test_eval']) == 0:
                is_test = True
            else:
                is_test = False


            if is_test:
                val_model, val_env = self.model, self.env
                val_no_aug, val_aug, gap_no_aug, gap_aug  = [], [], [], []
                g = 0
                for k, v in self.trainer_params['val_dataset_multi'].items():
                    # torch.cuda.synchronize()
                    # tik = time.time()
                    no_aug, aug = validate(model=val_model, env=val_env,
                                           batch_size=self.trainer_params['val_batch_size'],
                                           augment=True, load_path=v)
                    # print(time.time()-tik)
                    val_no_aug.append(no_aug)
                    val_aug.append(aug)
                    gap_no_aug.append((no_aug - self.trainer_params['LKH3_optimal'][g]) / self.trainer_params['LKH3_optimal'][g])
                    gap_aug.append((aug - self.trainer_params['LKH3_optimal'][g]) / self.trainer_params['LKH3_optimal'][g])
                    g += 1
                # Update Sample Weight
                def to_cdf(lst):
                    total = sum(lst)
                    cdf = []
                    cumulative = 0
                    for x in lst:
                        cumulative += x
                        cdf.append(cumulative / total)
                    return cdf
                
                
                if self.trainer_params['weight_update']:
                    # Uniform---Cluster---Mixed
                    gap_weight = torch.Tensor([gap_aug[0] + gap_no_aug[0], gap_aug[1] + gap_no_aug[1], gap_aug[2] + gap_no_aug[2]])
                    loss_weight = torch.Tensor([abs(loss_dict['uniform_loss']), abs(loss_dict['cluster_loss']), abs(loss_dict['mixed_loss'])])
                    total_weight = torch.Tensor([0, 0, 0])
                    gap_sum = sum(gap_weight)
                    loss_sum = sum(loss_weight)
                    for i in range(len(self.val_distributions)):
                        gap_weight[i] /= gap_sum
                        loss_weight[i] /= loss_sum
                        total_weight[i] = gap_weight[i] + loss_weight[i]
                    self.sample_weight = temperature_softmax(total_weight, T=self.weight_temperature)
                    self.cumulate_weight = to_cdf(self.sample_weight)

                
            # log
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('train_task_loss', epoch, train_task_loss)
            self.result_log.append('train_aux_loss', epoch, train_aux_loss)
            self.result_log.append('train_ce_loss', epoch, train_ce_loss)
            for _distribution in self.val_distributions:
                self.result_log.append(_distribution + '_score', epoch, locals()[_distribution + '_score'])
                self.result_log.append(_distribution + '_loss', epoch, locals()[_distribution + '_loss'])
                self.result_log.append(_distribution + '_task_loss', epoch, locals()[_distribution + '_task_loss'])
                self.result_log.append(_distribution + '_aux_loss', epoch, locals()[_distribution + '_aux_loss'])
                


            if is_test:
                note = self.val_distributions
                for i in range(len(note)):
                    self.result_log.append(note[i] + '_val_score_noAUG', epoch, val_no_aug[i])
                    self.result_log.append(note[i] + '_val_score_AUG', epoch, val_aug[i])
                    self.result_log.append(note[i] + '_val_gap_noAUG', epoch, gap_no_aug[i])
                    self.result_log.append(note[i] + '_val_gap_AUG', epoch, gap_aug[i])
                self.result_log.append('val_gap_AUG_mean', epoch, np.mean(gap_aug))
                self.result_log.append('val_gap_noAUG_mean', epoch, np.mean(gap_no_aug))


            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d} Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))
            if self.trainer_params['logging']['tb_logger']:
                self.writer.add_scalar('train_score', train_score, epoch)
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('train_task_loss', train_task_loss, epoch)
                self.writer.add_scalar('train_aux_loss', train_aux_loss, epoch)
                self.writer.add_scalar('train_ce_loss', train_ce_loss, epoch)
                for _distribution in self.val_distributions:
                    self.writer.add_scalar(_distribution + '_score', locals()[_distribution + '_score'], epoch)
                    self.writer.add_scalar(_distribution + '_loss', locals()[_distribution + '_loss'], epoch)
                    self.writer.add_scalar(_distribution + '_task_loss', locals()[_distribution + '_task_loss'], epoch)
                    self.writer.add_scalar(_distribution + '_aux_loss', locals()[_distribution + '_aux_loss'], epoch)
                    if not self.trainer_params['mix_strategy']:
                        for i in range(self.model_params['num_experts']):
                            self.writer.add_scalar(_distribution + '_de_expert_' + str(i), self.de_expert_dist[_distribution][i], epoch)
                            for l in range(self.model_params['encoder_layer_num']):
                                self.writer.add_scalar(_distribution + '_en_' + str(l) + '_expert_' + str(i), self.en_expert_dist[_distribution][l][i], epoch)
                if self.trainer_params['mix_strategy']:
                    _distribution = 'mix_three'
                    for i in range(self.model_params['num_experts']):
                        if self.model_params['is_moe']:
                            if self.model_params['decoder_moe']:
                                self.writer.add_scalar(_distribution + '_de_expert_' + str(i), self.de_expert_dist[_distribution][i], epoch)
                            if self.model_params['encoder_moe']:
                                for l in range(self.model_params['encoder_layer_num']):
                                    self.writer.add_scalar(_distribution + '_en_' + str(l) + '_expert_' + str(i), self.en_expert_dist[_distribution][l][i], epoch)

                if is_test:
                    note = self.val_distributions
                    for i in range(len(note)):
                        self.writer.add_scalar(note[i] + '_val_score_noAUG', val_no_aug[i], epoch)
                        self.writer.add_scalar(note[i] + '_val_score_AUG', val_aug[i], epoch)
                        self.writer.add_scalar(note[i] + '_val_gap_noAUG', gap_no_aug[i], epoch)
                        self.writer.add_scalar(note[i] + '_val_gap_AUG', gap_aug[i], epoch)
                    self.writer.add_scalar('val_gap_AUG_mean', np.mean(gap_aug), epoch)
                    self.writer.add_scalar('val_gap_noAUG_mean', np.mean(gap_no_aug), epoch)

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")

                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }

                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # logger
            if is_test:
                note = self.val_distributions
                for i in range(len(note)):
                    self.logger.info("Epoch {:3d}/{:3d} Validate in {}: Gap: noAUG[{:.3f}] AUG[{:.3f}]; Score: noAUG[{:.3f}] AUG[{:.3f}]".format(
                        epoch, self.trainer_params['epochs'],note[i], gap_no_aug[i], gap_aug[i],val_no_aug[i],val_aug[i]))
                self.logger.info("Epoch {:3d}/{:3d} Validate! mean Gap: noAUG[{:.3f}] AUG[{:.3f}]".format(epoch,
                        self.trainer_params['epochs'], np.mean(gap_no_aug)*100, np.mean(gap_aug)*100))
                if self.trainer_params['best']==0:
                    print(self.trainer_params['best'])
                    self.trainer_params['best'] = np.mean(gap_aug)*100
                elif  np.mean(gap_aug)*100 < self.trainer_params['best']:
                    self.trainer_params['best'] = np.mean(gap_aug) * 100
                    self.logger.info("Saving best trained_model")
                    checkpoint_dict = {
                        'epoch': epoch,
                        'best_gap': np.mean(gap_aug) * 100,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'result_log': self.result_log.get_raw_data()
                    }

                    torch.save(checkpoint_dict, '{}/checkpoint-best.pt'.format(self.result_folder))


            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _distill_one_epoch(self, epoch, teacher_prob = 0):
        distill_param = self.trainer_params['distill_param']
        self.logger.info("Start train student model epoch {}".format(epoch))

        # init
        if distill_param['distill_distribution']:
            uniform_score_AM, uniform_loss_AM, uniform_RL_loss_AM, uniform_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            cluster_score_AM, cluster_loss_AM, cluster_RL_loss_AM, cluster_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            mixed_score_AM, mixed_loss_AM, mixed_RL_loss_AM, mixed_KLD_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        else:
            score_AM = AverageMeter()
            loss_AM = AverageMeter()
            RL_loss_AM = AverageMeter()
            KLD_loss_AM = AverageMeter()

        # load the teacher model
        if distill_param['multi_teacher'] and distill_param['distill_distribution']: # multi teacher
            for i in ['uniform', 'cluster', 'mixed']:
                load_path = self.trainer_params['model_load']['load_path_multi'][i]
                self.logger.info(' [*] Loading model from {}'.format(load_path))
                checkpoint = torch.load(load_path, map_location=self.device)
                self.model[i].load_state_dict(checkpoint['model_state_dict'])
                if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']:  # adaptive prob based on the gap
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1,p=distill_param['teacher_prob'])
                else:  # equal prob
                    class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
        elif distill_param['distill_distribution']: # randomly choose a teacher
            if distill_param['adaptive_prob'] and epoch > distill_param['start_adaptive_epoch']: # adaptive prob based on the gap
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], size=1, p=distill_param['teacher_prob'])
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] Loading model from {}, prob: {}'.format(load_path,distill_param['teacher_prob']))
            else: # equal prob
                class_type = np.random.choice(['uniform', 'cluster', 'mixed'], 1)
                load_path = self.trainer_params['model_load']['load_path_multi'][class_type.item()]
                self.logger.info(' [*] Loading model from {}'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else: # distill one teacher
            load_path = self.trainer_params['model_load']['path']
            self.logger.info(' [*] Loading model from {}'.format(load_path))
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            class_type = None if not distill_param['distill_distribution'] else class_type
            if not isinstance(class_type, str) and class_type is not None:
                class_type = class_type.item()

            avg_score, avg_loss, RL_loss, KLD_loss  = self._distill_one_batch(batch_size, distribution=class_type)

            # update variables
            if distill_param['distill_distribution']:
                locals()[class_type + '_score_AM'].update(avg_score, batch_size)
                locals()[class_type + '_loss_AM'].update(avg_loss, batch_size)
                locals()[class_type + '_RL_loss_AM'].update(RL_loss, batch_size)
                locals()[class_type + '_KLD_loss_AM'].update(KLD_loss, batch_size)
            else:
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                RL_loss_AM.update(RL_loss, batch_size)
                KLD_loss_AM.update(KLD_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    if distill_param['distill_distribution']:
                        self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg,
                                                 locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg))
                    else:
                        self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg))

        torch.cuda.empty_cache()
        # Log Once, for each epoch
        if distill_param['distill_distribution']:
            self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                             .format(epoch, 100. * episode / train_num_episode,
                                     locals()[class_type + '_score_AM'].avg,
                                     locals()[class_type + '_loss_AM'].avg,
                                     locals()[class_type + '_RL_loss_AM'].avg,
                                     locals()[class_type + '_KLD_loss_AM'].avg))
            torch.cuda.empty_cache()
            return locals()[class_type + '_score_AM'].avg, locals()[class_type + '_loss_AM'].avg,\
                   locals()[class_type + '_RL_loss_AM'].avg, locals()[class_type + '_KLD_loss_AM'].avg, class_type
        else:
            self.logger.info(
                'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  RL_Loss: {:.4f},  KLD_Loss: {:.4f}'
                .format(epoch, 100. * episode / train_num_episode,
                        score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg))
            torch.cuda.empty_cache()
            return score_AM.avg, loss_AM.avg, RL_loss_AM.avg, KLD_loss_AM.avg

    def _distill_one_batch(self, batch_size,distribution=None):
        distill_param = self.trainer_params['distill_param']
        # Prep
        ###############################################
        self.student_model.train()
        if distill_param['multi_teacher'] and distill_param['distill_distribution']:
            for i in ['uniform', 'cluster', 'mixed']:
                self.model[i].eval()
        else:
            self.model.eval()

        self.env.load_problems(batch_size, distribution=distribution)
        depot_xy = self.env.reset_state.depot_xy
        node_xy = self.env.reset_state.node_xy
        node_demand = self.env.reset_state.node_demand
        self.student_env.load_problems(batch_size, copy=[depot_xy, node_xy, node_demand])

        if distill_param['router'] == 'teacher':# Teacher as the router

            # Teacher
            with torch.no_grad():
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state, attn_type=None)  # No return!

                teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))# shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))
                # Decoding
                while not done:
                    selected, prob, probs = self.model(state, return_probs=True, teacher=True)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)
                    teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                    teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                    teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                teacher_probs = teacher_probs + 0.00001 # avoid log0

            # Student
            student_reset_state, _, _ = self.student_env.reset()
            self.student_model.pre_forward(student_reset_state, attn_type=None) # No return!

            student_prob_list = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0)) # shape: (batch, pomo, 0~problem)
            # POMO Rollout
            ###############################################
            student_state, student_reward, student_done = self.student_env.pre_step()
            student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
            student_probs = torch.zeros(size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size+1, 0))
            # Decoding
            while not student_done:
                student_selected, student_prob, probs = self.student_model(student_state, route=teacher_pi, return_probs=True)
                # shape: (batch, pomo)
                student_state, student_reward, student_done = self.student_env.step(student_selected)
                student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
            student_probs = student_probs + 0.00001 # avoid log0

        else:# Student as the router
            if self.trainer_params['distill_param']['multi_teacher']:
                student_reset_state, _, _ = self.student_env.reset()
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!

                student_prob_list = torch.zeros(
                    size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(
                    size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size + 1, 0))
                # Decoding
                while not student_done:
                    student_selected, student_prob, probs = self.student_model(student_state, return_probs=True)
                    # shape: (batch, pomo)
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                # if not distill_param['KLD_student_to_teacher']:
                student_probs = student_probs + 0.00001  # avoid log0

                # Teacher follow the route of student (multi)
                teacher_probs_multi=[]
                for i in ['uniform', 'cluster', 'mixed']:
                    with torch.no_grad():
                        reset_state, _, _ = self.env.reset()
                        self.model[i].pre_forward(reset_state, attn_type=None)  # No return!

                        teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                        # POMO Rollout
                        ###############################################
                        state, reward, done = self.env.pre_step()
                        teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                        teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))

                        # Decoding
                        while not done:
                            selected, prob, probs = self.model[i](state, route=student_pi, return_probs=True)
                            # shape: (batch, pomo)
                            state, reward, done = self.env.step(selected)
                            teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                            teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                            teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                        # if distill_param['KLD_student_to_teacher']:
                        teacher_probs = teacher_probs + 0.00001  # avoid log0
                        teacher_probs_multi.append(teacher_probs)


            else:
                # Student
                student_reset_state, _, _ = self.student_env.reset()
                self.student_model.pre_forward(student_reset_state, attn_type=None)  # No return!

                student_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                # POMO Rollout
                ###############################################
                student_state, student_reward, student_done = self.student_env.pre_step()
                student_pi = torch.zeros(size=(batch_size, self.student_env.pomo_size, 0))
                student_probs = torch.zeros(
                    size=(batch_size, self.student_env.pomo_size, self.student_env.problem_size + 1, 0))
                # Decoding
                while not student_done:
                    student_selected, student_prob, probs = self.student_model(student_state,return_probs=True)
                    # shape: (batch, pomo)
                    student_state, student_reward, student_done = self.student_env.step(student_selected)
                    student_prob_list = torch.cat((student_prob_list, student_prob[:, :, None]), dim=2)
                    student_pi = torch.cat((student_pi, student_selected[:, :, None]), dim=2)
                    student_probs = torch.cat((student_probs, probs[:, :, :, None]), dim=3)
                # if not distill_param['KLD_student_to_teacher']:
                student_probs = student_probs + 0.00001  # avoid log0

                # Teacher follow the route of student
                with torch.no_grad():
                    reset_state, _, _ = self.env.reset()
                    self.model.pre_forward(reset_state, attn_type=None)  # No return!

                    teacher_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))  # shape: (batch, pomo, 0~problem)
                    # POMO Rollout
                    ###############################################
                    state, reward, done = self.env.pre_step()
                    teacher_pi = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
                    teacher_probs = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size + 1, 0))
                    # Decoding
                    while not done:
                        selected, prob, probs = self.model(state, route=student_pi, return_probs=True)
                        # shape: (batch, pomo)
                        state, reward, done = self.env.step(selected)
                        teacher_prob_list = torch.cat((teacher_prob_list, prob[:, :, None]), dim=2)
                        teacher_pi = torch.cat((teacher_pi, selected[:, :, None]), dim=2)
                        teacher_probs = torch.cat((teacher_probs, probs[:, :, :, None]), dim=3)
                    teacher_probs = teacher_probs + 0.00001  # avoid log0

        assert torch.equal(teacher_pi, student_pi), "Teacher route and student route are not the same!"

        # Loss for student model
        ###############################################
        advantage = student_reward - student_reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = student_prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        task_loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        task_loss = task_loss.mean()

        if distill_param['meaningful_KLD']:
            if distill_param['multi_teacher']:
                for i in range(len(teacher_probs_multi)):
                    if i == 0:
                        soft_loss = (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                    else:
                        soft_loss = soft_loss + (student_probs * (student_probs.log() - teacher_probs_multi[i].log())).sum(dim=2).mean() if distill_param['KLD_student_to_teacher'] \
                            else (teacher_probs_multi[i] * (teacher_probs_multi[i].log() - student_probs.log())).sum(dim=2).mean()
                soft_loss = soft_loss / 3
            else:
                soft_loss = (student_probs * (student_probs.log() - teacher_probs.log())).sum(dim=2).mean() if \
                distill_param['KLD_student_to_teacher'] \
                    else (teacher_probs * (teacher_probs.log() - student_probs.log())).sum(dim=2).mean()
        else:
            soft_loss = nn.KLDivLoss()(student_probs.log(), teacher_probs) if not distill_param['KLD_student_to_teacher'] \
                else nn.KLDivLoss()(teacher_probs.log(), student_probs)
        loss = task_loss * distill_param['rl_alpha'] + soft_loss * distill_param['distill_alpha']

        # Score
        ###############################################
        max_pomo_reward, _ = student_reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################---==
        self.student_model.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

        return score_mean.item(), loss.item(), task_loss.item(), soft_loss.item()


    def _train_one_moe_epoch(self, epoch):
        self.logger.info("Start train model epoch {}".format(epoch))

        # init
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        task_loss_AM, aux_loss_AM, ce_loss_AM = AverageMeter(), AverageMeter(), AverageMeter()
        uniform_score_AM, uniform_task_loss_AM, uniform_aux_loss_AM, uniform_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        cluster_score_AM, cluster_task_loss_AM, cluster_aux_loss_AM, cluster_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        mixed_score_AM, mixed_task_loss_AM, mixed_aux_loss_AM, mixed_loss_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        # MOE Record
        self.en_expert_dist = {}
        self.de_expert_dist = {}
        self.histogram = {
                            'uniform': torch.zeros(self.model_params['num_experts']),
                            'cluster': torch.zeros(self.model_params['num_experts']),
                            'mixed':   torch.zeros(self.model_params['num_experts']),
                        }
        for dist in self.distributions:
            self.en_expert_dist[dist] = torch.zeros(self.model_params['encoder_layer_num'], self.model_params['num_experts'])
            self.de_expert_dist[dist] = torch.zeros(self.model_params['num_experts'])

        # start training
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        step = 0
        while episode < train_num_episode:
            # Params Config
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'] * len(self.distributions), remaining)
            # minibatch_size = self.trainer_params['train_minibatch_size']

            # Update
            avg_score, avg_task_loss, avg_aux_loss, avg_ce_loss, avg_loss, dirtribution_dict  = self._train_one_moe_batch(batch_size, self.cumulate_weight, step)
            

            for _distribution in self.val_distributions:
                _distribution_batchsize = dirtribution_dict[_distribution + '_size']
                locals()[_distribution + '_aux_loss_AM'].update(dirtribution_dict[_distribution + '_aux_loss'], _distribution_batchsize)
                locals()[_distribution + '_task_loss_AM'].update(dirtribution_dict[_distribution + '_task_loss'], _distribution_batchsize)
                locals()[_distribution + '_loss_AM'].update(dirtribution_dict[_distribution + '_loss'], _distribution_batchsize)
                locals()[_distribution + '_score_AM'].update(dirtribution_dict[_distribution + '_score'], _distribution_batchsize)



            # for key in dirtribution_dict.keys():
            #     locals()[key].update(dirtribution_dict[key], actual_minibatch_size)

            score_AM.update(avg_score, batch_size)
            task_loss_AM.update(avg_task_loss, batch_size)
            aux_loss_AM.update(avg_aux_loss, batch_size)
            ce_loss_AM.update(avg_ce_loss, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Task Loss:{:.4f},  Aux Loss:{:.4f},  CE Loss:{:.4f},  Loss: {:.4f},  uniform_Loss: {:.4f},  cluster_Loss: {:.4f},  mixed_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 score_AM.avg, task_loss_AM.avg, aux_loss_AM.avg, ce_loss_AM.avg, loss_AM.avg, uniform_loss_AM.avg, cluster_loss_AM.avg, mixed_loss_AM.avg))
            step += 1

        for dist in self.distributions:
            for l in range(self.model_params['encoder_layer_num']):
                self.en_expert_dist[dist][l] /= self.en_expert_dist[dist][l].sum()
            self.de_expert_dist[dist] /= self.de_expert_dist[dist].sum()

        torch.cuda.empty_cache()
        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Task Loss:{:.4f},  Aux Loss:{:.4f},  CE Loss:{:.4f},  Loss: {:.4f},  uniform_Loss: {:.4f},  cluster_Loss: {:.4f},  mixed_Loss: {:.4f}'
                                         .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                 score_AM.avg, task_loss_AM.avg, aux_loss_AM.avg, ce_loss_AM.avg, loss_AM.avg, uniform_loss_AM.avg, cluster_loss_AM.avg, mixed_loss_AM.avg))
        torch.cuda.empty_cache()
        if self.model_params['is_moe'] and self.model_params['router_method'] == 'instance':
            for dist, count in self.histogram.items():
                total = count.sum()

                if total > 0:
                    self.histogram[dist] = count.float() / total
                else:
                    self.histogram[dist] = count.float()
            
            for dist, tensor in self.histogram.items():
                vals = tensor.detach().cpu().tolist()

                formatted = ", ".join(f"{v:.2f}" for v in vals)

                self.logger.info("%s: [%s]", dist, formatted)

        if self.trainer_params['weight_update']:
            self.logger.info('Epoch {:3d}/{:3d} Weight Factor-----Uniform: {:.4f}    Cluster: {:.4f}    Mixed: {:.4f}'.format(epoch, self.trainer_params['epochs'], self.sample_weight[0], self.sample_weight[1], self.sample_weight[2]))
        loss_dict = {'uniform_score': uniform_score_AM.avg, 
                    'uniform_task_loss': uniform_task_loss_AM.avg,
                    'uniform_aux_loss': uniform_aux_loss_AM.avg,
                    'uniform_loss': uniform_loss_AM.avg,
                    'cluster_score': cluster_score_AM.avg,
                    'cluster_task_loss': cluster_task_loss_AM.avg,
                    'cluster_aux_loss': cluster_aux_loss_AM.avg, 
                    'cluster_loss': cluster_loss_AM.avg,
                    'mixed_score': mixed_score_AM.avg, 
                    'mixed_task_loss': mixed_task_loss_AM.avg,
                    'mixed_aux_loss': mixed_aux_loss_AM.avg,
                    'mixed_loss': mixed_loss_AM.avg}
        
        return score_AM.avg, task_loss_AM.avg, aux_loss_AM.avg, ce_loss_AM.avg, loss_AM.avg, loss_dict

    def _train_one_moe_batch(self, batch_size, cumulate_weight, step):
        # Prep
        ###############################################
        self.model.train()

        # MiniBatch
        ###############################################
        minibatch_size = batch_size // len(self.distributions)
        total_loss = 0.0
        total_task_loss = 0.0
        total_aux_loss = 0.0
        total_ce_loss = 0.0
        score_sum = 0.0
        distribution_loss = {}
        # loss_list = [0] * len(self.distributions)


        for i, _distribution in enumerate(self.distributions):
            if _distribution == 'mix_three':
                # Extra Process
                score, task_loss, aux_loss, ce_loss, info_dict = self._train_one_moe_minibatch(minibatch_size, _distribution, cumulate_weight)
                loss = (task_loss + aux_loss + ce_loss)
                for key in info_dict.keys():     
                    distribution_loss[key] = info_dict[key]
            else:
                score, task_loss, aux_loss, ce_loss = self._train_one_moe_minibatch(minibatch_size, _distribution, cumulate_weight)
                loss = (task_loss + aux_loss + ce_loss)
                distribution_loss[_distribution + '_task_loss'] = task_loss.item()
                distribution_loss[_distribution + '_aux_loss'] = aux_loss.item()
                distribution_loss[_distribution + '_loss'] = loss.item()
                distribution_loss[_distribution + '_score'] = score.item()
                distribution_loss[_distribution + '_size'] = minibatch_size

            # Record
            total_loss += loss
            score_sum += score
            total_task_loss += task_loss
            total_aux_loss += aux_loss
            total_ce_loss += ce_loss
            loss /= self.trainer_params['grad_accumulation']['accumulation_steps']

            loss.backward()
            if (step + 1) % self.trainer_params['grad_accumulation']['accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                


        score_mean = score_sum / len(self.distributions)
        task_loss_mean = total_task_loss / len(self.distributions)
        aux_loss_mean = total_aux_loss / len(self.distributions)
        ce_loss_mean = total_ce_loss / len(self.distributions)
        loss_mean = total_loss / len(self.distributions)    # After Scaling

        

        return score_mean.item(), task_loss_mean.item(), aux_loss_mean.item(), ce_loss_mean.item(), loss_mean.item(), distribution_loss



    def _train_one_moe_minibatch(self, minibatch_size, distribution, cumulate_weight):
        self.env.load_problems(minibatch_size, distribution=distribution, cumulate_weight=cumulate_weight)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        if self.model_params['is_moe'] and self.model_params['router_method'] == 'instance' and self.model_params['decoder_moe']:
            counts, probs = self.expert_distribution(minibatch_size, cumulate_weight, self.model.decoder.histogram_expert)

            self.histogram['uniform'] += probs[0]
            self.histogram['cluster'] += probs[1]
            self.histogram['mixed'] += probs[2]

        
        if self.model_params['is_moe'] and self.model_params['router_method'] == 'instance' and self.model_params['CE_Loss'] :
            ce_loss_mean = self.ce_weight * self.model.compute_ce_loss(minibatch_size, distribution, cumulate_weight)
        else:
            ce_loss_mean = torch.tensor(0.0)

        prob_list = torch.zeros(size=(minibatch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Aux Loss Scaling
        ###############################################
        if self.model_params['is_moe']:
            if self.model_params['encoder_moe'] and self.model_params['decoder_moe']:
                self.model.aux_loss = self.model.aux_loss / (self.model_params['encoder_layer_num'] + self.model.decoder.count)
            elif self.model_params['encoder_moe']:
                self.model.aux_loss = self.model.aux_loss / (self.model_params['encoder_layer_num'])
            else:
                self.model.aux_loss = self.model.aux_loss / self.model.decoder.count
        
        # Aux Loss
        ###############################################
        aux_loss_mean = self.aux_weight * self.model.aux_loss

        # Expert Record
        ###############################################
        if self.model_params['is_moe'] and self.model_params['encoder_moe']:
            for i in range(self.model_params['encoder_layer_num']):
                self.en_expert_dist[distribution][i] += self.model.encoder.layers[i].feed_forward._load

        if self.model_params['is_moe'] and self.model_params['decoder_moe']:
            self.de_expert_dist[distribution] += self.model.decoder.multi_head_combine._load / self.model.decoder.multi_head_combine._load.sum()

        if not self.model_params['is_moe'] or not self.model_params['loading_balance_loss']:
            aux_loss_mean = torch.Tensor([0])


        # Task Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        task_loss_mean = loss.mean()


        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Return
        ###############################################
        if distribution == 'mix_three':
            idx_uni = int(minibatch_size * cumulate_weight[0])
            idx_clu = int(minibatch_size * cumulate_weight[1])
            idx_mix = int(minibatch_size * cumulate_weight[2])
            return score_mean, task_loss_mean, aux_loss_mean, ce_loss_mean, {
                                                                            'uniform_score' : -max_pomo_reward[:idx_uni].float().mean().item(),
                                                                            'uniform_task_loss' : loss[:idx_uni].mean().item(),
                                                                            'uniform_aux_loss' : aux_loss_mean.item(),
                                                                            'uniform_loss': (loss[:idx_uni].mean() + aux_loss_mean).item(),
                                                                            'uniform_size': idx_uni,
                                                                            'cluster_score' : -max_pomo_reward[idx_uni:idx_clu].float().mean().item(),
                                                                            'cluster_task_loss' : loss[idx_uni:idx_clu].mean().item(),
                                                                            'cluster_aux_loss' : aux_loss_mean.item(),
                                                                            'cluster_loss': (loss[idx_uni:idx_clu].mean() + aux_loss_mean).item(),
                                                                            'cluster_size': idx_clu - idx_uni,
                                                                            'mixed_score' : -max_pomo_reward[idx_clu:idx_mix].float().mean().item(),
                                                                            'mixed_task_loss' : loss[idx_clu:idx_mix].mean().item(),
                                                                            'mixed_aux_loss' : aux_loss_mean.item(),
                                                                            'mixed_loss': (loss[idx_clu:idx_mix].mean() + aux_loss_mean).item(),
                                                                            'mixed_size': idx_mix - idx_clu,
                                                                            }
        else:
            return score_mean, task_loss_mean, aux_loss_mean, ce_loss_mean




    def expert_distribution(self, minibatch_size, cumulate_weight, indices):
        B, N, E = indices.shape

        # Calculate dict
        idx_uni = int(minibatch_size * cumulate_weight[0])      # [0, idx_uni) -> uniform
        idx_clu = int(minibatch_size * cumulate_weight[1])      # [idx_uni, idx_clu) -> cluster
        idx_mix = int(minibatch_size * cumulate_weight[2]) 
        uni_size = idx_uni
        clu_size = idx_clu - idx_uni
        mix_size = idx_mix - idx_clu
        assert uni_size + clu_size + mix_size == B

        num_experts = self.model_params['num_experts']

        uniform_idx = indices[:idx_uni].reshape(-1)
        cluster_idx = indices[idx_uni:idx_clu].reshape(-1)
        mixed_idx   = indices[idx_clu:idx_mix].reshape(-1)

        uniform_count = torch.bincount(uniform_idx, minlength=num_experts)
        cluster_count = torch.bincount(cluster_idx, minlength=num_experts)
        mixed_count   = torch.bincount(mixed_idx,   minlength=num_experts)

        counts = torch.stack([uniform_count, cluster_count, mixed_count], dim=0)
        # 0: uniform, 1: cluster, 2: mixed

        probs = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)

        return counts, probs
