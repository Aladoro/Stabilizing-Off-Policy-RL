
import copy
import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class ChangingModule(abc.ABC):
    def __init__(self, ):
        raise NotImplementedError
        self.drop_layers = []
        self.drop_rates = []
        self.unique_drop = True
        self.need_change = True
        self.mod_type = ''

    def change(self, step):
        metrics = dict()
        if not self.need_change:
            return metrics
        if self.unique_drop:
            drv = utils.schedule(self.drop_rates[0], step)
            for dlayer in self.drop_layers:
                dlayer.update_drop_rate(drv)
            metrics['{}_drop_rate'.format(self.mod_type)] = drv
            return metrics
        else:
            i = 0
            for dlayer, dr in zip(self.drop_layers, self.drop_rates):
                drv = utils.schedule(dr, step)
                dlayer.update_drop_rate(drv)
                metrics['{}_drop_rate_{}'.format(self.mod_type, i)] = drv
            return metrics


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class NoShiftAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape, pretrained=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

        if pretrained:
            pretrained_agent = torch.load(pretrained)
            self.load_state_dict(pretrained_agent.encoder.state_dict())

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def mean(self, obs):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        return mu

    def forward(self, obs, std):
        mu = self.mean(obs)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self._n_parallel = 2

        self.trunk = nn.Sequential(nn.Linear(encoder.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.QS = nn.Sequential(
            utils.DenseParallel(feature_dim + action_shape[0], hidden_dim, self._n_parallel),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, hidden_dim, self._n_parallel),
            nn.ReLU(inplace=True),
            utils.DenseParallel(hidden_dim, 1, 2))

        self.apply(utils.weight_init)

    @property
    def n_parallel(self):
        return self._n_parallel

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        qs = self.QS(h_action)

        return torch.squeeze(torch.transpose(qs, 0, 1), dim=-1)


class DrQV2Agent:
    def __init__(self, encoder, actor, critic,
                 obs_shape, action_shape, device, lr,
                 feature_dim, hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 aug=RandomShiftsAug(pad=4), srank_batches=8,
                 gradient_steps_per_update=1, **kwargs):

        self.train_log_format = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                                 ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                                 ('episode_reward', 'R', 'float'),
                                 ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                                 ('total_time', 'T', 'time')]

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.gradient_steps_per_update = gradient_steps_per_update
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = encoder.to(device)
        self.actor = actor.to(device)

        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        encoder_lr = kwargs.get('encoder_lr', lr)
        actor_lr = kwargs.get('actor_lr', lr)
        critic_lr = kwargs.get('critic_lr', lr)

        # optimizers
        encoder_optimizer_builder = kwargs.get('encoder_optimizer_builder', None)
        if encoder_optimizer_builder is None:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=encoder_lr)
        else:
            self.encoder_opt = encoder_optimizer_builder(encoder)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(),
                                          lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

        # data augmentation
        self.aug = aug

        self.train()
        self.critic_target.train()

        if isinstance(self.encoder, ChangingModule):
            self.encoder_change = True
        else:
            self.encoder_change = False
        if isinstance(self.actor, ChangingModule):
            self.actor_change = True
        else:
            self.actor_change = False
        if isinstance(self.critic, ChangingModule):
            self.critic_change = True
        else:
            self.critic_change = False

        self.srank_batches = srank_batches

    def change_modules(self, step):
        metrics = dict()
        if self.encoder_change:
            metrics.update(self.encoder.change(step))
        if self.actor_change:
            metrics.update(self.actor.change(step))
        if self.critic_change:
            metrics.update(self.critic.change(step))
        return metrics

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def critic_loss(self, enc_obs, action, reward, discount, enc_next_obs,
                    step):
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(enc_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_QS = self.critic_target(enc_next_obs, next_action)
            target_V = target_QS.amin(dim=1, keepdim=True)
            target_Q = reward + (discount * target_V)

        QS = self.critic(enc_obs, action)
        critic_loss = (QS - target_Q).square().sum(1).mean()
        return critic_loss, QS, target_Q, target_V, next_action

    def critic_optim(self, critic_loss):
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

    def update_critic(self, enc_obs, action, reward, discount, enc_next_obs,
                      step, **kwargs):
        metrics = dict()
        critic_loss, QS, target_Q, target_V, next_action = self.critic_loss(
            enc_obs, action, reward, discount, enc_next_obs, step)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = QS[..., 0].mean().item()
            metrics['critic_q2'] = QS[..., 1].mean().item()
            metrics['critic_loss'] = critic_loss.item()

        self.critic_optim(critic_loss)
        return metrics

    def update_actor(self, enc_obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(enc_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        QS = self.critic(enc_obs, action)
        Q = QS.amin(dim=1)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_targets(self,):
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

    def update(self, replay_buffer, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        metrics.update(self.change_modules(step))
        
        for _ in range(self.gradient_steps_per_update):

            batch = next(replay_buffer)
            obs, action, reward, discount, next_obs = utils.to_torch(
                batch, self.device)

            # augment
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            enc_obs = self.encoder(obs)
            with torch.no_grad():
                enc_next_obs = self.encoder(next_obs)

            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()

            # update critic
            metrics.update(
                self.update_critic(enc_obs, action, reward, discount, enc_next_obs, step,
                                   obs=obs, next_obs=next_obs))

            # update actor
            metrics.update(self.update_actor(enc_obs.detach(), step))

            # update critic target
            self.update_targets()

        return metrics

    def calculate_critic_srank(self, replay_buffer, augment=False):
        feat_enc = []
        feat = []
        with torch.no_grad():
            for _ in range(self.srank_batches):
                batch = next(replay_buffer)
                obs, action, _, _, _ = utils.to_torch(batch, self.device)
                if augment:
                    obs = self.aug(obs)
                obs = self.encoder(obs)
                feat_enc.append(obs.cpu().numpy())
                feat.append(utils.get_network_repr_before_final(self.critic, obs, action).cpu().numpy())
            feat_enc = np.concatenate(feat_enc, axis=-2)
            feat = np.concatenate(feat, axis=-2)
            return utils.calculate_feature_srank(feat, delta=0.01), utils.calculate_feature_srank(feat_enc, delta=0.01)
