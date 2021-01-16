import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nmp.model.mlp_block import MLPBlock

# from nmp.model.utils import SublayerConnection, clones
from rlkit.torch import pytorch_util as ptu



from torch.nn import init, Parameter
from torch.autograd import Variable

def identity(x):
    return x


class PointNet(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_sizes,
        robot_props,
        elem_dim,
        q_action_dim,
        input_indices,
        coordinate_frame,
        output_activation=identity,
        init_w=3e-3,
        hidden_activation=F.elu,
        deep_pointnet=False,
        **kwargs,
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_indices = input_indices
        self.link_dim = robot_props[coordinate_frame]["link_dim"]
        self.config_dim = robot_props[coordinate_frame]["config_dim"]
        self.goal_dim = robot_props[coordinate_frame]["goal_rep_dim"]
        self.elem_dim = elem_dim
        self.coordinate_frame = coordinate_frame
        self.output_activation = output_activation
        self.deep_pointnet = deep_pointnet

        self.q_action_dim = q_action_dim
        self.blocks_sizes = get_blocks_sizes(
            self.elem_dim,
            self.config_dim,
            self.goal_dim,
            self.q_action_dim,
            self.hidden_sizes,
            self.coordinate_frame,
        )

        self.block0 = MLPBlock(
            self.blocks_sizes[0],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        self.block1 = MLPBlock(
            self.blocks_sizes[1],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        # if self.deep_pointnet:
        #     self.block01 = MLPBlock(
        #         self.blocks_sizes[1],
        #         hidden_activation=hidden_activation,
        #         output_activation=F.elu,
        #     )
        #     self.block01 = MLPBlock(
        #         self.blocks_sizes[1],
        #         hidden_activation=hidden_activation,
        #         output_activation=F.elu,
        #     )

        self.init_last_fc(output_size, init_w)

    def init_last_fc(self, output_size, init_w=3e-3):
        self.last_fc = nn.Linear(self.hidden_sizes[-1], output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *input, return_features=False):
        obstacles, links, goal, action, mask = process_input(
            self.input_indices, self.elem_dim, self.coordinate_frame, *input
        )
        batch_size = obstacles.shape[0]

        if self.coordinate_frame == "local":
            # early action integration
            h = torch.cat((obstacles, action), dim=2)
            # late action integration
            # h = obstacles
        elif self.coordinate_frame == "global":
            h = torch.cat((obstacles, links, goal, action), dim=2)

        h = self.block0(h)
        h = h * mask[..., None]
        h = torch.max(h, 1)[0]

        if self.coordinate_frame == "local":
            if self.goal_dim > 0:
                h = torch.cat((h, goal), dim=1)

        h = self.block1(h)

        if return_features:
            return h

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()


def get_blocks_sizes(
    elem_dim, config_dim, goal_dim, q_action_dim, hidden_sizes, coordinate_frame,
):
    if coordinate_frame == "local":
        # early action integration
        obstacles_sizes = [elem_dim + q_action_dim] + hidden_sizes
        global_sizes = [hidden_sizes[0] + goal_dim] + hidden_sizes
    elif coordinate_frame == "global":
        obstacles_sizes = [
            elem_dim + config_dim + q_action_dim + goal_dim
        ] + hidden_sizes
        global_sizes = [hidden_sizes[0]] + hidden_sizes

    return obstacles_sizes, global_sizes


def process_input(input_indices, elem_dim, coordinate_frame, *input):
    """
    input: s or (s, a)
    BS x N
    """
    if len(input) > 1:
        out, action = input
    else:
        out, action = input[0], None

    if len(out.shape) == 1:
        out = out.unsqueeze(0)
    batch_size = out.shape[0]
    obstacles = out[:, input_indices["obstacles"]]
    n_elems = obstacles[:, -1]
    obstacles = obstacles[:, :-1]
    obstacles = obstacles.view(batch_size, -1, elem_dim)
    n_elems_pad = obstacles.shape[1]

    mask = torch.arange(n_elems_pad, device=obstacles.device)
    mask = mask[None, :] < n_elems[:, None]

    if action is None:
        action = torch.zeros((batch_size, 0), device=obstacles.device)
    # ealry action integration
    action = action.unsqueeze(1).expand(-1, n_elems_pad, -1)

    goal = out[:, input_indices["goal"]]
    if coordinate_frame == "global":
        goal = goal.unsqueeze(1).expand(batch_size, n_elems_pad, goal.shape[-1])

    links = None
    if coordinate_frame == "global":
        links = out[:, input_indices["robot"]]
        links = links.unsqueeze(1).expand(-1, n_elems_pad, -1)

    return obstacles, links, goal, action, mask




## from https://github.com/Kaixhin/NoisyNet-A3C/blob/master/model.py



# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, device=False):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()
    self.device = device

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight).to(self.device), self.bias + self.sigma_bias * Variable(self.epsilon_bias).to(self.device))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)


class NoisyPointNet(nn.Module):
    def __init__(
            self,
            output_size,
            hidden_sizes,
            robot_props,
            elem_dim,
            q_action_dim,
            input_indices,
            coordinate_frame,
            output_activation=identity,
            init_w=3e-3,
            hidden_activation=F.elu,
            deep_pointnet=False,
            sigma_init=0.017,
            device="cpu",
            **kwargs,
    ):
        super().__init__()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_indices = input_indices
        self.link_dim = robot_props[coordinate_frame]["link_dim"]
        self.config_dim = robot_props[coordinate_frame]["config_dim"]
        self.goal_dim = robot_props[coordinate_frame]["goal_rep_dim"]
        self.elem_dim = elem_dim
        self.coordinate_frame = coordinate_frame
        self.output_activation = output_activation
        self.deep_pointnet = deep_pointnet

        self.q_action_dim = q_action_dim
        self.blocks_sizes = get_blocks_sizes(
            self.elem_dim,
            self.config_dim,
            self.goal_dim,
            self.q_action_dim,
            self.hidden_sizes,
            self.coordinate_frame,
        )

        self.block0 = MLPBlock(
            self.blocks_sizes[0],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        self.block1 = MLPBlock(
            self.blocks_sizes[1],
            hidden_activation=hidden_activation,
            output_activation=F.elu,
        )
        # if self.deep_pointnet:
        #     self.block01 = MLPBlock(
        #         self.blocks_sizes[1],
        #         hidden_activation=hidden_activation,
        #         output_activation=F.elu,
        #     )
        #     self.block01 = MLPBlock(
        #         self.blocks_sizes[1],
        #         hidden_activation=hidden_activation,
        #         output_activation=F.elu,
        #     )
        self.sigma_init = sigma_init
        self.device = device

        self.init_last_fc(output_size, init_w, self.sigma_init)

    def init_last_fc(self, output_size, init_w=3e-3, sigma_init=1):
        # self.last_fc = nn.Linear(self.hidden_sizes[-1], output_size)
        self.last_fc = NoisyLinear(self.hidden_sizes[-1], output_size, sigma_init=sigma_init, device=self.device)
        # self.last_fc.weight.data.uniform_(-init_w, init_w)
        # self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *input, return_features=False):
        obstacles, links, goal, action, mask = process_input(
            self.input_indices, self.elem_dim, self.coordinate_frame, *input
        )
        batch_size = obstacles.shape[0]

        if self.coordinate_frame == "local":
            # early action integration
            h = torch.cat((obstacles, action), dim=2)
            # late action integration
            # h = obstacles
        elif self.coordinate_frame == "global":
            h = torch.cat((obstacles, links, goal, action), dim=2)

        h = self.block0(h)
        h = h * mask[..., None]
        h = torch.max(h, 1)[0]

        if self.coordinate_frame == "local":
            if self.goal_dim > 0:
                h = torch.cat((h, goal), dim=1)

        h = self.block1(h)

        if return_features:
            return h

        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        return output

    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
        return n_params.item()

    def sample_noise(self):
        self.last_fc.sample_noise()

    def remove_noise(self):
        self.last_fc.remove_noise()

