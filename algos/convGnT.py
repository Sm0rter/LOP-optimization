from torch.nn import Conv2d, Linear
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
import torch
import sys
from lop.utils.AdamGnT import AdamGnT
from torch.nn.init import calculate_gain
from lop.utils.miscellaneous import get_layer_bound
from lop.algos.Nnetmon import Nnetmon


class ConvGnT(object):
    """
    Generate-and-Test algorithm for ConvNets, maturity threshold based tester, accumulates probability of replacement,
    with various measures of feature utility
    """
    def __init__(self, net, hidden_activation, opt, decay_rate=0.99, replacement_rate=1e-4, init='kaiming',
                 num_last_filter_outputs=4, util_type='contribution', maturity_threshold=100, device='cpu'):
        super(ConvGnT, self).__init__()

        self.net = net
        self.num_hidden_layers = int(len(self.net)/2)
        self.util_type = util_type
        self.device = device

        self.health_mon = Nnetmon(net, decay_rate=decay_rate, replacement_rate=replacement_rate,
                                  num_last_filter_outputs=num_last_filter_outputs,util_type=util_type,
                                  maturity_threshold=maturity_threshold, device=device
                                  )

        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'AdamGnT'


        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.util_type = util_type

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        if hidden_activation == 'selu': init = 'lecun'
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)
        """
        Pre calculate number of features to replace per layer per update
        """
        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                if isinstance(self.net[i * 2], Linear):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_features)
                elif isinstance(self.net[i * 2], Conv2d):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_channels)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']: hidden_activation = 'relu'
        bounds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            bounds.append(get_layer_bound(layer=self.net[i * 2], init=init, gain=gain))
        bounds.append(get_layer_bound(layer=self.net[-1], init=init, gain=1))
        return bounds

    def save_states(self):
        self.health_mon.flush_to_file()

    def update_optim_params(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Update Optimizer's state
        """
        if self.opt_type == 'AdamGnT':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace[i] == 0:
                    continue
                # input weights
                self.opt.state[self.net[i * 2].bias]['exp_avg'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['exp_avg_sq'][features_to_replace_input_indices[i], :] = 0.0
                self.opt.state[self.net[i * 2].bias]['exp_avg_sq'][features_to_replace_input_indices[i]] = 0.0
                self.opt.state[self.net[i * 2].weight]['step'][features_to_replace_input_indices[i], :] = 0
                self.opt.state[self.net[i * 2].bias]['step'][features_to_replace_input_indices[i]] = 0
                # output weights
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['exp_avg_sq'][:, features_to_replace_output_indices[i]] = 0.0
                self.opt.state[self.net[i * 2 + 2].weight]['step'][:, features_to_replace_output_indices[i]] = 0

    def gen_new_features(self, features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
                current_layer = self.net[i * 2]
                next_layer = self.net[i * 2 + 2]

                if isinstance(current_layer, Linear):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty(num_features_to_replace[i], current_layer.in_features).uniform_(-self.bounds[i],
                                                                                                self.bounds[i]).to(self.device)
                elif isinstance(current_layer, Conv2d):
                    current_layer.weight.data[features_to_replace_input_indices[i], :] *= 0.0
                    current_layer.weight.data[features_to_replace_input_indices[i], :] -= - \
                        empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:])). \
                            uniform_(-self.bounds[i], self.bounds[i]).to(self.device)

                current_layer.bias.data[features_to_replace_input_indices[i]] *= 0.0
                """
                # Set the outgoing weights and ages to zero
                """
                next_layer.weight.data[:, features_to_replace_output_indices[i]] = 0
                self.health_mon.reset_feature_stats(i, features_to_replace_input_indices[i])

    def gen_and_test(self, features, task_id = -1, epoch_id = -1):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace \
            = self.health_mon.test_features(features=features)
        self.health_mon.update_nn_health(features)
        self.gen_new_features(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)
        self.update_optim_params(features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace)

        self.health_mon.save_states(task_id=task_id, epoch_id=epoch_id, save_frequency = 1)