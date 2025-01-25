import torch
from torch.nn import Conv2d, Linear
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor, ones
from lop.utils.miscellaneous import erank
from lop.utils.miscellaneous import save_data
from datetime import date, datetime
import time
import pickle
import os
#import gc
#import psutil
#process = psutil.Process(os.getpid())

class Nnetmon(object):
    def __init__(self, net, batches_per_epoch=12, rec_interval=10, decay_rate=0.99, replacement_rate=1e-4,
                 num_last_filter_outputs=4, util_type=None, maturity_threshold=100, total_tasks=2000,
                 epochs_per_task=250, logfile='data/nn_summary', device='cpu'):
        super(Nnetmon, self).__init__()
        self.net = net
        self.num_hidden_layers = int(len(self.net) / 2)
        self.device = device

        # current batch count
        self.curr_batch = 0

        self.batches_per_epoch = batches_per_epoch

        self.current_epoch = 0
        self.rec_interval = rec_interval
        self.health_history = int(total_tasks * epochs_per_task / self.rec_interval)

        # output data file
        self.data_file = logfile + '_' + str(date.today()) + '_' + str(datetime.now().hour) + '_' + str(
            datetime.now().minute)

        self.num_last_filter_outputs = num_last_filter_outputs
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.replacement_rate = replacement_rate
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util, self.bias_corrected_util, self.ages, self.mean_feature_act, self.mean_abs_feature_act, \
            = [], [], [], [], []

        # dynamic replacement rate
        '''
        magnifier coefficient of each neuron
            mag ::= |w_out|/|w_in|
        growth rate of a neurons
            g_rate ::= |h(t)-h(t-1)|
        neuron value 
        '''
        self.mag, self.val_p, self.g_rate, self.dead_neuron, self.h_t_1, self.removed_neuron = [], [], [], [], [], []
        self.saved_data = []
        self.curr_task_id = -1
        self.curr_epoch_id = -1
        self.bp_id = -1
        self.f_prefix = None
        self.curr_erank = zeros(self.num_hidden_layers, device=self.device)
        self.prev_erank = zeros(self.num_hidden_layers, device=self.device)

# has more checks, but not necessary to add
        if self.util_type in ['nova_contribution', 'nova_contribution dead neuron']:
            print('Initialize nova contribution', self.util_type)

        for i in range(self.num_hidden_layers):
            if isinstance(self.net[i * 2], Conv2d):
                n_neurons = self.net[i * 2].out_channels

            elif isinstance(self.net[i * 2], Linear):
                n_neurons = self.net[i * 2].out_features

            self.util.append(zeros(n_neurons, device=self.device))
            self.bias_corrected_util.append(zeros(n_neurons, device=self.device))
            self.ages.append(zeros(n_neurons, device=self.device))
            self.mean_feature_act.append(zeros(n_neurons, device=self.device))
            self.mean_abs_feature_act.append(zeros(n_neurons, device=self.device))
            self.removed_neuron.append(zeros(n_neurons, device=self.device))

            self.mag.append(zeros(n_neurons, device=self.device))
            self.g_rate.append(zeros(n_neurons, device=self.device))
            self.val_p.append(zeros(n_neurons, device=self.device))
            self.h_t_1.append(None)
            self.dead_neuron.append(zeros(n_neurons, device=self.device))
            self.removed_neuron.append(zeros(n_neurons, device=self.device))

        # current sequence ID for Nnet health summary
        self.sid = 0

        self.task, self.epoch, self.e_rank, self.dead_n, self.f_val, self.mag_w = [], [], [], [], [], []
        self.task = torch.zeros(self.health_history, dtype=torch.int, device=self.device)
        self.epoch = torch.zeros(self.health_history, dtype=torch.int, device=self.device)
        self.e_rank = torch.zeros(self.health_history, self.num_hidden_layers, dtype=torch.float, device=self.device)
        self.dead_n = torch.zeros(self.health_history, self.num_hidden_layers, dtype=torch.int, device=self.device)
        self.f_val = torch.zeros(self.health_history, self.num_hidden_layers, dtype=torch.float, device=self.device)
        self.mag_w = torch.zeros(self.health_history, self.num_hidden_layers, dtype=torch.float, device=self.device)

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            with no_grad():
                if isinstance(self.net[i * 2], Linear):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_features)
                elif isinstance(self.net[i * 2], Conv2d):
                    self.num_new_features_to_replace.append(self.replacement_rate * self.net[i * 2].out_channels)

    def update_nn_health(self, features):
        with torch.no_grad():
            self.curr_batch += 1
            if self.curr_batch % self.batches_per_epoch > 0:
                return
            self.current_epoch += 1
            if self.current_epoch % self.rec_interval > 0:
                return

            self.task[self.sid] = self.curr_task_id
            self.epoch[self.sid] = self.current_epoch
            for i in range(self.num_hidden_layers):
                fshape = features[i].shape
                self.e_rank[self.sid] = erank(features[i].view(fshape[0], -1), use_scipy=True)
                self.prev_erank[i] = self.curr_erank[i]
                self.curr_erank[i] = self.e_rank[self.sid][i]
                self.dead_n[self.sid][i] = (features[i].abs().sum(dim=0) == 0).sum()
                self.f_val[self.sid][i] = features[i].abs().mean()
                self.mag_w[self.sid][i] = self.net[i * 2].weight.data.abs().mean()

            self.sid += 1
            if self.sid % 10 == 0:
                save_data(data={
                    'total_history': self.sid,
                    'task': self.task.cpu(),
                    'epoch': self.epoch.cpu(),
                    'e_rank': self.e_rank.cpu(),
                    'dead_n': self.dead_n.cpu(),
                    'feature_abs_val': self.f_val.cpu(),
                    'weight_magnitude': self.mag_w.cpu(),
                }, data_file=self.data_file)
                print('data saved at', self.data_file, 'total number saved at', self.sid)

    def reset_feature_stats(self, layer_idx=0, inds=None):
        self.util[layer_idx][inds] = 0.
        self.bias_corrected_util[layer_idx][inds] = 0.
        self.ages[layer_idx][inds] = 0.
        self.mean_feature_act[layer_idx][inds] = 0.
        self.mean_abs_feature_act[layer_idx][inds] = 0.
        self.mag[layer_idx][inds] = 0.
        self.g_rate[layer_idx][inds] = 0.
        self.val_p[layer_idx][inds] = 0.

    def init_saved_data(self):
        with torch.no_grad():

            self.saved_data = []
            self.bp_id = 0
            for i in range(self.num_hidden_layers):
                self.removed_neuron[i] = zeros(len(self.removed_neuron[i]), device=self.device)

    def flush_to_file(self, file_prefix='data/nn_health'):
        if self.f_prefix is None:
            self.f_prefix = file_prefix + str(date.today()) + '_' + str(datetime.now().hour) + '_' + str(
                datetime.now().minute) + '/'
            print(self.f_prefix)
            os.makedirs(self.f_prefix, exist_ok = True)
            time.sleep(1)

        print('flushing', self.curr_task_id, 'to file')
        with open(str(self.f_prefix) + str(self.curr_task_id), 'wb+') as f:
            pickle.dump(self.saved_data, f)
        self.init_saved_data()

    def save_incremental(self):
        with torch.no_grad():
            data = {
                'task_id': self.curr_task_id,
                'epoch_id': self.curr_epoch_id,
                'bp_iter_id': self.bp_id,
                'erank 0': self.curr_erank[0].cpu(),
                'erank 1': self.curr_erank[1].cpu(),
                'erank 2': self.curr_erank[2].cpu(),
                'feature 0': self.mean_abs_feature_act[0].clone().detach().cpu(),
                'feature 1': self.mean_abs_feature_act[1].clone().detach().cpu(),
                'feature 2': self.mean_abs_feature_act[2].clone().detach().cpu(),
                'mag_ratio 0': self.mag[0].clone().detach().cpu(),
                'mag_ratio 1': self.mag[1].clone().detach().cpu(),
                'mag_ratio 2': self.mag[2].clone().detach().cpu(),
                'value_prop 0': self.val_p[0].clone().detach().cpu(),
                'value_prop 1': self.val_p[1].clone().detach().cpu(),
                'value_prop 2': self.val_p[2].clone().detach().cpu(),
                'growth_rate 0': self.g_rate[0].clone().detach().cpu(),
                'growth_rate 1': self.g_rate[1].clone().detach().cpu(),
                'growth_rate 2': self.g_rate[2].clone().detach().cpu(),
                'dead_neuron 0': self.dead_n[0].clone().detach().cpu(),
                'dead_neuron 1': self.dead_n[1].clone().detach().cpu(),
                'dead_neuron 2': self.dead_n[2].clone().detach().cpu(),
                'utility 0': self.util[0].clone().detach().cpu(),
                'utility 1': self.util[1].clone().detach().cpu(),
                'utility 2': self.util[2].clone().detach().cpu(),
                'rm_neuron 0': self.removed_neuron[0].clone().detach().cpu(),
                'rm_neuron 1': self.removed_neuron[1].clone().detach().cpu(),
                'rm_neuron 2': self.removed_neuron[2].clone().detach().cpu(),
            }
        self.saved_data.append(data)
        self.removed_neuron = []
        for i in range(self.num_hidden_layers):
            self.removed_neuron.append(zeros(len(self.ages[i]), device=self.device))

    def save_states(self, task_id=-1, epoch_id=-1, save_frequency=1):
        with torch.no_grad():
            if self.curr_task_id == -1:
                self.curr_task_id = task_id
                self.curr_epoch_id = epoch_id
                self.init_saved_data()
            else:
                if self.curr_task_id != task_id:
                    self.flush_to_file()
                    self.init_saved_data()
                    self.curr_task_id = task_id
                    self.curr_epoch_id = epoch_id
            self.curr_epoch_id = epoch_id
            self.bp_id += 1
            if self.curr_task_id % save_frequency != 0:
                return
            self.save_incremental()
        return

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace_input_indices = [empty(0, dtype=long, device=self.device) for _ in
                                             range(self.num_hidden_layers)]
        features_to_replace_output_indices = [empty(0, dtype=long, device=self.device) for _ in
                                              range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace
        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            """
            Update feature utility
            """
            self.update_utility(layer_idx=i, features=features[i])
            if self.util_type is None:
                continue
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[i] += self.num_new_features_to_replace[i]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0:
                continue


            """
            Find features to replace in the current layer
            """
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                           num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]
            self.removed_neuron[i][new_features_to_replace] = 1
            """
            Initialize utility for new features
            """

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace_input_indices[i] = new_features_to_replace
            features_to_replace_output_indices[i] = new_features_to_replace
            if isinstance(self.net[i * 2], Conv2d) and isinstance(self.net[i * 2 + 2], Linear):
                features_to_replace_output_indices[i] = \
                    (new_features_to_replace * self.num_last_filter_outputs).repeat_interleave(
                        self.num_last_filter_outputs) + \
                    tensor([i for i in range(self.num_last_filter_outputs)]).repeat(
                        new_features_to_replace.size()[0])
            self.reset_feature_stats(i, new_features_to_replace)
        return features_to_replace_input_indices, features_to_replace_output_indices, num_features_to_replace

    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            if self.h_t_1[layer_idx] is None:
                self.h_t_1[layer_idx] = features * 0
            self.util[layer_idx] *= self.decay_rate
            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_abs_feature_act[layer_idx] *= self.decay_rate
            self.mag[layer_idx] *= self.decay_rate
            self.g_rate[layer_idx] *= self.decay_rate
            self.val_p[layer_idx] *= self.decay_rate

            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            current_layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]

            if isinstance(next_layer, Linear):
                output_weight_mag = next_layer.weight.data.abs().mean(dim=0)
            elif isinstance(next_layer, Conv2d):
                output_weight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))

            if isinstance(current_layer, Linear):
                input_weight_mag = current_layer.weight.data.abs().mean(dim=1)
                feature_mean = features.mean(dim=0)
                feature_abs_mean = features.abs().mean(dim=0)
                feature_abs_sum = features.abs().sum(dim=0)
                abs_growth_rate = (features - self.h_t_1[layer_idx]).abs().mean(dim=0)

            elif isinstance(current_layer, Conv2d):
                input_weight_mag = current_layer.weight.data.abs().mean(dim=(1, 2, 3))

                if isinstance(next_layer, Conv2d):
                    feature_mean = features.mean(dim=(0,2,3))
                    feature_abs_mean = features.abs().mean(dim=(0,2,3))
                    feature_abs_sum = features.abs().sum(dim=(0,2,3))
                    abs_growth_rate = (features - self.h_t_1[layer_idx]).abs().mean(dim=(0,2,3))
                else:

                    feature_mean = features.mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)

                    feature_abs_mean = features.abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)

                    feature_abs_sum = features.abs().sum(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)

                    output_weight_mag = output_weight_mag.view(-1, self.num_last_filter_outputs).mean(dim=1)

                    abs_growth_rate = ((features - self.h_t_1[layer_idx]).abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1))

        self.mean_feature_act[layer_idx] += (1-self.decay_rate)*feature_mean
        self.mean_abs_feature_act[layer_idx] += (1-self.decay_rate)*feature_abs_mean
        self.mag[layer_idx] += (1-self.decay_rate)*output_weight_mag/input_weight_mag
        self.g_rate[layer_idx] += (1-self.decay_rate)*abs_growth_rate
        self.val_p[layer_idx] += (1-self.decay_rate)*feature_abs_mean*output_weight_mag
        del abs_growth_rate
        bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

        if self.util_type == 'adaptation':
            new_util = 1 / input_weight_mag
        elif self.util_type in ['contribution', 'zero_contribution', 'adaptable_contribution']:
            if self.util_type == 'contribution':
                bias_corrected_act = 0
            else:
                if isinstance(current_layer, Conv2d):
                    if isinstance(next_layer, Conv2d):
                        bias_corrected_act = bias_corrected_act.view(1, -1, 1, 1)
                    else:
                        bias_corrected_act = bias_corrected_act.repeat_interleave(
                            self.num_last_filter_outputs).view(1, -1)
            if isinstance(next_layer, Linear):
                if isinstance(current_layer, Linear):
                    new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0)
                elif isinstance(current_layer, Conv2d):
                    new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=0).view(-1, self.num_last_filter_outputs).mean(dim=1)
            elif isinstance(next_layer, Conv2d):
                new_util = output_weight_mag * (features - bias_corrected_act).abs().mean(dim=(0, 2, 3))
            if self.util_type == 'adaptable_contribution':
                new_util = new_util / input_weight_mag

        elif (self.util_type in ['nova_contribution', 'nova_contribution dead neuron', 'nova_contribution dead neuron'
                                , 'nova_contribution g_rate', 'nova_contribution mag_debias'
                                , 'nova_contribution value prop', 'nova_contribution feature.abs.mean']) or (self.util_type is None):
            new_util = self.g_rate[layer_idx] * self.mag[layer_idx] * self.val_p[layer_idx]


        if self.util_type == 'random':
            self.bias_corrected_util[layer_idx] = rand(self.util[layer_idx].shape, device=self.device)
        else:
            self.util[layer_idx] += (1 - self.decay_rate) * new_util
            # correct the bias in the utility computation
            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction
        if self.util_type in ['nova_contribution feature.abs.mean']:
            self.bias_corrected_util[layer_idx] = self.mean_abs_feature_act[layer_idx]
        if self.util_type in ['nova_contribution dead neuron']:
            self.bias_corrected_util[layer_idx] = self.dead_neuron[layer_idx]
        if self.util_type in ['nova_contribution mag']:
            self.bias_corrected_util[layer_idx] = self.mag[layer_idx]
        if self.util_type in ['nova_contribution g_rate']:
            self.bias_corrected_util[layer_idx] = self.g_rate[layer_idx]
        if self.util_type in ['nova_contribution value prop']:
            self.bias_corrected_util[layer_idx] = self.val_p[layer_idx]
        if self.util_type in ['nova_contribution mag_debias']:
            self.bias_corrected_util[layer_idx] = self.mag[layer_idx]/bias_correction


        self.h_t_1[layer_idx] = features
'''
    def update_utility_old(self, layer_idx=0, features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            current_layer = self.net[layer_idx * 2]
            next_layer = self.net[layer_idx * 2 + 2]

            if isinstance(next_layer, Linear):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=0)
            elif isinstance(next_layer, Conv2d):
                output_wight_mag = next_layer.weight.data.abs().mean(dim=(0, 2, 3))

            self.mean_feature_act[layer_idx] *= self.decay_rate
            self.mean_abs_feature_act[layer_idx] *= self.decay_rate
            if isinstance(current_layer, Linear):
                input_wight_mag = current_layer.weight.data.abs().mean(dim=1)
                self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=0)
                self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=0)
            elif isinstance(current_layer, Conv2d):
                input_wight_mag = current_layer.weight.data.abs().mean(dim=(1, 2, 3))
                if isinstance(next_layer, Conv2d):
                    self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=(0, 2, 3))
                    self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=(0, 2, 3))
                else:
                    self.mean_feature_act[layer_idx] += (1 - self.decay_rate) * features.mean(dim=0).view(-1,
                                                                                                          self.num_last_filter_outputs).mean(
                        dim=1)
                    self.mean_abs_feature_act[layer_idx] += (1 - self.decay_rate) * features.abs().mean(dim=0).view(-1,
                                                                                                                    self.num_last_filter_outputs).mean(
                        dim=1)

            bias_corrected_act = self.mean_feature_act[layer_idx] / bias_correction

            if self.util_type == 'adaptation':
                new_util = 1 / input_wight_mag
            elif self.util_type in ['contribution', 'zero_contribution', 'adaptable_contribution']:
                if self.util_type == 'contribution':
                    bias_corrected_act = 0
                else:
                    if isinstance(current_layer, Conv2d):
                        if isinstance(next_layer, Conv2d):
                            bias_corrected_act = bias_corrected_act.view(1, -1, 1, 1)
                        else:
                            bias_corrected_act = bias_corrected_act.repeat_interleave(
                                self.num_last_filter_outputs).view(1, -1)
                if isinstance(next_layer, Linear):
                    if isinstance(current_layer, Linear):
                        new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)
                    elif isinstance(current_layer, Conv2d):
                        new_util = (output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)).view(-1,
                                                                                                               self.num_last_filter_outputs).mean(
                            dim=1)
                elif isinstance(next_layer, Conv2d):
                    new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=(0, 2, 3))
                if self.util_type == 'adaptable_contribution':
                    new_util = new_util / input_wight_mag

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = rand(self.util[layer_idx].shape)
            else:
                self.util[layer_idx] += (1 - self.decay_rate) * new_util
                # correct the bias in the utility computation
                self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction
'''