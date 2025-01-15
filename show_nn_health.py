import sys
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import time

'''
'erank 0'
'erank 1'
'erank 2'
'feature 0'
'feature 1'
'feature 2'
'mag_ratio 0'
'mag_ratio 1'
'mag_ratio 2'
'value_prop 0'
'value_prop 1'
'value_prop 2'
'growth_rate 0'
'growth_rate 1'
'growth_rate 2'
'dead_neuron 0'
'dead_neuron 1'
'dead_neuron 2'
'utility 0'
'utility 1'
'utility 2'
'rm_neuron 0'
'rm_neuron 1'
'rm_neuron 2'

            '''

def show_nn_health(dir, s_tid = -1, e_tid = -1, attr='feature 0', down_sample=20):
    try:
        total_tasks = e_tid - s_tid - 1
        attr_val = torch.zeros([50000,1000])
        len_attr = 1000

        bp_ind = -1
        for tid in range(s_tid, e_tid):
            cur_f = dir+str(tid)
            if attr is None:
                print('to load', cur_f)
            try:
                print(cur_f)
                with open(cur_f, 'rb') as f:
                    data = pickle.load(f)
                bp_iter = -1
                for data_entry in data:
                    bp_iter += 1
                    bp_ind += 1
                    curr_attr = data_entry[attr]
                    len_attr = len(curr_attr)
                    attr_val[bp_ind, :len_attr] = curr_attr
            except FileNotFoundError as e:
                print('skip loading file <', cur_f, '>')
                pass
        show_data(data=attr_val[:bp_ind, :len_attr], down_sample=down_sample,
                  caption='tasks [' + str(s_tid) + ',' + str(e_tid) + '] (' + attr + ')')
    except OSError as e:
        print('An error has occurred:', e)

def show_data(data = None, down_sample=20, caption = None):
    print('to show data', data.shape)
    fig, ax = plt.subplots()
    if down_sample > 1:
        data = movingMean2d(data, win_sz=down_sample)
    im = ax.imshow(torch.transpose(data,0,1))
    time.sleep(1)
    fig.colorbar(im, ax=ax, label='Interactive Colorbar')
    if caption is not None:
        ax.set_title(caption)
    plt.show()

def movingMean2d(in_vec, win_sz=1):
    len_out = int(in_vec.shape[0] / win_sz)
    out_vec = torch.zeros(len_out, in_vec.shape[1])
    for j in range(len_out):
        cur_seg = in_vec[j*win_sz:(j+1)*win_sz, :]
        mean_val = torch.mean(cur_seg, dim=0)
        out_vec[j, :] = mean_val
    return out_vec

def main(arguments):
    #-in_dir data/hist/25_01_11_0_nbp_ep30_t5/nnh -downsample 24 -s 0 -e 5
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-in_dir', help="Input file containing information",
                        type=str, default='data/nn_health/')
    parser.add_argument('-s', help="Start ID of task to show",
                        type=int, default=0)
    parser.add_argument('-e', help="End ID of task to show",
                        type=int, default=1)
    parser.add_argument('-attr', help="Which attribute to show",
                        type=str, default='feature 0')
    parser.add_argument('-downsample', help="Many data points -> 1",
                        type=int, default=12)

    args = parser.parse_args(arguments)
    nnh_dir = args.in_dir
    print('nnh dir:', nnh_dir)




    show_nn_health(dir=nnh_dir, s_tid=args.s, e_tid=args.e, attr=args.attr, down_sample=args.downsample)




if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))




