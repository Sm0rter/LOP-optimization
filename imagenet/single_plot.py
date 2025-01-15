import argparse
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lop.utils.miscellaneous import  *


'''
Given an input vector, partition it to segments of size win_sz
return the mean and std of each segment
'''

def movingStat(in_vec, win_sz=1):
    len_out = int(in_vec.shape[0]/win_sz)
    out_vec = torch.zeros(len_out, 2)
    for l in range(len_out):
        cur_seg = in_vec[l*win_sz: (l+1) * win_sz]
        out_vec[l,0] = cur_seg.mean()
        out_vec[l,1] = cur_seg.std()
    return out_vec


'''
Given input vector, partition into segments of size win_sz
return each segments median
'''
def movingMedian(in_vec, win_sz=1):
    len_out = int(in_vec.shape[0]/win_sz)
    out_vec = torch.zeros(len_out)
    for l in range(len_out):
        cur_seg = in_vec[l*win_sz: (l+1) * win_sz]
        out_vec[l] = torch.median(cur_seg)
    return out_vec

'''
accuracy data stored as acc_data[task_id, epoch_id]
because can run less than the max number of epochs, so need to pad accuracy matrix with the accuracy of last training epoch
'''

def pad_file(acc_data):
    for i in range(acc_data.shape[0]):
        converging_perf = 0
        for j in range(acc_data.shape[1]):
            if acc_data[i][j] < 0.00001:
                acc_data[i][j] = converging_perf
            else:
                converging_perf += acc_data[i][j]
    return acc_data

def plotAcc(perffile=None, total_task=1400, stat_win_sz=11, epoch_per_task=150, with_padding=False):
    print(perffile)
    with open(perffile, 'rb') as f:
        data = pickle.load(f)
    acc_data = data['test_accuracies']
    print(acc_data[:,-1].flatten())
    acc_data = acc_data[:total_task, :epoch_per_task]
    print(acc_data.flatten())

    if with_padding:
        acc_data = pad_file(acc_data)
        print((acc_data*100).flatten())
    acc_data = (acc_data).flatten()
    xdata = range(int(len(acc_data)/stat_win_sz))
    perf_stat = movingStat(acc_data, win_sz = stat_win_sz)

    colors = [(1, 0, 0, 1), (0.5, 0.5, 0, 1), (0, 1, 0, 1), (0, 0.5, 0.5, 0.5)]
    fig, ax = plt.subplots()
    fig.set_size_inches(16,9)

    plt.plot(xdata, perf_stat[:, 0], '-', label = 'model accuracy, window size of ' + str(stat_win_sz), color = colors[0])
    plt.fill_between(xdata, perf_stat[:, 0] - perf_stat[:, 1], perf_stat[:, 0]+ perf_stat[:, 1], color=colors[0], alpha=0.2)

    plt.show()

def plotNNHealth(logfile=None, perffile = None, movingstat = 0, partition = 1, epoch_per_task = 150):
    try:
        print(logfile)
        print(perffile)
        if logfile is not None:
            with open(logfile, 'rb') as f:
                data = pickle.load(f)
                len = data['total_history']
                epoch = data['epoch']
                erank = data['e_rank']
                dead_n = data['dead_n']
                f_val = data['feature_abs_val']
                mag_w = data['weight_magnitude']
                epoch = epoch[:len]
                print('total epochs', len)
                print('epoch', epoch)
                erank = erank[:len, :]
                print('erank', erank)
                dead_n = dead_n[:len, :]
                total_dead_n = dead_n.sum(dim=1)
                f_val = f_val[:len, :]
                mag_w = mag_w[:len, :]
            if perffile is not None:
                with open(perffile, 'rb') as f:
                    data = pickle.load(f)
                acc_data = data['test_accuracies']
                perf_data = acc_data.flatten()
                perf = perf_data[epoch-1]
                print('perf', perf)

                e_p = partition
                step = int(len/e_p)
                for i in range(e_p):
                    s_i = i*step
                    e_i = (i + 1) * step
                    t_s = int(epoch[s_i]/epoch_per_task)
                    t_e = int(epoch[e_i-1]/epoch_per_task)
                    plot_nn_health(epoch[s_i: e_i], erank[s_i: e_i, :], perf[s_i: e_i] * 100,
                                   save_to='nn_health_erank_w_perf_'+str(i)+'.png',
                                   caption = 'total rank trend and model accuracy trend ( ' + str(t_s) + ' to ' + str(t_e) + ' )')
                    plot_nn_health(epoch[s_i: e_i], dead_n[s_i: e_i, :], perf[s_i: e_i] * e_p,
                                   save_to='nn_health_dead_neuron_w_perf_' + str(i) + '.png',
                                   caption='total rank trend and model accuracy trend ( ' + str(t_s) + ' to ' + str(
                                       t_e) + ' )')
                    plot_nn_health(epoch[s_i: e_i], total_dead_n[s_i: e_i], perf[s_i: e_i] * 100,
                                   save_to='nn_health_total_dead_neuron_w_perf_' + str(i) + '.png',
                                   caption='total rank trend and model accuracy trend ( ' + str(t_s) + ' to ' + str(
                                       t_e) + ' )')
                    plot_nn_health(epoch[s_i: e_i], f_val[s_i: e_i, :], perf[s_i: e_i] * 10,
                                   save_to='nn_health_erank_w_perf_' + str(i) + '.png',
                                   caption='total rank trend and model accuracy trend ( ' + str(t_s) + ' to ' + str(
                                       t_e) + ' )')
                    plot_nn_health(epoch[s_i: e_i], mag_w[s_i: e_i, :], perf[s_i: e_i],
                                   save_to='nn_health_erank_w_perf_' + str(i) + '.png',
                                   caption='total rank trend and model accuracy trend ( ' + str(t_s) + ' to ' + str(
                                       t_e) + ' )')
            else:
                plot_nn_health(epoch, erank,
                               save_to='nn_health_erank.png', caption='rank_trend')
                plot_nn_health(epoch, dead_n,
                               save_to='nn_health_dead_n.png', caption='dead neuron counts')
                plot_nn_health(epoch, total_dead_n,
                               save_to='nn_health_total_dead_neuron.png', caption='total dead neuron counts')
                plot_nn_health(epoch, f_val,
                               save_to='nn_health_feature_weight.png', caption='feature mean abs value')
                plot_nn_health(epoch, mag_w,
                               save_to='nn_health_weight_mag.png', caption='weight magnitude')

    except FileNotFoundError as e:
        print('cannot find file <', e, '>')
        pass
    except OSError as e:
        print('An error has occurred <', e, '>')

def plot_nn_health(xdata, ydata, perf=None, stat_win = 0, save_to = None, caption = None, svg = False,
                   labels = ['layer 0', 'layer 1', 'layer 2', 'layer 3', 'model accuracy']):
    fig, ax = plt.subplots()
    fig.set_size_inches(16,8)
    colors = [(1, 0, 0, 1), (0.5, 0.5, 0, 1), (0, 1, 0, 1), (0, 0.5, 0.5, 0.5)]

    if stat_win > 0:
        xstats = movingMedian(xdata, stat_win)

    if ydata.dim() > 1:
        for i in range(ydata.shape[1]):
            if stat_win > 0:
                ystats = movingStat(ydata[:, i], win_sz = stat_win)
                plt.plot(xstats, ystats[:,0], '-', label=labels[i], color=colors[i])
                plt.fill_between(xstats, ystats[:,0]- ystats[:,1], ystats[:,0] + ystats[:,1], color=colors[i], alpha=0.2)
            else:
                plt.plot(xdata, ydata[:,i], '-', label=labels[i], color=colors[i])
    else:
        if stat_win > 0:
            ystats = movingStat(ydata, win_sz=stat_win)
            plt.plot(xstats, ystats[:,0], '-', label=labels[-1], color=colors[0])
            plt.fill_between(xstats, ystats[:, 0] - ystats[:, 1], ystats[:, 0] + ystats[:, 1], color=colors[0],
                             alpha=0.2)
        else:
            plt.plot(xdata, ydata, '-', label=labels[0], color=colors[0])

    if perf is not None:
        if stat_win > 0:
            perf_stats = movingStat(perf, win_sz=stat_win)
            plt.plot(xstats, perf_stats[:,0], '-', label='model accuracy', color=colors[-1])
            plt.fill_between(xstats, perf_stats[:, 0] - perf_stats[:, 1], perf_stats[:, 0] + perf_stats[:, 1], color=colors[-0],
                             alpha=0.2)
        else:
            plt.plot(xdata, perf, '-', label=labels[-1], color=colors[-1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if labels is not None:
        plt.legend()
    if caption is not None:
        ax.set_title(caption)
    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight', dpi=500)

    plt.show()


def showImageNetDataSet(plot_task_level_samples=True):
    with open('../lop/imagenet/class_order', 'rb+') as f:
        class_order = pickle.load(f)
        class_order = class_order[0]
    print(class_order.shape)
    class_order = np.concatenate([class_order]*2)
    idx = 0
    for task_id in range(50):
        x_train, y_train, x_test, y_test = [],[],[],[]
        classes = class_order[idx: idx+2]
        idx += 2

        for img_class_type, class_id in enumerate(classes):
            data_file = 'data/classes/' + str(classes[img_class_type]) + '.npy'
            new_x = np.load(data_file)
            new_x = (new_x+1)/2
            x_train.append(new_x[:600])
            x_test.append(new_x[600:])
            y_train.append(np.array([img_class_type]*600))
            y_test.append(np.array([img_class_type] * 100))
        x_train = torch.tensor(np.concatenate(x_train))
        x_test = torch.tensor(np.concatenate(x_test))
        y_train = torch.tensor(np.concatenate(y_train))
        y_test = torch.tensor(np.concatenate(y_test))

        example_order = np.random.permutation(600*2)
        x_train = x_train[example_order]
        y_train = y_train[example_order]

        if plot_task_level_samples:
            fig, axs = plt.subplots(3,4)
            fig.set_size_inches(12,8)
            img_idx = 0
            for ax_i in range(2):
                for ax_j in range(4):
                    axs[ax_i, ax_j].imshow(x_train[img_idx].T)
                    axs[ax_i, ax_j].title.set_text(int(y_train[img_idx]))
                    img_idx += 1
        img_idx = 0
        for ax_j in range(4):
            axs[2, ax_j].imshow(x_train[img_idx].T)
            axs[2, ax_j].title.set_text(int(y_train[img_idx]))
            img_idx += 1
        fig.subtitle('class <' + str(int(classes[0])) + ',' + str(int(classes[1])) + '>')
        plt.show()



def main(arguments):
    # -input_file data/hist/25_01_11_0_nbp_ep30_t5/0 -nn_summary data/hist/25_01_11_0_nbp_ep30_t5/nn_summary_2025-01-11_20_12
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', help="The input file containing experiment information",
                        type=str, default=None)
    parser.add_argument('-nn_summary', help="The log file containing nnet summary",
                        type=str, default=None)


    args = parser.parse_args(arguments)
    perf_file = args.input_file
    health_log = args.nn_summary

    plotAcc(perf_file, total_task = 5, stat_win_sz = 1)
    return
    if health_log is not None:
        plotNNHealth(logfile = health_log, perffile = None)
        sys.exit(0)





#    cfg_file = args.c
#    print(cfg_file)




if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
