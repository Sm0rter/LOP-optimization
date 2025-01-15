import sys
import json
import copy
import argparse
import subprocess
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
print(os.getcwd())
from lop.utils.miscellaneous import get_configurations


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment",
                        type=str, default='../lop/imagenet/cfg/cbp.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, hyper_param_settings = get_configurations(params=params)

    # make a directory for temp cfg files
    bash_command = "mkdir temp_cfg"
    subprocess.run(bash_command, shell=True)
    print('made temp_cfg' )
#    os.rmdir(params['data_dir'])
    os.makedirs(params['data_dir'], exist_ok=True)
    print("made", params['data_dir'], 'folder')
    """
        Set and write all the parameters for the individual config files
    """
    for setting_index, param_setting in enumerate(hyper_param_settings):
        new_params = copy.deepcopy(params)
        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]
        new_params['index'] = setting_index
        new_params['data_dir'] = params['data_dir'] + str(setting_index) + '/'
        """
            Make the data directory
        """
        os.makedirs(new_params['data_dir'], exist_ok=True)
        print("made", new_params['data_dir'])

        for idx in tqdm(range(params['num_runs'])):
            new_params['data_file'] = new_params['data_dir'] + str(idx)
            new_params['run_idx'] = str(idx)
            """
                write data in config files
            """
            new_cfg_file = 'temp_cfg/'+str(setting_index*new_params['num_runs']+idx)+'.json'
            try:
                f = open(new_cfg_file, 'w+')
                print('opened')

            except: f = open(new_cfg_file, 'w+')
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f, sort_keys=False, indent=4)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
