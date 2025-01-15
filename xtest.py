import subprocess
import sys
import json
import copy
import argparse
import subprocess
from tqdm import tqdm
import os
import sys

bash_command = "mkdir loss_of_plasticity_main/temp_cfg/"
print("Running command:", bash_command)
subprocess.Popen(bash_command, stdout=subprocess.PIPE, shell=True)
