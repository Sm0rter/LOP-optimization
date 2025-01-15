# Loss of Plasticity in Deep Continual Learning
# Acknowledgements
Thanks to Shibhansh et al. for the original code that served as a foundation to this study


## Contents

- [Overview](#overview)
- [Repository Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](./LICENSE)
- [Citation](./citation.bib)


## Repository Contents
- [lop/algos](./lop/algos): All the algorithms used, including the continual backpropagation algorithm (CHANGED).
- [lop/nets](./lop/nets): The network architectures used in the paper (CHANGED).
- [lop/imagenet](./lop/imagenet): Demonstration and mitigation of loss of plasticity in a task-incremental problem using ImageNet (CHANGED).
- [lop/incremental_cifar](./lop/incremental_cifar): Demonstration and mitigation of loss of plasticity in a class-incremental problem (UNCHANGED.
- [lop/slowly_changing_regression](./lop/slowly_changing_regression): A small problem for quick demonstration of loss of plasticity (UNCHANGED).
- [lop/rl](./lop/rl): Loss of plasticity in standard reinforcement learning problems using the PPO algorithm (UNCHANGED).

Changes to specific subdirectories can be found in the README's of said subdirectories.


## Installation Guide

Create a virtual environment
```sh
mkdir ~/envs
virtualenv --no-download --python=/usr/bin/python3.8 ~/envs/lop
source ~/envs/lop/bin/activate
pip3 install --no-index --upgrade pip
```

Download the repository and install the requirements
```sh
git clone https://github.com/shibhansh/loss-of-plasticity.git
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add this lines in your `~/.zshrc` or `~/.bashrc`
```sh
source ~/envs/lop/bin/activate
```

Installation on a normal laptop with good internet connection should only take a few minutes
