# Loss of Plasticity in Continual ImageNet

# Acknowledgement
Thanks to Shibhansh et al. for the original code that served as a foundation to this study

# Key Contributions
1. Windows compatibility: Adjustments were made to ensure all code runs and works on Windows.
2. New Features: [`../single_plot.py`](../single_plot.py): Algorithm to better display data gotten from tracking the neural net during training & [`../show_nn_health.py`](../show_nn_health.py): New file to add tracking to the neural net during training.
3. Integration: Code changes were made to integrate the functionality of neural net health tracking into other files


# Setup and usage

This directory contains code implemented for the Imagenet Binary Classification problem. Additionally, functionality was added to monitor key details of the Neural Net, such as weight magnitude, feature value, etc.

To run the program first download the data. The data can be downloaded [here](https://drive.google.com/file/d/1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z/view?usp=sharing).
Create a directory named `data` and extract the downloaded file in `data`
```sh
cd lop/imagenet/
mkdir data
```

Changes were made to `../nets/conv_net.py` to shorten testing time, allowing for wider accessibility.
The network is specified in [`../nets/conv_net_small.py`](../nets/conv_net_small.py)

The following command produces 30 temporary cfg files in `temp_cfg`.

```sh
python3.8 multi_param_expr.py -c cfg/bp.json 
```

Each of the new temporary cfg files can then be used to do one run of backprop.
```sh
python3.8 expr.py -c temp_cfg/0.json 
```

# Overview of features
Original Functionality: 
Implementation of Imagenet binary classification problem
Methods to test deep convolutional neural networks with various training strategies
Tools for visualizing accuracy across training methods

Added functionality
[`../single_plot.py`](../single_plot.py): plots erank, dead neuron count, total dead neuron count, feature abs value, weight magnitude

[`../show_nn_health.py`](../show_nn_health.py): plots specific data points, such as erank, feature value, neuron value, weight magnitude ratio, weight magnitude growth rate, dead neuron count, removed neuron count (working on) utility (as define by Shibhansh) for each layer of the neural net



