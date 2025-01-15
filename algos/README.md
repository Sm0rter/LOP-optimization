# Implementation of Continual Backpropagation
# Acknowledgement
Thanks to Shibhansh et al. for the original code that served as a foundation to this study

# Key Contributions
Windows compatibility: Adjustments made to ensure programs can run on windows.

New Features: 

`nnetmon.py`: tracker; can be injected into code to take account of various metrics during training 

`bp.py`: implemented Neural Net trackers within the code and added compatibility to Windows and CPU usage.

`convGnT.py`: some features moved to Nnetmon to centralize tracking of Neural Net Health.

`convCBP.py`: Added save

# Key Features

Original Implementations
Continual backpropagation for feed-forward networks (`cbp.py`), convolutional neural networks (`convCBP.py`), and residual networks (`res_gnt.py`)


`nnetmon.py`: Tracks neural network health metrics over time. Gives insights into anomalies and performance hindrances.


# Setup and usage
This directory contains different implementations of continual backpropagation. The results in the paper for feed-forward, convolutional, and residual networks in the paper are generated using `cbp.py,` 
`convCBP.py,` `res_gnt.py` respectively. 

`cbp_linear.py` and `cbp_conv.py` contain a newer and easier-to-use implementation of continual backpropagation.
This implementation allows you to use continual backpropagation like a layer in a network (similar to dropout or batch norm).

To use CBP as a layer, define a CBP layer in the network and make sure that activation passes through the CBP layer during the forward pass. See [../nets/conv_net2.py](../nets/conv_net2.py) for an example.

