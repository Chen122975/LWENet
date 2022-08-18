# LWENet
LWENet Pytorch
Here is the code for LWENet.
This work has been accepted by IEEE Signal Processing Letters
# Abstract
Abstract—In this letter, a lightweight and effective deep steaganalysis network (DSN) with less than 400,000 parameters, called LWENet, is proposed, which focuses on increasing the performance as well as significantly reducing the number of parameters (NP) from three perspectives. Firstly, in the pre-processing part, several lightweight bottleneck residual blocks are combined into the spatial rich model filters to improve the signal-to-noise ratio of stego signals while slightly increasing NP, thereby improving the subsequent performance. Secondly, a depthwise separable convolution layer is exploited at the end of the feature extraction part to largely reduce NP and increase the performance by capturing salient correlations while ignoring trivial ones among feature maps. Finally, to keep LWENet lightweight, we have to select only one fully connected (FC) layer. Simultaneously, multi-view global pooling is employed prior to the FC layer to yield multi-view features and further improve the detection performance. Extensive experiments demonstrate that our network achieves better performance than several state of-the-art DSNs.
# Train
Run  pre_train_pair_conv_net.py
