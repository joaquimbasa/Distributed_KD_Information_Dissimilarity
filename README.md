# Distributed_KD_Information_Dissimilarity
This repo is all about the code used for the experiment of our paper titled Information Dissimilarity Measures in Decentralized Knowledge Distillation: A Comparative Analysis. 


The provided code implements a PyTorch-based framework for training a DL model with knowledge distillation, specifically designed to perform distillation using multiple distance metrics. It leverages a custom ResNet model architecture with two output heads, designed to support Distributed Knowledge Distillation scenarios where multiple remote clients participate.

Key Components:

1. Model Architecture: A custom ResNet model (CustomResNet) is defined with two heads for multitask learning. The base model is a ResNet18, where the final fully connected (FC) layer is replaced with two separate heads: one for direct classification and the other for incorporating knowledge from remote clients.
2. Distillation Losses: Several custom distillation loss functions are implemented, including Jensen-Shannon Divergence, Triangular Distance, Structural Entropic Distance (SED), and KL divergence and Cross Entropy. These functions measure the dissimilarity between the soft predictions (probability distributions) of the student and teacher models to transfer knowledge.
3. Training Process: The training is split into two phases:

First Head Training: The custom model is trained starting from a pre-trained model on an imagenet dataset. 
Second Head Training: After loading the trained weights, the second head is trained using knowledge distillation with different distillation dissimilarity metrics, temperatures, and alpha blending coefficients between the student loss and distillation loss.

Note: The model of each client is trained using the client's private dataset, and the value corresponding to the number of clients has to be adjusted from variables currentClient and remoteClients at lines 129 and 130 of the code.   
