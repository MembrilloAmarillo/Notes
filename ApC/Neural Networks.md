## Introduction
![[Neuron.excalidraw|center]]
>[!Weights]
Usually a neuron receives multiple simultaneous entries. Each entry has its own relative weight which gives the importance to the entry inside the neuron's aggregation function.
Weights are coefficients that can adapt inside the net which determine the intensity of the input signal registered by the artificial neuron. 

>[!Propagation Function]
>This rule allows to obtain, given the inputs and the weights, the potential postsynaptic value $h_{i}$ of the neuron: $$h_{i}(t) = \sigma_{i}(w_{ij}, x_{j})$$
>The most habitual function is the weighted sum of all entries. We can group the entries and weights in two vectors, and down the scalar product of each vector:$$h_{i}(t)=\sum_{j}w_{ij}x_{j}$$

>[!Activation/Transfer function]
>The result of the propagation function (in the majority of cases is a weighted sum) is transformed into the neurons real output by an algorithmic process known as activation function: $$a_{i}(t) = f_{i}(a_{i}(t-1), h_{i}(t))$$
>In this case the activation function depends on the postsynaptic  potential $h_{i}(t)$ and of previous activation function itself. However, in many models the NN it is considered that the current state of the neuron does not depend on its previous state $a_{i}(t-1)$, but only from the current one: $$a_{i}(t) = f_{i}(h_{i}(t))$$
>



