>[!info]
>[Webpage](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.59788&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) to try neural networks

The central idea behind this algorithm is that the errors for the units of the hidden layer are determined by back propagating the error of the output layer.

![[Backprop_1.excalidraw|center]]
Can be considered as a generalization of the [delta rule](https://en.wikipedia.org/wiki/Delta_rule) for non-linear activation functions and multi-layer networks.

#### Phases
1. Propagation phase
	1. An input pattern is applied to the first layer of neurons in the network
	2. It is propagated by other upper layers until generate an output
	3. The actual network output is compared with the desired output value, we usually end up with an error in each of the output units
2. Adaptation phase
	1. Distribute the error of an output unit to all the hidden units that is it connected to, weighted by this connection
	2. This phase involves a backward pass through the network during which the error signal is passed to each unit in the network and appropriate weight changes are calculated.
#### General algorithm
* Initialize weights randomly $N(0, \sigma^2)$
* Loop until convergency
	* Compute gradient $\frac{dE(w)}{d(w)}$
	* Update weights $w = w - \mu \frac{dE(w)}{d(w)}$
* Return weights
Let's define total error as $$E = \sum_{i=1}^ne_{1}^2=\sum_{i=1}^n(t_{i}-o_{i})^2$$
* The goal is to minimize this error
* There not exist an analytical solution, the it is used an iterative algorithm
#### Back-propagation
* The Backpropagation algorithm is an iterative algorithm to train neural networks
* The backpropagation algorithm is based on the gradient method, where the weights are modified by the following iterative process $$w_{ij} = w_{ij}-\mu \frac{\Delta E}{\Delta w_{ij}}$$
Lets define:
* $net_{j}^{h}=$ Net inflow to $j$ unit
* $net_{k}^s=$ Net inflow to $k$ unit
* $w_{ij}=$   Weight from $i$ to $j$ unit
The input to the different units will be:
$$net_{j}^{h}=\sum_{i} x_{i}w_{ij}$$
$$net_{k}^s=\sum_{j}x_{j}w_{jk}$$
and the output
$o_{j} = F(net_{j}^h)$            $o_{k} = F(net_{k}^s)$
$$\frac{\Delta E}{\Delta w_{jk}}=\frac{\Delta E}{\Delta net_{k}^s}\frac{\Delta net_{k}^s}{\Delta w_{jk}}=\nabla_{k}x_{j}$$
Then $$\nabla_{k} = \frac{\Delta E}{\Delta net_{k}^{s}}=\frac{\Delta E}{\Delta e_{k}}\frac{\Delta e_{k}}{\Delta o_{k}}\frac{\Delta o_{k}}{\Delta net_{k}^{s}}=e_{k}(-1)F'(net_{k}^s)$$
The weights will be updated as: $$w_{jk} = w_{jk}-\mu\frac{\Delta E}{\Delta w_{jk}}=w_{jk}-\mu \nabla_{k}x_{j}$$
$$w_{jk}=w_{jk}+\mu(t_{k}-o_{k})F'(net_{k}^s)x_{j}$$
For the *hidden units*
$$\frac{\Delta E}{\Delta w_{ij}}=\frac{\Delta E}{\Delta net_{j}^h}\frac{\Delta net_{j}^h}{\Delta w_{ij}}=\nabla _{j}x_{j}$$
$$\nabla _{j}=\sum_{k}\left( \frac{\Delta E}{\Delta net_{k}^s}\frac{\Delta net_{k}^s}{\Delta o_{j}} \right)\frac{\Delta o_{j}}{\Delta net_{j}^h}=\sum_{k}(\nabla_{k}w_{jk})F'(net_{j}^h)$$
The the weights of the *hidden units* will be updated as $$w_{ij}=w_{ij}-\mu\frac{\Delta E}{\Delta w_{ij}=w_{ij}-\mu \nabla_{j}x_{i}}$$
$$w_{ij}=w_{ij}-\mu \nabla_{j}x_{i}$$
* The algorithm requires to derive the activation function
* It will be used, therefore, continuous activation functions and whose derivative exists and is easy to calculate $$F(x)=\tanh(x) \space \space F'(x)=1-F(x@)$$ $$F(x)=\frac{1}{1+\exp(-x)} \space \space F'(x)=F(x)(1-F(x))$$
##### Main features
* The algorithm searches the minimum of the error function from a set of training patterns
* Requires to derive the activation function
* Training consists of modifying weights of the network
* The weights are changed to the downward direction of the error function
* The trained network is able to generalize, correctly classifying noisy or incomplete patterns
>[!info]
>The algorithm makes the system evolve in the direction of the line of maximum slope, but the maximum slope is no always the most direct route to the minimum of the error function. If the starting point is different, the minimum value obtained is also different
>- It an converge into a local minimum
>- If the slope is reduced, the training is very slow
>- If the slope is zero, the algorithm is stopped
>- Difficulty in choosing the network architecture
>- The algorithm requires that modifications to the weights of the connections are infinitesimal. For practical purposes, finite values are sufficient for convergence

* ! Learning rate
	* Indicates how changes the weights
	* If the alpha value is very small, the learning speed is very slow. In case it is very large, it can show oscillatory behaviour
* ! Training a NN is difficult
	* Try lots of different learning rates and see how it work ```(traingd)```
	* Design an adaptive learning rate which adapts to the problem ```(traingda)```
	* Try different initialization of the weights to avoid falling into local minima
	* Prevent over-fitting
		* Early stopping
		* Regularization
	* Determine the appropriate complexity
#### Example
```octave
clear all, close all 

NUMPOINTS=1000; 

x = 4*(rand(1,NUMPOINTS)-0.5); 
yok=1.8*tanh(3.2*x+0.8)-2.5*tanh(2.1*x+1.2)- 0.2*tanh(0.1*x - 0.5);

RUIDO = 0.2*std(yok); 
y = yok + RUIDO*randn(size(yok)); 

NUMDATA=50; 

xtrain = 4*(rand(1,NUMDATA)-0.5); 
ytrain = 1.8*tanh(3.2*xtrain + 0.8)- 2.5*tanh(2.1*xtrain + 1.2)- 0.2*tanh(0.1*xtrain - 0.5) + RUIDO*randn(size(xtrain)); plot(xtrain,ytrain,'o')

net = newff(minmax(xtrain),[2 1],{'tansig' 'purelin'},'traingd'); 
% traingd traingdm traingda trainlm 
% net = fitnet([5],'traingd'); 
% net = configure(net,xtrain,ytrain); 
%Inicializa los pesos 
% net.layers{1}.transferFcn = 'tansig'; 
% net.layers{2}.transferFcn = 'purelin'; 
net.trainParam.epochs = 1000; 
net.trainParam.goal = 0.01; 
net = train(net,xtrain,ytrain); 
y2 = sim(net,xtrain); 
plot(xtrain,ytrain,'o'), hold on, plot(xtrain,y2,'xr') y2 = sim(net,x); figure, plot(xtrain,ytrain,'or'), hold on, plot(x,yok,'.k'), plot(x,y2,'xr')
```



