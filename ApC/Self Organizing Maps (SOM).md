### Biological Background

> [!note]
> Certain single neural cells in the brain respond selectively to tom specific sensory stimuli.  These cells often perform local assemblies, in which their [[topological]] location corresponds to some specific feature value of a specific stimulus in an orderly fashion

### Key ideas
* @ Self-Organizing maps:
	* No supervision is required
	* SOMs learn on their own through unsupervised learning
* @ Maps
	* SOMs attempt to map their weights  to conform to the given input data. 
	* The nodes in a SOM network attempt to become like the inputs presented to them.  In this sense, this is how they learn
* @ Feature Maps:
	* Retain principle features of the input data
	* Topological relationships between input data are preserved when mapped to a SOM network

> [!note]
> More similar data will be associated with nodes that are closer in the network, whereas less similar data input will be mapped gradually farther away in the network
> ![[SOM_Draw.excalidraw | center]]
> 

### Learning process

#### Competition
When a *pattern* from the training set is loaded in the input layer, each unit in the competitive layer compute a discriminant function (usually a metric distance) compared its stored pattern/weight with the input.
>[!Basic functions]
$$d_{k}(t) = ||x(t) - w_{k}(t)||$$
$$BMU = arg_{k}\space min \space d_{k}(t)$$

The minimum discriminant function is computed to get the unit with the most similar weight vector: the *Best Matching Unit*, the winner of the competition.
#### Cooperation
The *BMU* determines the spacial location of a topological neighborhood of "excited units", thereby providing the basis for cooperation among such neighboring units.
The neighborhood will depend on the type of lattice, the neighborhood radius and the neighborhood function.
#### Adaptation
- @ Units increase their resemblance to input pattern throught a suitable adjustment on their weights 
	- It decreases with lateral distance to BMU
- ! The unit whos weight is closes to the current input becomes the winning or active unit or *Best Matching Unit*
- ! During the training stage, the weights of the winning unit are adjusted (*more*) as well as its neighbors(*less*), in an attempt to preser neighborhood relationships that exist withing the input data set
The adaptation process should be decomposed into two phases
- @ Ordering
	- During this phase the topological ordering of the weight vectors takes place
	- Relatively short, in comparison to the second phase
	- *Initially*, large values of neighborhood radius $\sigma (t)$ and learning rate $\alpha (t)$ should be used, such that the neuron's weights initially take large steps all together toward the are of input space where input vectors are ocurring
	- The values then should decrease to their tunning values, and consequently the neighborhood decreases to encompass only the closest neighbors
- @ Convergence (fine-tunning phase)
	- It last the rest of the adaptation process, it usually is several times longer than the ordering phase
	- It is necessary to fiene tune the network and therefore provide an accurate statistical quantification of the input space
	- The neighborhood $\sigma (t)$ should be fairly *small*, encompassing only the inmediate neighbors (or only the winner)
	- The learning rate $\alpha (t)$ should be also fairly *small*, so that the magnitude of the weight updates is also *small*
>[!info]
 The network must be exposed to a sufficient number of input patterns to ensure the self-organizing process reaches a stable state. 
 Usually the input pattern are recicled *(epochs)*
The neighborhood of the BMU decreases along the learning process up to just the BMU 
-> *Ordering phase*: unit's weights spread over the input space 
-> *Convergence phase*: weights fits to inputs in a small area

### Learning
1. Weights are initialized randomly
$$W_{i}=[w_{i1}, w_{2i}, \dots, w_{in}]$$
2. The only neuron that is activated is the closes to the input vector $$X = [x_{1}, x_{2}, \dots, x_{n}]$$
3. Only active neuron weights are modified, and its neighbourhood $$d_{k}(t) = ||x(t)-w_{k}(t)$$$$d_{c}(t) = min \space d_{k} \rightarrow c = BMU = arg_{k} \space min \space d_{k}(t)$$
4. Weights are moved to the input vector $$\Delta w_{ij} = \alpha (x_{j} - w_{ij})$$ $$w_{ij}(t) = w_{ij}(t-1) + \Delta w_{ij}$$
5. The neighborhood define which neurons have to learn when BMU is activated
	1. Linear, Square, Hexagonal, etc.
6. The weight update can be done via:
	1. Gaussian function $$h_{ck}(t) = e^{\frac{-||r_{k} - r_{c}||^2}{\sigma(t)^2}}$$
	2. Bubble/Top hat $$\begin{equation}
h_{ck}(t) = \left\{ \begin{array}{l} 
1 & if \space ||r_{k} - r_{c}||^{2} < \sigma(t) \\
0 & otherwise
\end{array} \right\}
\end{equation}$$
7. $\Omega(t)$: The *topological neighborhood* width decreases monotonically over time from a value no less than half the largest diagonal of the grid map to a value that encompass the immediate neighbor (radius = 1)
8. ! Learning rate is decreased with iterations $$\alpha_{t} = \alpha_{o}\left( 1 - \frac{1}{T} \right)$$
9. Neighborhood is modified in a similar way $$\Omega_{t} = \Omega_{o}\left( 1 - \frac{t}{T} \right) $$
### Conclusion
* @ Advantages
	* Non-supervised training
	* No precise pairs of input / output, only input patterns
	* Simply organizes itself autonomously to best suit the data used in training
* ! Disadvantages
	* Only provide information about what area of the input space belongs a certain pattern
	* We must interpret this information
	* For classification purposes, new data whose classification is known are needed


```octave
% Load the dataset
% Assume customerData is an NxM matrix, where N is the number of customers,
% and M is the number of features.
load customerData.mat % Replace with your data loading mechanism

% Normalize the data
customerData = normalize(customerData);

% Create a Self-Organizing Map (SOM)
dimensions = [10 10]; % Define the map dimensions
som_net = selforgmap(dimensions);

% Train the SOM
som_net = train(som_net, customerData');

% Visualize the SOM
figure;
plotsomtop(som_net); % Topology of the SOM
title('SOM Topology');

figure;
plotsomnc(som_net); % Neighbor connections
title('SOM Neighbor Connections');

figure;
plotsompos(som_net, customerData'); % Data point positions on the SOM
title('Customer Data Points on SOM');

% Cluster the data
som_output = som_net(customerData');
[~, clusterIndices] = max(som_output, [], 1);

% Visualize clusters
figure;
gscatter(customerData(:, 1), customerData(:, 2), clusterIndices);
title('Customer Segments');
xlabel('Feature 1 (e.g., Annual Income)');
ylabel('Feature 2 (e.g., Spending Score)');

% Display the clustering result
disp('Cluster indices for each customer:');
disp(clusterIndices);
```



