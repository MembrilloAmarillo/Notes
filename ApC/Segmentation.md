### Fully convolution network (FCN)
>A FCN is a neural network that only performs convolution (subsampling (encoder) or upsampling (decoder)) operations. Equivalently, a FCN is a CNN without fully connected layers.

The encoder extracts features from the image through filters. The decoder is responsible for generating the final output which is usually a segmentation mask containing the outline of the object. 

The **decoder (upsampler)** access the low-lever features produced by the encoders **(pooling)** layers. Because the encoder reduces the image resolution, the segmentation lacks well-defined edges, meaning that the boundaries between the images are not clearly defined.

The is accomplished by skip connections. ==**Skip connections** bypass layers and transfer the information intact to the next layers. They are used to pass the information from early layer of the encoder to the decoder, bypassing the downsampling layers.== And indeed, this helped improve the details of the segmentation with much more accurate shapes and edges.
By using a skip connection, we provide an alternative path for the gradient (with backpropagation). Is is experimentally validated that this addiation paths are often beneficial for the model convergence. 
> Skip connections skip some layer in the neural network and feeds the output of one layer as the input to the next layers (insetead of only the next one)

## Image segmentation

Is one of the fundamentals tasks in computer vision alongside with object recognition and detection. In semantic segmentation, the goal is to classify each pixel of the image in a specific category. The difference from image classification is that we do no classify the whole image in on class but each individual pixel. So we have a set of predefined categories and we want to assign a label in each pixel of the image. And we do this assignment based on the context of the different objects in the image.
#### U-Nets: long skip connections
This kind of network is used for tasks that the prediction has the same spatial dimension as the input such as [[https://theaisummer.com/Semantic_Segmentation/]], optical flow estimation, video prediction, etc.
![[Pasted image 20240813181515.png]]
This would be the encoding part, we could implement the simmetric encoder just changing the max pooling for up-convolution 2x2 ( using padding ) and for the final result using a conv 1x1 and reducing the filters to the desired output.

| Layers      | Dimensions      | Operation       | Filters |
| ----------- | --------------- | --------------- | ------- |
| Input Layer | 254 x 254       |                 | 1       |
| Conv1       | 254 x 254       | Conv 3x3        | 64      |
| Rel1        | 252 x 252 x 64  | Relu            | 64      |
| Conv2       | 252 x 252 x 64  | Conv 3x3        | 64      |
| Rel2        | 250 x 250 x 64  | Relu            | 64      |
| Conv3       | 250 x 250 x 64  | Conv 3x3        | 64      |
| Rel3        | 248 x 248 x 64  | Relu            | 64      |
| MaxPool1    | 248 x 248 x 64  | Max Pooling 2x2 | 64      |
| Conv4       | 124 x 124 x 128 | Conv 3x3        | 128     |
| Rel4        | 122 x 122 x 128 | Relu            | 128     |
| Conv5       | 122 x 122 x 128 | Conv 3x3        | 128     |
| Rel5        | 120 x 120 x 128 | Relu            | 128     |
| MaxPool2    | 120 x 120 x 128 | Max Pooling 2x2 | 128     |
| Conv6       | 60 x 60 x 128   | Conv 3 x 3      | 512     |
| Rel6        | 58 x 58 x 512   | Relu            | 512     |
| Conv7       | 58 x 58 x 128   | Conv 3 x 3      | 512     |
| Rel7        | 56 x 56 x 512   | Relu            | 512     |
| MaxPool2    | 56 x 56 x 512   | Max Pooling 2x2 | 512     |
| Conv8       | 38 x 38 x 1024  | Conv 3 x 3      | 1024    |
| Rel8        | 38 x 38 x 1024  | Relu            | 1024    |
| Conv9       | 36 x 36 x 1024  | Conv 3 x 3      | 1024    |
| Rel9        | 36 x 36 x 1024  | Relu            | 1024    |
| Conv10      | 34 x 34 x 1024  | Conv 3 x 3      | 1024    |
| Rel10       | 34 x 34 x 1024  | Relu            | 1024    |
| Conv11      | 32 x 32 x 1024  | Conv 3 x 3      | 1024    |
| Rel11       | 32 x 32 x 1024  | Relu            | 1024    |
