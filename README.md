# Fashion-MNIST-Image-Classification

## Google Collab Link Here: https://colab.research.google.com/drive/1KBjNR5Zajv-jh9Ltd5p_sIIuadWe9eQ8?usp=sharing

1. What is the Fashion MNIST dataset?
 
= Fashion MNIST dataset is a collecton of images designed as a modern alternative to the classic MNIST dataset for benchmarking machine learning models particularly in image classification tasks.

2. Why do we normalize image pixel values before training?

= Normalizing image pixel improves neural network training bby stablizing gradient, accelerating convergence andenhancing generalization by reducing sensitivity to varying intensities.

3. List the layers used in the neural network and their functions.

= Input Layer: Accepts the raw image data (28x28 pixel arrays) and passes it to subsequent   layers without transformation.
Convolutional Layer (Conv2D): Applies filters to detect features like edges, textures, or patterns by sliding kernels over the input, producing feature maps that highlight spatial hierarchies.

= Activation Layer (ReLU): Introduces non-linearity by applying functions like ReLU (max(0, x)) to outputs, enabling the network to learn complex patterns and avoid vanishing gradients.

= Pooling Layer (MaxPooling2D): Reduces spatial dimensions of feature maps by downsampling (taking the max value in a window), decreasing computational load and improving translation invariance.

= Flatten Layer: Converts multi-dimensional feature maps into a 1D vector, preparing data for fully connected layers.

= Fully Connected (Dense) Layer: Connects every neuron from the previous layer to the next, performing classification by learning global patterns and relationships.

= Output Layer (Softmax): Produces probability distributions over classes (10 fashion categories in Fashion MNIST) for multi-class prediction.

4. What does an epoch mean in model training?

= Epoch mean in model training is one complete pass through the entire training dataset during which the model processes all samples (usually in batches) to update its parameters via optimization algorithms likegradient descent 

5. Compare the predicted label and actual label for the first test image.

= Predicted Label: 9 (Ankle boot) – The model's softmax output indicates the highest probability   for class 9.
= Actual Label: 9 (Ankle boot) – Ground-truth from the dataset.
= Match: Yes – The prediction is correct.
= Confidence: The predicted probability for class 9 is approximately 0.

6. What could be done to improve the model’s accuracy?
= To improve model accuracy incorporate data augmentation, add ropout an batch normalization for regularization, increase convolutional layers or filters, tune hyperparameters like learning rate and epochs and consider transfer learning from pre-trained models.
