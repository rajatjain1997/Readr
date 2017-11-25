# Readr

Readr is a library using which programmers can create and compare neural networks capable of supervised pattern recognition without knowledge of machine learning. These networks are fuzzy-neuro systems with fuzzy controllers and tuners regulating learning parameters after each epoch to achieve faster convergence.

## Learning Algorithms

The learning algorithms provided by Readr are:

1. **General Back Propagation**: The standard neural network with no fuzzy controllers acting on it. It converges with the slowest speed within all networks provided by Readr. Also used as a benchmark for performance of other neural networks.

2. **GBP with Momentum**: A neural network with GBP and momentum applied to it. It is also used as a benchmark for other neural networks provided by Readr. About 200% faster convergence than GBP.

3. **GBP with Fuzzy Gain**: The network proposed by the paper we implemented. It passes the gain of the activation function to a fuzzy controller in order to maximize convergence speed. Approximately 1000% faster than GBP.

4. **GBP with Fuzzy Momentum**: The network fuzzifies momentum in order to enable fasster convergence. This is slower than GBP with Fuzzy Gain, but faster than the unfuzzified momentum.

5. **GBP with Fuzzy Gain and Momentum**: This neural network fuzzifies gain and momentum both in order to maximize convergence speed.

*Note:* All networks currently use log-sigmoid as their activation funcations and have 3 layers. These parameters will be made tunable in future releases.

## Usage

To use Readr, import the library in your script as:

	import reader as rd

The following functions allow you to deploy networks:

### session() 

Returns a session object with a new neural network. This object is required for all operations on the network.

### train(sess, imagepath, actualresult,enableGainFuzzification= True)

Trains the neural network with the image provided.

Parameter    | Description
------------ | ---------------------------------------------
sess         | Session object for neural network
imagepath    | Path to the image to train with
actualresult | Desired output
enableGainFuzzification | Fuzzy control for gain

**Returns:** Convergence of the network. (~98%)

### read(sess, imagepath)

Reads the image and determines the pattern.

Parameter    | Description
------------ | ---------------------------------------------
sess         | Session object for neural network
imagepath    | Path to the image to get pattern from

**Returns:** Pattern Matrix from output layer.

### outputCharacter(sess, imagepath)

Reads the image and returns a numeric answer.

Parameter    | Description
------------ | ---------------------------------------------
sess         | Session object for neural network
imagepath    | Path to the image to get pattern from

**Returns:** Node number at which output == 1 from the output layer.

### provideMnistTraining(sess, numTrainingSamples, enableGainFuzzifization = True, enableEtaFuzzification = True, enableMomentumFuzzification=True, momentumConstant = 0.0)

Trains the network with MNIST dataset to up to 90% convergence. Current implementation depends on sensitivity, which is an internal variable. This tuning will be made automatic in future releases.

Parameter                         | Description
--------------------------------- | ---------------------------------------------------------------
sess                              | Session object for neural network
numTrainingSamples                | Number of training samples to train the pattern with
enableGainFuzzification           | Fuzzy control for gain
enableEtaFuzzification            | Fuzzy control for learning rate
enableMomentumFuzzification       | Fuzzy control for momentum (Does not work if momentum is disabled in the network)
momentumConstant                  | Enable or disable momentum in the network

**Returns:** Convergence of the network. (~90%).

### reset(sess)

Resets the neural network.

Parameter    | Description
------------ | ---------------------------------------------
sess         | Session object for neural network

### storeModel(session, filename)

Saves the model in a file on the system. Useful to store networks once training is complete.

Parameter    | Description
------------ | ---------------------------------------------
sesison      | Session object for neural network
filename     | Path where the network should be stored

### restoreModel(session, filename)

Loads the model from a file on the system. Used when tarined networks are to be loaded into memory.

Parameter    | Description
------------ | ---------------------------------------------
sesison      | Session object for neural network
filename     | Path where the network should be stored

## Licence

All Readr source code is made available under the terms of the GNU Affero Public License (GNU AGPLv3).
