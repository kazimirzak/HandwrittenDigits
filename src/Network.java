import Math.Functions;

import java.util.Random;

/**
 * @author Kenny Brink - kebri18@student.sdu.dk
 */

public class Network {

    private final double[][][] weights;
    private double[][][] totalWeightError;
    private final double[][] bias, activation, weightedInput;
    private double[][] totalBiasError;

    /**
     * layout = [inputSize, numOfNeurons, numOfNeuron, ... , outputSize];
     *                      layer 0       layer 1              last layer
     */

    Network(int[] layout) {
        int layers = layout.length - 1;
        Random randomizer = new Random();
        weights = new double[layers][][];
        bias = new double[layers][];
        for (int i = 0; i < layers; i++) {
            weights[i] = initWeights(layout[i], layout[i + 1], randomizer);
            bias[i] = initBias(layout[i + 1], randomizer);
        }
        activation = new double[layers][];
        weightedInput = new double[layers][];
        initError();
    }

    /**
     * Initializes the weights of this network.
     */

    private double[][] initWeights(int numOfWeights, int numOfNeurons, Random randomizer) {
        double[][] result = new double[numOfNeurons][numOfWeights];
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                result[i][j] = randomizer.nextGaussian();
            }
        }
        return result;
    }

    /**
     * Initializes the biases of this network.
     */

    private double[] initBias(int numOfNeurons, Random randomizer) {
        double[] result = new double[numOfNeurons];
        for(int i = 0; i < result.length; i++) {
            result[i] = randomizer.nextGaussian();
        }
        return result;
    }

    /**
     * Initializes the error arrays of this network.
     */

    private void initError() {
        initBiasError();
        initWeightError();
    }

    /**
     * Initializes the totalBiasError of this network.
     */

    private void initBiasError() {
        totalBiasError = new double[bias.length][];
        for(int i = 0; i < totalBiasError.length; i++) {
            double[] temp = new double[bias[i].length];
            totalBiasError[i] = temp;
        }
    }

    /**
     * Initializes the totalWeightError of this network.
     */

    private void initWeightError() {
        totalWeightError = new double[weights.length][][];
        for(int i = 0; i < totalBiasError.length; i++) {
            double[][] temp = new double[weights[i].length][];
            for(int j = 0; j <weights[i].length; j++) {
                double[] aux = new double[weights[i][j].length];
                temp[j] = aux;
            }
            totalWeightError[i] = temp;
        }
    }

    /**
     * Runs training on this network and prints a result of the testing.
     * @param trainingImages Training images to train on.
     * @param trainingLabels Training labels to train on.
     * @param testImages Testing images to test on.
     * @param testLabels Testing labels to test on.
     * @param batchSize The batch size of each batch in each epoch.
     * @param learningRate The learning rate for the training.
     * @param epochs How many epochs the network should train for.
     */

    void training(double[][] trainingImages, int[] trainingLabels, double[][] testImages, int[] testLabels, int batchSize, double learningRate, int epochs) {
        int totalBatches = trainingImages.length / batchSize;
        for(int i = 0; i < epochs; i++) {
            int currentImage = 0;
            for(int j = 0; j < totalBatches; j++) {
                for(int k = 0; k < batchSize; k++) {
                    double[] output = feedForward(trainingImages[currentImage], 0);
                    backPropagate(output, trainingImages[currentImage], trainingLabels[currentImage]);
                    currentImage++;
                }
                doGradientDescent(learningRate, batchSize);
            }
            doTest(testImages, testLabels, i);
            shuffle(trainingImages, trainingLabels);
            shuffle(testImages, testLabels);
        }
    }

    /**
     * Runs stochastic gradient descent on this network.
     * @param learningRate the learning given at training().
     * @param batchSize the batch size given at training().
     */

    private void doGradientDescent(double learningRate, int batchSize) {
        int layers = totalBiasError.length;
        double gradient = learningRate / (double) batchSize;
        for(int i = 0; i < layers; i++) {
            weights[i] = Matrix.subtract(weights[i], Matrix.constantMultiplication(totalWeightError[i], gradient));
            bias[i] = Matrix.subtract(bias[i], Matrix.constantMultiplication(totalBiasError[i], gradient));
        }
        initError();
    }

    /**
     * Runs a test on this network and prints the amount of correct images.
     * @param images The images to test on.
     * @param labels The labels to test on.
     * @param epoch The current epoch.
     */

    private void doTest(double[][] images, int[] labels, int epoch) {
        int correctImages = 0;
        for(int i = 0; i < images.length; i++) {
            double[] output = feedForward(images[i], 0);
            if(isCorrect(output, labels[i]))
                correctImages++;
        }
        System.out.println("Epoch: " + epoch + " Correct Images: " + correctImages + "/" + images.length);
    }

    /**
     * Shuffles the given inputs.
     * @param images Images to shuffle.
     * @param labels Labels to shuffle.
     */

    private void shuffle(double[][] images, int[] labels) {
        Random randomizer = new Random();
        for(int i = images.length - 1; i > 0; i--) {
            int index = randomizer.nextInt(i);
            swap(images, index, i);
            swap(labels, index, i);
        }
    }

    /**
     * Swaps the two given indices in the given array.
     * @param array array to swap indices on.
     * @param index1 index 1 to swap with.
     * @param index2 index 2 to swap with.
     */

    private void swap(double[][] array, int index1, int index2) {
        double[] temp = array[index1];
        array[index1] = array[index2];
        array[index2] = temp;
    }

    /**
     * Swaps the two given indices of the given array.
     * @param array The array to swap on.
     * @param index1 Index 1 to swap with.
     * @param index2 Index 2 to swap with.
     */

    private void swap(int[] array, int index1, int index2) {
        int temp = array[index1];
        array[index1] = array[index2];
        array[index2] = temp;
    }

    /**
     * Feeds the input forward in this network recursively.
     * @param input The image to feed forward.
     * @param layer The current layer should be = 0 when method is called.
     * @return The output of this network.
     */

    private double[] feedForward(double[] input, int layer) {
        if(layer == weights.length - 1)
            return input(input, layer);
        else
            return feedForward(input(input, layer), layer + 1);
    }

    /**
     * Runs the input through the layer and calculates the weighted input and activation for the given layer.
     * @param input the image to calculate on.
     * @param layer the current layer.
     * @return returns the activation for the given layer.
     */

    private double[] input(double[] input, int layer) {
        weightedInput[layer] = weightedInput(input, layer);
        activation[layer] = activation(weightedInput[layer]);
        return activation[layer];
    }

    /**
     * Returns the weighted input for the given layer in this network.
     * @param input The image to calculate the weighted input from.
     * @param layer The layer to calculate the weighted input on.
     * @return Returns the weighted input from this layer.
     */

    private double[] weightedInput(double[] input, int layer) {
        return Matrix.add(Matrix.multiplication(weights[layer], input), bias[layer]);
    }

    /**
     * Returns the activation from this given input using the sigmoid function.
     * @param input The array to apply the sigmoid function to.
     * @return Returns a new array where the sigmoid function has been applied to all indices from the input.
     */

    private static double[] activation(double[] input) {
        double[] result = new double[input.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = Functions.sigmoid(input[i]);
        }
        return result;
    }

    /**
     * backPropagates this network by first calculating the output error and then calls backPropagateRec to do all the other layers.
     * @param output The output from the network from feeding forward.
     * @param input The input used to get the output.
     * @param label The label that corresponds to the input.
     */

    private void backPropagate(double[] output, double[] input, int label) {
        int layer = totalBiasError.length - 1;
        double[] outputError = getOutputError(output, label);
        addToBiasError(outputError, layer);
        addToWeightError(input, outputError, layer);
        backPropagateRec(outputError, input, layer - 1);
    }

    /**
     * backPropagates on this network recursively.
     * @param prevError The error from the previous layer.
     * @param input The input is the image used to get output from feeding forward.
     * @param layer The current layer to backPropagate.
     */

    private void backPropagateRec(double[] prevError, double[] input, int layer) {
        if(layer >= 0) {
        double[] weightedError = getWeightedError(prevError, layer);
        double[] error = Matrix.hadamardProduct(weightedError, sigmoidPrime(weightedInput[layer]));
        addToBiasError(error, layer);
        addToWeightError(input, error, layer);
        backPropagateRec(error, input, layer - 1);
        }
    }

    /**
     * @param prevError The error from layer + 1.
     * @param layer The current layer to get the weighted error from.
     * @return The weighted error from the givens layer.
     */

    private double[] getWeightedError(double[] prevError, int layer) {
        return Matrix.multiplication(Matrix.transpose(weights[layer + 1]), prevError);
    }

    /**
     * Calculates the output error from the output gotten from feeding forward an input on this network.
     * @param output The output from feeding forward on this network.
     * @param label The label that corresponds to the input given when feeding forward.
     * @return Returns the error of the output layer in this network.
     */

    private double[] getOutputError(double[] output, int label) {
        double[] result = new double[output.length];
        for(int i = 0; i < result.length; i++) {
            if(i == label)
                result[i] = output[i] - 1.0;
            else
                result[i] = output[i] - 0.0;
        }
        return Matrix.hadamardProduct(result, sigmoidPrime(weightedInput[weightedInput.length - 1]));
    }

    /**
     *
     *
     * Adds the given error to the total bias error at the given layer.
     * @param error The error to add.
     * @param layer The layer to add it to.
     */

    private void addToBiasError(double[] error, int layer) {
        totalBiasError[layer] = Matrix.add(totalBiasError[layer], error);
    }

    /**
     * Adds the given error to the total weight error in the given layer.
     * @param error The error to add on.
     * @param layer The layer to add it on to.
     */

    private void addToWeightError(double[] input, double[] error, int layer) {
        if(layer == 0)
            totalWeightError[layer] = Matrix.add(totalWeightError[layer], Matrix.multiplication(error, Matrix.transpose(input)));
        else
            totalWeightError[layer] = Matrix.add(totalWeightError[layer], Matrix.multiplication(error, Matrix.transpose(activation[layer - 1])));
    }

    /**
     * Calculates the sigmoid prime of the given input.
     * @param input The input to calculate the sigmoid prime.
     * @return Returns a new array with each index having applied sigmoid prime on each index in the input.
     */

    private static double[] sigmoidPrime(double[] input) {
        double[] result = new double[input.length];
        for(int i = 0; i < result.length; i++) {
            result[i] = Functions.sigmoidPrime(input[i]);
        }
        return result;
    }

    /**
     * Checks whether the given output from feeding forward is correct.
     * @param output The output from feeding forward in this network.
     * @param label The label that corresponds to the input used to get the output.
     * @return Returns true if the output is correct.
     */

    private static boolean isCorrect(double[] output, int label) {
        int highest = 0;
        for(int i = 0; i < output.length; i++) {
            if(output[i] >= output[highest])
                highest = i;
        }
        return (highest == label);
    }

    /**
     * @return Returns a textual representation of this network.
     */

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        for(int i = 0; i < weights.length; i++) {
            result.append("Layer: ").append(i).append(", ");
            result.append("Number of Neurons: ").append(weights[i].length).append(", ");
            result.append("Number of Weights per neuron: ").append(weights[i][0].length).append("\n");
            result.append("Weights: \n");
            result.append(weightsToString(i));
            result.append("Bias: ");
            result.append(biasToString(i));
        }
        return result.toString();
    }

    /**
     * Prints the weights in the given layer, prints them out for each neuron like
     * Neuron 0 [weight 1, weight 2, ... weight n]
     * Neuron 1 [weight 1, weight 2, ... weight n]
     * .
     * .
     * Neuron n [weight 1, weight 2, ... weight n]
     */

    private String weightsToString(int layer) {
        StringBuilder result = new StringBuilder();
        for(int i = 0; i < weights[layer].length; i++) {
            result.append("Neuron ").append(i).append(": [");
            for(int j = 0; j < weights[layer][i].length; j++) {
                if(j != weights[layer][i].length - 1)
                    result.append(weights[layer][i][j]).append(", ");
                else
                    result.append(weights[layer][i][j]).append("] \n");
            }
        }
        return result.toString();
    }

    /**
     * Prints out the biases for the given layer, output is as following;
     * Bias: [BiasForNeuron 0, BiasForNeuron 1, .. BiasForNeuron n]
     */

    private String biasToString(int layer) {
        StringBuilder result = new StringBuilder("[");
        for(int i = 0; i < bias[layer].length; i++) {
            if(i != bias[layer].length - 1)
                result.append(bias[layer][i]).append(", ");
            else
                result.append(bias[layer][i]).append("] \n");
        }
        return result.toString();
    }
}
