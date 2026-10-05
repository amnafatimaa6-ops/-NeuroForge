
#include "NeuralNetwork.hpp"
#include "Activation.hpp"

#include <random>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(
    std::size_t inputSize,
    std::size_t hiddenSize,
    double learningRate
)
    : weightsInputHidden_(hiddenSize, inputSize),
      weightsHiddenOutput_(1, hiddenSize),
      biasHidden_(hiddenSize, 1),
      biasOutput_(1, 1),
      learningRate_(learningRate)
{
    std::random_device rd;
    std::mt19937 generator(rd());

    std::uniform_real_distribution<double>
        distribution(-1.0, 1.0);

    for (std::size_t i = 0;
         i < hiddenSize;
         ++i)
    {
        for (std::size_t j = 0;
             j < inputSize;
             ++j)
        {
            weightsInputHidden_(i, j) =
                distribution(generator);
        }
    }

    for (std::size_t i = 0;
         i < hiddenSize;
         ++i)
    {
        weightsHiddenOutput_(0, i) =
            distribution(generator);
    }
}

double NeuralNetwork::predict(
    const Matrix& input
)
{
    Matrix hidden =
        weightsInputHidden_.multiply(input);

    hidden =
        hidden.add(biasHidden_);

    for (std::size_t i = 0;
         i < hidden.rows();
         ++i)
    {
        hidden(i, 0) =
            sigmoid(hidden(i, 0));
    }

    Matrix output =
        weightsHiddenOutput_.multiply(hidden);

    output =
        output.add(biasOutput_);

    return sigmoid(output(0, 0));
}

void NeuralNetwork::train(
    const Matrix& input,
    double target,
    double classWeight
)
{
    // Forward pass
    Matrix hiddenRaw =
        weightsInputHidden_.multiply(input);

    hiddenRaw =
        hiddenRaw.add(biasHidden_);

    Matrix hidden = hiddenRaw;

    for (std::size_t i = 0;
         i < hidden.rows();
         ++i)
    {
        hidden(i, 0) =
            sigmoid(hidden(i, 0));
    }

    Matrix outputRaw =
        weightsHiddenOutput_.multiply(hidden);

    outputRaw =
        outputRaw.add(biasOutput_);

    double output =
        sigmoid(outputRaw(0, 0));

    // Weighted output error.
    double outputError =
        (output - target) * classWeight;

    double outputGradient =
        outputError *
        sigmoidDerivative(
            outputRaw(0, 0)
        );

    // Save the original output weights.
    // The hidden-layer gradients must use
    // the weights from before the update.
    Matrix oldOutputWeights =
        weightsHiddenOutput_;

    // Update hidden -> output weights.
    for (std::size_t i = 0;
         i < hidden.rows();
         ++i)
    {
        double gradient =
            outputGradient *
            hidden(i, 0);

        weightsHiddenOutput_(0, i) -=
            learningRate_ * gradient;
    }

    // Update output bias.
    biasOutput_(0, 0) -=
        learningRate_ *
        outputGradient;

    // Update input -> hidden weights.
    for (std::size_t i = 0;
         i < hidden.rows();
         ++i)
    {
        double hiddenError =
            oldOutputWeights(0, i) *
            outputGradient;

        double hiddenGradient =
            hiddenError *
            sigmoidDerivative(
                hiddenRaw(i, 0)
            );

        for (std::size_t j = 0;
             j < input.rows();
             ++j)
        {
            weightsInputHidden_(i, j) -=
                learningRate_ *
                hiddenGradient *
                input(j, 0);
        }

        // Update hidden bias.
        biasHidden_(i, 0) -=
            learningRate_ *
            hiddenGradient;
    }
}

double NeuralNetwork::loss(
    double prediction,
    double target
)
{
    double error =
        prediction - target;

    return 0.5 * error * error;
}
