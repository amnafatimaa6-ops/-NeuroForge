
#pragma once

#include "Matrix.hpp"

class NeuralNetwork
{
private:
    Matrix weightsInputHidden_;
    Matrix weightsHiddenOutput_;

    Matrix biasHidden_;
    Matrix biasOutput_;

    double learningRate_;

public:
    NeuralNetwork(
        std::size_t inputSize,
        std::size_t hiddenSize,
        double learningRate = 0.01
    );

    double predict(
        const Matrix& input
    );

    void train(
        const Matrix& input,
        double target,
        double classWeight = 1.0
    );

    double loss(
        double prediction,
        double target
    );
};
