
#include <iostream>
#include <iomanip>

#include "Preprocessor.hpp"
#include "Dataset.hpp"
#include "NeuralNetwork.hpp"

int main()
{
    std::cout
        << "NeuroForge - NASA Exoplanet Classifier\n";

    std::cout
        << "========================================\n\n";

    const std::string inputFile =
        "data/exoplanets.csv";

    const std::string cleanFile =
        "data/exoplanets_clean.csv";

    // --------------------------------------------------
    // STEP 1: Preprocess NASA dataset
    // --------------------------------------------------

    Preprocessor preprocessor;

    std::cout
        << "Preprocessing NASA dataset...\n";

    if (!preprocessor.createCleanDataset(
            inputFile,
            cleanFile))
    {
        std::cerr
            << "ERROR: Preprocessing failed.\n";

        return 1;
    }

    // --------------------------------------------------
    // STEP 2: Load dataset
    // --------------------------------------------------

    Dataset dataset;

    if (!dataset.loadCSV(
            cleanFile,
            9))
    {
        std::cerr
            << "ERROR: Dataset loading failed.\n";

        return 1;
    }

    std::cout
        << "Dataset loaded successfully.\n";

    std::cout
        << "Samples: "
        << dataset.size()
        << '\n';

    std::cout
        << "Features: "
        << dataset.featureCount()
        << "\n\n";

    // Normalize numerical features.
    dataset.normalize();

    // --------------------------------------------------
    // STEP 3: Train/test split
    // --------------------------------------------------

    Dataset trainSet;
    Dataset testSet;

    dataset.trainTestSplit(
        0.8,
        trainSet,
        testSet
    );

    std::cout
        << "Training samples: "
        << trainSet.size()
        << '\n';

    std::cout
        << "Testing samples: "
        << testSet.size()
        << "\n\n";

    // --------------------------------------------------
    // STEP 4: Count classes
    // --------------------------------------------------

    std::size_t positiveCount = 0;
    std::size_t negativeCount = 0;

    for (const auto& sample :
         trainSet.samples())
    {
        if (sample.label == 1.0)
        {
            ++positiveCount;
        }
        else
        {
            ++negativeCount;
        }
    }

    std::cout
        << "Training class distribution:\n";

    std::cout
        << "  Transit (1): "
        << positiveCount
        << '\n';

    std::cout
        << "  Non-Transit (0): "
        << negativeCount
        << "\n\n";

    if (
        positiveCount == 0 ||
        negativeCount == 0
    )
    {
        std::cerr
            << "ERROR: Training split contains only one class.\n";

        return 1;
    }

    // --------------------------------------------------
    // STEP 5: Class weighting
    // --------------------------------------------------

    double negativeWeight = 1.0;
    double positiveWeight = 1.0;

    if (negativeCount < positiveCount)
    {
        negativeWeight =
            static_cast<double>(positiveCount) /
            static_cast<double>(negativeCount);

        if (negativeWeight > 20.0)
        {
            negativeWeight = 20.0;
        }
    }
    else if (positiveCount < negativeCount)
    {
        positiveWeight =
            static_cast<double>(negativeCount) /
            static_cast<double>(positiveCount);

        if (positiveWeight > 20.0)
        {
            positiveWeight = 20.0;
        }
    }

    std::cout
        << "Class weights:\n";

    std::cout
        << "  Non-Transit: "
        << negativeWeight
        << '\n';

    std::cout
        << "  Transit: "
        << positiveWeight
        << "\n\n";

    // --------------------------------------------------
    // STEP 6: Build neural network
    // --------------------------------------------------

    const std::size_t inputSize =
        dataset.featureCount();

    const std::size_t hiddenSize =
        8;

    const double learningRate =
        0.01;

    NeuralNetwork network(
        inputSize,
        hiddenSize,
        learningRate
    );

    // FAST VERSION
    const int epochs =
        300;

    // --------------------------------------------------
    // STEP 7: Train
    // --------------------------------------------------

    std::cout
        << "Training NeuroForge...\n\n";

    for (int epoch = 0;
         epoch < epochs;
         ++epoch)
    {
        double totalLoss = 0.0;

        for (const auto& sample :
             trainSet.samples())
        {
            Matrix input(
                inputSize,
                1
            );

            for (std::size_t i = 0;
                 i < inputSize;
                 ++i)
            {
                input(i, 0) =
                    sample.features[i];
            }

            double classWeight =
                sample.label == 1.0
                    ? positiveWeight
                    : negativeWeight;

            network.train(
                input,
                sample.label,
                classWeight
            );

            double prediction =
                network.predict(input);

            totalLoss +=
                network.loss(
                    prediction,
                    sample.label
                );
        }

        if (
            epoch % 25 == 0 ||
            epoch == epochs - 1
        )
        {
            double averageLoss =
                totalLoss /
                static_cast<double>(
                    trainSet.size()
                );

            std::cout
                << "Epoch "
                << epoch
                << " | Loss: "
                << averageLoss
                << '\n';
        }
    }

    // --------------------------------------------------
    // STEP 8: Evaluation
    // --------------------------------------------------

    std::cout
        << "\nEvaluation\n";

    std::cout
        << "----------\n";

    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;

    for (const auto& sample :
         testSet.samples())
    {
        Matrix input(
            inputSize,
            1
        );

        for (std::size_t i = 0;
             i < inputSize;
             ++i)
        {
            input(i, 0) =
                sample.features[i];
        }

        double prediction =
            network.predict(input);

        int predictedClass =
            prediction >= 0.5
                ? 1
                : 0;

        int actualClass =
            static_cast<int>(
                sample.label
            );

        if (
            predictedClass == 1 &&
            actualClass == 1
        )
        {
            ++truePositive;
        }
        else if (
            predictedClass == 0 &&
            actualClass == 0
        )
        {
            ++trueNegative;
        }
        else if (
            predictedClass == 1 &&
            actualClass == 0
        )
        {
            ++falsePositive;
        }
        else
        {
            ++falseNegative;
        }
    }

    int totalCorrect =
        truePositive +
        trueNegative;

    int totalSamples =
        truePositive +
        trueNegative +
        falsePositive +
        falseNegative;

    double accuracy =
        100.0 *
        static_cast<double>(totalCorrect) /
        static_cast<double>(totalSamples);

    std::cout
        << "\nConfusion Matrix\n";

    std::cout
        << "                  Predicted\n";

    std::cout
        << "                  0       1\n";

    std::cout
        << "Actual 0       "
        << std::setw(5)
        << trueNegative
        << "   "
        << std::setw(5)
        << falsePositive
        << '\n';

    std::cout
        << "Actual 1       "
        << std::setw(5)
        << falseNegative
        << "   "
        << std::setw(5)
        << truePositive
        << "\n\n";

    std::cout
        << std::fixed
        << std::setprecision(2);

    std::cout
        << "Test Accuracy: "
        << accuracy
        << "%\n";

    // --------------------------------------------------
    // Additional metrics
    // --------------------------------------------------

    double precision = 0.0;
    double recall = 0.0;
    double specificity = 0.0;

    if (
        truePositive + falsePositive > 0
    )
    {
        precision =
            static_cast<double>(truePositive) /
            static_cast<double>(
                truePositive + falsePositive
            );
    }

    if (
        truePositive + falseNegative > 0
    )
    {
        recall =
            static_cast<double>(truePositive) /
            static_cast<double>(
                truePositive + falseNegative
            );
    }

    if (
        trueNegative + falsePositive > 0
    )
    {
        specificity =
            static_cast<double>(trueNegative) /
            static_cast<double>(
                trueNegative + falsePositive
            );
    }

    double balancedAccuracy =
        100.0 *
        (recall + specificity) /
        2.0;

    std::cout
        << "Precision: "
        << precision
        << '\n';

    std::cout
        << "Recall: "
        << recall
        << '\n';

    std::cout
        << "Balanced Accuracy: "
        << balancedAccuracy
        << "%\n";

    std::cout
        << "\nNeuroForge NASA experiment complete!\n";

    return 0;
}

