# 🚀 NeuroForge

### A Neural Network Engine Built from Scratch in C++17 for Scientific Data

**NeuroForge** is a lightweight neural-network engine implemented from scratch in **C++17 and the Standard Template Library (STL)**.

The project was created to understand what happens underneath high-level machine-learning frameworks by implementing the fundamental components of a neural network directly — including matrix operations, activation functions, forward propagation, backpropagation, gradient descent, bias updates, dataset preprocessing, and class-weighted learning.

Rather than stopping at a toy problem, NeuroForge was evaluated on real astronomical data from the **NASA Exoplanet Archive**.

---

## Scientific Experiment

The main experiment applies NeuroForge to exoplanet discovery data and asks:

> **Can a small neural network implemented entirely from scratch learn to distinguish transit discoveries from non-transit discoveries using planetary and stellar properties?**

The model uses nine numerical features describing properties such as orbital characteristics, planetary radius, equilibrium temperature, and host-star properties.

### Dataset Source

The astronomical observations were obtained from the:

**NASA Exoplanet Archive**

The archive is operated by the NASA Exoplanet Science Institute (NExScI) and provides publicly available astronomical data for research and analysis.

The original dataset was preprocessed specifically for this experiment. The processed dataset included in this repository is:

```text
data/exoplanets_clean.csv
```

The large raw archive export is **not included** in the repository because of file-size considerations.

---

# 🔬 Experiment Overview

The original NASA dataset contained:

* **40,194 observations**
* **9 numerical input features**
* **1 binary target**

The target was constructed as:

```text
Transit     → 1
Non-Transit → 0
```

For this experiment, every discovery method other than `Transit` was grouped into the non-transit class.

### Selected Features

| Feature       | Description                     |
| ------------- | ------------------------------- |
| `pl_orbper`   | Planet orbital period           |
| `pl_orbsmax`  | Planet orbital semi-major axis  |
| `pl_orbeccen` | Planet orbital eccentricity     |
| `pl_eqt`      | Planet equilibrium temperature  |
| `pl_rade`     | Planet radius                   |
| `st_teff`     | Host-star effective temperature |
| `st_mass`     | Host-star mass                  |
| `st_rad`      | Host-star radius                |
| `st_met`      | Host-star metallicity           |

---

# 🧹 Data Preprocessing

The raw NASA dataset contains missing numerical observations.

NeuroForge performs preprocessing directly in C++ rather than relying on an external data-science framework.

### Processing pipeline

```text
NASA Exoplanet Archive
          ↓
     CSV parsing
          ↓
   Feature selection
          ↓
   Missing-value detection
          ↓
     Median imputation
          ↓
   Binary label creation
          ↓
    Feature normalization
          ↓
      Train / Test split
          ↓
      Neural Network
```

### Missing Values

Missing numerical observations are replaced using the **median of the corresponding feature**.

The median was chosen because it is less sensitive to extreme values than the mean.

### Class Imbalance

The dataset contains substantially more transit observations than non-transit observations.

To reduce the effect of this imbalance, NeuroForge implements **class-weighted learning**.

The minority class receives a larger contribution to the training error, encouraging the network to pay attention to both classes.

---

#  Neural Network Architecture

The final experiment uses a compact feed-forward neural network:

```text
9 Input Features
       │
       ▼
8 Hidden Neurons
       │
       ▼
1 Output Neuron
       │
       ▼
Transit Probability
```

### Configuration

| Parameter         |     Value |
| ----------------- | --------: |
| Input neurons     |         9 |
| Hidden neurons    |         8 |
| Output neurons    |         1 |
| Hidden activation |   Sigmoid |
| Output activation |   Sigmoid |
| Learning rate     |      0.01 |
| Training epochs   |       300 |
| Train/Test split  | 80% / 20% |

The model contains only a small number of parameters, keeping the implementation transparent and computationally lightweight.

---

# 📐 Mathematics

One of the main goals of **NeuroForge** is to expose the mathematics behind the neural network rather than hiding the implementation behind a high-level machine-learning API.

## Forward Propagation

For the hidden layer, the linear transformation is:

$$
z_h = W_hx + b_h
$$

The sigmoid activation is then applied:

$$
a_h = \sigma(z_h)
$$

where:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The output layer uses the hidden-layer activations:

$$
z_o = W_oa_h + b_o
$$

followed by the sigmoid function:

$$
\hat{y} = \sigma(z_o)
$$

This produces a continuous prediction between 0 and 1.

---

## Loss Function

NeuroForge minimises the squared-error loss:

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

where $y$ is the target value and $\hat{y}$ is the network prediction.

For class-weighted training, the loss contribution can be scaled by a class-specific weight $w$:

$$
L_w = wL
$$

---

## Backpropagation

The error at the output is:

$$
e = \hat{y} - y
$$

Using the derivative of the sigmoid activation, the output-layer error signal becomes:

$$
\delta_o =
(\hat{y} - y)\sigma'(z_o)
$$

The error is then propagated back to the hidden layer through the output weights:

$$
\delta_h =
(W_o\delta_o)
\odot
\sigma'(z_h)
$$

where $\odot$ denotes element-wise multiplication.

These error signals are used to compute the gradients of the weights and biases.

---

## Parameter Updates

NeuroForge updates its parameters using gradient descent:

$$
W \leftarrow W - \eta\nabla W
$$

$$
b \leftarrow b - \eta\nabla b
$$

where $\eta$ is the learning rate, $\nabla W$ represents the weight gradient, and $\nabla b$ represents the bias gradient.

This completes the training cycle: the network performs a forward pass, evaluates its error, propagates that error backwards, and adjusts its parameters to reduce the loss.

---

# ⚙️ What NeuroForge Implements

The engine was implemented without using a high-level machine-learning framework for the core learning system.

### Core ML implementation

* Matrix multiplication
* Matrix addition
* Matrix subtraction
* Sigmoid activation
* Sigmoid derivative
* ReLU activation
* ReLU derivative
* Forward propagation
* Backpropagation
* Gradient descent
* Weight initialization
* Bias updates
* Class-weighted learning
* Mean training loss

### Data processing

* CSV parsing
* Feature extraction
* Missing-value detection
* Median imputation
* Numerical normalization
* Random train/test splitting
* Binary target construction

### Evaluation

* Confusion matrix
* Accuracy
* Precision
* Recall
* Specificity
* Balanced accuracy

---

# 📊 Results

NeuroForge was trained on **32,155 observations** and evaluated on a separate **8,039-observation test set**.

### Final Test Performance

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **89.45%** |
| Balanced Accuracy | **87.41%** |
| Precision         |   **0.98** |
| Recall            |   **0.90** |

### Confusion Matrix

```text
                  Predicted
                  0       1

Actual 0         705     126
Actual 1         722    6486
```

The model achieved high precision while maintaining strong recall for the transit class.

The balanced accuracy is also reported because ordinary accuracy can be misleading when the classes are imbalanced.

---

#  Training Behaviour

Training loss decreased throughout the experiment:

```text
Epoch 0       → 0.119837
Epoch 25      → 0.0574817
Epoch 50      → 0.0478365
Epoch 100     → 0.0430867
Epoch 150     → 0.0420875
Epoch 200     → 0.0418468
Epoch 250     → 0.0416575
Epoch 299     → 0.0415183
```

The decreasing loss indicates that the network successfully learned a predictive mapping from the selected astronomical features to the binary target.

---

#  Project Structure

```text
NeuroForge/
│
├── include/
│   ├── Matrix.hpp
│   ├── NeuralNetwork.hpp
│   ├── Activation.hpp
│   ├── Dataset.hpp
│   └── Preprocessor.hpp
│
├── src/
│   ├── Matrix.cpp
│   ├── NeuralNetwork.cpp
│   ├── Activation.cpp
│   ├── Dataset.cpp
│   └── Preprocessor.cpp
│
├── data/
│   ├── xor.csv
│   ├── test.csv
│   └── exoplanets_clean.csv
│
├── main.cpp
├── CMakeLists.txt
└── README.md
```

---

# 🛠️ Technologies

* **C++17**
* **STL**
* Standard file I/O
* Standard mathematical functions
* CMake
* Git / GitHub

No TensorFlow, PyTorch, Eigen, or other high-level ML library is used for the core neural-network implementation.

---

#  Building and Running

Clone the repository:

```bash
git clone https://github.com/amnafatimaa6-ops/-NeuroForge.git
```

Enter the project:

```bash
cd NeuroForge
```

Compile with a C++17 compiler:

```bash
g++ -std=c++17 main.cpp src/*.cpp -Iinclude -o NeuroForge
```

Run:

```bash
./NeuroForge
```

On Windows using MinGW:

```bash
NeuroForge.exe
```

---

#  Why Build a Neural Network from Scratch?

Modern machine-learning frameworks make model development extremely convenient, but that convenience can hide the underlying mathematics and computational processes.

NeuroForge was designed as a learning and engineering experiment to make those processes explicit.

Instead of writing:

```python
model.fit(X, y)
```

the project implements the mechanisms behind learning:

```text
Matrix Operations
       ↓
Forward Propagation
       ↓
Prediction
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Gradient Calculation
       ↓
Parameter Updates
       ↓
Learning
```

This provides a low-level perspective on how neural networks actually learn.

---

#  Why Astronomy?

Astronomy produces large scientific datasets containing measurements of planets, stars, orbital systems, and transient phenomena.

This makes it an interesting environment for experimenting with machine learning while connecting software engineering with scientific computing.

NeuroForge therefore serves as both:

1. A **from-scratch machine-learning implementation**, and
2. A **scientific-data experiment using real astronomical observations**.

---

#  Limitations

This project is an educational and experimental neural-network implementation rather than a production astronomical classification system.

Important limitations include:

* The model architecture is intentionally small.
* Only nine numerical features were used.
* Discovery methods were simplified into a binary target.
* The experiment does not reproduce the complete scientific pipeline used by astronomers.
* The implementation does not use advanced optimizers such as Adam.
* The experiment does not perform systematic hyperparameter optimization.
* The processed dataset is provided for reproducibility, while the original raw archive export is excluded because of file-size constraints.

The reported results should therefore be interpreted as an engineering and machine-learning experiment rather than a scientific claim about exoplanet discovery.

---

#  Future Work

Possible extensions include:

* Mini-batch training
* Additional optimization algorithms
* More advanced neural-network architectures
* Better handling of class imbalance
* Cross-validation
* Feature-selection experiments
* ROC and Precision-Recall analysis
* Comparison with conventional ML models
* Larger astronomical datasets
* Reproducible random seeds
* Performance optimization using multithreading

---

#  Project Goals

NeuroForge was built around three goals:

### 1. Understand the mathematics

Implement the fundamental equations behind neural-network learning rather than treating the model as a black box.

### 2. Understand the engineering

Build the supporting infrastructure — matrices, datasets, preprocessing, training, and evaluation — directly in C++.

### 3. Apply it to real science

Move beyond toy datasets and evaluate the system on publicly available astronomical observations.

---

#  Author

**Amna Fatima**

Computer Science / Artificial Intelligence student interested in:

* Artificial Intelligence
* Machine Learning
* Scientific Computing
* Astronomy
* Computational Physics
* Data Science

---

#  License

This project is released under the license included in this repository.

The astronomical data used in the experiment originates from the **NASA Exoplanet Archive** and remains subject to the archive's applicable data policies and attribution requirements.

---

##  Summary

**NeuroForge** is an attempt to look underneath the abstractions of modern machine learning.

It combines:

**C++ systems programming + neural-network mathematics + scientific data + astronomy**

to build a transparent machine-learning system from the ground up.

> **From matrices to predictions — NeuroForge makes the learning process visible.** 🚀
