# 📘 Multilinear Regression — Detailed Explanation (Based on Implementation)

---

## 📌 What is Multilinear Regression?

Multilinear Regression is a supervised learning algorithm used to model the relationship between **multiple input features** and a **single continuous output**.

Mathematically:

    y = w1*x1 + w2*x2 + ... + wn*xn + b

In vectorized form (what we actually implement):

    y = X · w + b

Where:
- X → matrix of input features
- w → weight vector
- b → bias (intercept)
- y → predicted output

---

## 🧠 Core Intuition

Each feature contributes **some weighted influence** to the final prediction.

In your dataset:
- Feature 1 contributes with weight ~2
- Feature 2 contributes with weight ~3

Your model’s job is to **learn these weights automatically**.

---

## ❗ Why we use a weight vector (w) instead of w1, w2 separately

You defined:

    w = np.zeros((X.shape[1], 1))

Instead of:
    w1, w2

### Why?

Because:
- Your data has 2 features now — but real datasets can have 100+
- Using separate variables does not scale
- Vectorization allows:
  - cleaner code
  - faster computation
  - easier math

---

## ⚡ What NumPy is doing behind the scenes

NumPy enables:

    np.dot(X, w)

Instead of manually doing:

    for each row:
        multiply each feature with its weight

NumPy:
- uses optimized C implementations
- performs matrix operations efficiently
- avoids Python loops (which are slow)

This is called **vectorization**, and it’s essential for performance.

---

# 📦 CODE WALKTHROUGH

## 🧩 Cell 1: Imports and Random Seed

``` python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
``` 

### Explanation:

- NumPy is imported for numerical operations
- Matplotlib is imported for visualization (not relevant to learning logic)
- `np.random.seed(123)` ensures reproducibility

👉 This means every time you run the code, you get the same random data.

---

## 🧩 Cell 2: Dataset Creation

``` python
X = np.random.rand(500, 2)
y = 4 + 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(500) * 0.1

y = y.reshape(-1, 1)
``` 

### What’s happening:

- `X` is a matrix of shape (500, 2)
  → 500 samples, 2 features

- Target `y` is generated using a known formula:
  
      y = 4 + 2*x1 + 3*x2 + noise

- `np.random.randn(500) * 0.1`
  adds small noise to simulate real-world data

---

### Important detail:

``` python
y = y.reshape(-1, 1)
``` 

This converts y into a column vector:

- Before: shape (500,)
- After:  shape (500,1)

👉 This ensures compatibility with matrix operations.

---

## 🧩 Cell 3: Hypothesis Function

``` python
def hypothesis(X, w, b):
    return np.dot(X,w) + b
``` 

### Explanation:

This is your prediction function.

- `np.dot(X, w)` computes weighted sum of features
- `+ b` shifts the output

Output:
- predicted values for all samples

---

## 🧩 Cell 4: Loss Function

``` python
def loss(y, y_pred):
    n = len(y)
    return (1/n) * np.sum((y-y_pred) ** 2)
``` 

### Explanation:

This implements **Mean Squared Error (MSE)**.

Formula:

    MSE = (1/n) * Σ(y - y_pred)^2

### Why squared error?
- Penalizes larger mistakes more heavily
- Always positive
- Smooth function → good for optimization

---

## 🧩 Cell 5: Gradient Function

``` python
def gradient(X, y, y_pred):
    n = len(y)

    dw = (-2/n) * np.dot(X.T, (y-y_pred))
    db = (-2/n) * np.sum(y-y_pred)

    return dw, db
``` 

### This is the most important part of your model.

It computes:
- dw → gradient for weights
- db → gradient for bias

---

### Why gradients matter:

They tell you:
> “Which direction should I move to reduce error?”

---

### Key detail:

``` python
np.dot(X.T, (y - y_pred))
``` 

- `X.T` → transpose of X
- Aligns dimensions for matrix multiplication

---


## 🧩 Cell 6: Training Loop (Gradient Descent)

``` python
w = np.zeros((X.shape[1], 1))
b = 0
learning_rate = 0.001
epochs = 3000

for i in range(epochs):
    y_pred = hypothesis(X,w,b)
    model_loss = loss(y, y_pred)

    dw, db = gradient(X, y, y_pred)

    w = w - learning_rate*dw
    b = b - learning_rate*db

    if (i % 100 == 0):
        print(f"Iteration {i}, Loss: {model_loss}")
``` 

---

### Step-by-step breakdown:

#### 1. Initialize parameters
- w starts as zeros
- b starts as 0

#### 2. Loop over epochs
Each loop = one learning step

#### 3. Predict
``` 
y_pred = hypothesis(X,w,b)
``` 

#### 4. Compute loss
``` 
model_loss = loss(y, y_pred)
``` 

#### 5. Compute gradients
``` 
dw, db = gradient(X, y, y_pred)
``` 

#### 6. Update parameters
``` 
w = w - learning_rate * dw
b = b - learning_rate * db
``` 

---

### Learning Rate Insight:

- Too large → model diverges
- Too small → very slow learning

Your choice (0.001) is safe and stable.

---

## 🧩 Cell 7: Output Learned Parameters

``` python
w, b
``` 

### What this shows:

- Learned weights should be close to:
    [2, 3]
- Bias should be close to:
    4

If your model worked correctly, you’ll see similar values.

---

## 🔍 Final Understanding

This model:
- learned from synthetic data
- approximated the true relationship
- minimized prediction error using gradient descent

---