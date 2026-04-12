# 📈 Linear Regression from Scratch (NumPy)

This project implements **Linear Regression using Gradient Descent** from scratch using Python and NumPy — no ML libraries like scikit-learn.

The goal is simple:
We want to learn a line that best fits our data.

---

# 🧠 What This Project Teaches

- How linear regression works mathematically
- What gradient descent actually does
- How loss is minimized step-by-step
- How weights (w) and bias (b) are updated

---

# 📦 Step-by-Step Code Explanation

---

## 🔹 1. Importing Libraries

``` python
import numpy as np
import matplotlib.pyplot as plt
``` 

- **numpy (np)** → used for numerical operations (arrays, dot products)
- **matplotlib** → used to visualize the dataset

---

## 🔹 2. Creating the Dataset

``` python
X = 2 * np.random.rand(500, 1)
y = 5 + 3 * X + np.random.randn(500, 1)
``` 

### What’s happening:

- `X` → 500 random input values between 0 and 2
- `y` → generated using the equation:

  y = 5 + 3X + noise

So the **true relationship** is:
- slope = 3
- intercept = 5

Noise is added using:
`np.random.randn()` → makes the data realistic (not perfectly linear)

---

## 🔹 3. Visualizing the Dataset

``` python
plt.scatter(X, y)
``` 

- Each dot = one data point
- You should see a rough straight-line pattern

---

## 🔹 4. Hypothesis Function

``` python
def hypothesis(X, w, b):
    return np.dot(X, w) + b
``` 

This is your model:

👉 Prediction formula:
ŷ = X·w + b

### Variables:
- `X` → input data
- `w` → weight (slope)
- `b` → bias (intercept)
- `y_pred` → predicted output

---

## 🔹 5. Loss Function (Mean Squared Error)

``` python
def loss(y, y_pred): 
    n = len(y)
    return (1/n) * (np.sum((y - y_pred) ** 2))
``` 

### What it does:
- Measures how bad predictions are

### Formula:
MSE = `(1/n) * Σ(y - y_pred)^2`

- If loss is high → model is bad
- If loss is low → model is good

---

## 🔹 6. Gradient Function

``` python
def gradient(X, y, y_pred):
    n = len(y)
    dw = -(2/n) * np.dot(X.T, (y - y_pred))
    db = -(2/n) * np.sum(y-y_pred)

    return dw, db
``` 

### This is the most important part.

It calculates how to update:
- weight (w)
- bias (b)

### Variables:
- `dw` → gradient for weight
- `db` → gradient for bias

These tell us:
👉 "Which direction should we move to reduce error?"

---
### 🔍 Why do we use `np.dot` for dw instead of `np.sum`?

This is not just a stylistic choice — it's **mathematically important**.

#### 👉 Case 1: Using `np.sum` (your earlier version)

```python
dw = -(2/n) * np.sum((y - y_pred) * X)
```

- Works fine **only when X has a single feature**
- This is element-wise multiplication followed by summation

---

#### 👉 Case 2: Using `np.dot`

```python
dw = -(2/n) * np.dot(X.T, (y - y_pred))
```

- This is the **correct and scalable approach**
- Works for:
  - single feature ✅
  - multiple features ✅

---

### ⚡ Why `np.dot` is better:
According to the mathematical formula for gradient:

`∂J/∂w = -(2/n) * Σ (y - y_pred) * X`

Here, you see a **summation (Σ)** over all training examples.

But when we move to matrix form (which is how NumPy works efficiently),  
this summation can be written as:

`Xᵀ · (y - y_pred)`

So instead of manually summing element-wise products, we use:

```python
np.dot(X.T, (y - y_pred))
```

This is just the **vectorized version of the same summation**,  
not a different formula.


---

### 🧩 What about db?

```python
db = -(2/n) * np.sum(y - y_pred)
```

- Bias is a single number → so simple summation works fine
- No need for dot product here


---

## 🔹 7. Training Loop (Gradient Descent)

``` python
w , b = 0, 0
learning_rate = 0.01
epochs = 5000
``` 

### Initial Setup:
- `w, b = 0` → start with random guess
- `learning_rate` → step size (how fast we learn)
- `epochs` → number of iterations

---

### 🔁 Training Process

``` python
for i in range(epochs):
    y_pred = hypothesis(X, w, b)

    model_loss = loss(y, y_pred)

    dw, db = gradient(X, y, y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db
``` 

### What happens each loop:

1. Predict values → `y_pred`
2. Calculate error → `loss`
3. Compute gradients → `dw, db`
4. Update parameters:
   - w = w - lr * dw
   - b = b - lr * db

👉 This gradually improves the model

---

## 🔹 8. Logging Progress

``` python
if (i % 100 == 0):
    print(f"The loop ran {i} times and the loss is {model_loss}")
``` 

- Prints loss every 100 iterations
- You should see loss decreasing over time

If it doesn’t → something is wrong

---

# ⚠️ Important Notes

- If loss increases → learning rate is too high
- If training is slow → learning rate is too low
- If shapes mismatch → NumPy will silently break things

---

# ✅ Final Outcome

After training:
- `w` should be close to **3**
- `b` should be close to **5**

That means your model successfully learned the underlying pattern.

---