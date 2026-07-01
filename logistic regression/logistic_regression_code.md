# 📦 Logistic Regression — Code Walkthrough (Based on Implementation)

> This explains the code in `logistic_fixed.ipynb`. For the underlying theory (sigmoid, cross-entropy derivation, gradients, confusion matrix, regularization), see `logistic_regression_notes.md`.

---

## 🧩 Cell 1: Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

- NumPy → numerical operations (dot products, exponentials, sums)
- Matplotlib → plotting the data and the model's output

---

## 🧩 Cell 2: Load Dataset

```python
data = np.loadtxt("dataset.csv", delimiter=",", skiprows=1)
X = data[:, :3]
y = data[:, 3]

y = y.reshape(-1,1)
```

### Explanation:

- Loads `dataset.csv` (columns: `hours_studied`, `attendance`, `sleep_hours`, `passed`), skipping the header row
- `X` = first 3 columns (the features)
- `y` = last column (the target: 0 = fail, 1 = pass)
- `y.reshape(-1,1)` turns `y` from shape `(n,)` into `(n,1)` — a column vector, so it lines up correctly with matrix operations later (`X @ w` produces shape `(n,1)` too)

---

## 🧩 Cell 3: Feature Standardization

```python
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # the datapoints are now in a particular range
```

### Explanation:

This is **standardization** (z-score normalization): for each feature, subtract its mean and divide by its standard deviation.

- After this, every feature has **mean 0** and **standard deviation 1**
- Important because `hours_studied`, `attendance`, and `sleep_hours` are on very different scales (e.g. attendance in the 0–100 range vs. sleep hours in the 0–10 range) — without standardizing, gradient descent would move much faster along one feature's axis than another, making training slow or unstable
- Also why the x-axis in the plots later shows small numbers like -2 to 2, instead of raw hours/percentages

---

## 🧩 Cells 4–6: Visualizing Each Feature vs Target

```python
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y)
plt.xlabel("Feature (hours_studied)")
plt.ylabel("Target (passed)")
plt.title("Hours Studied vs Y")
plt.show()
```

*(repeated for `attendance` and `sleep_hours`)*

### Explanation:

Simple exploratory scatter plots — for each feature independently, plot it against the pass/fail outcome (0 or 1). This is just to visually sanity-check that the features seem to relate to the outcome (e.g. more study hours → more passes) before building the model.

---

## 🧩 Cell 7: Hypothesis Function (Sigmoid)

```python
def hypothesis(w, X, b): 
    z = np.dot(X,w) + b
    return 1/(1 + np.exp(-z))
```

### Explanation:

This is the model itself, directly implementing:

    z = Xw + b
    ŷ = σ(z) = 1 / (1 + e^-z)

- `np.dot(X, w)` computes the weighted sum of features for every sample at once (vectorized — no loop needed)
- `+ b` adds the bias
- `1/(1 + np.exp(-z))` applies the sigmoid, squashing every output into (0, 1) so it can be read as a probability

---

## 🧩 Cell 8: Loss Function (Cross-Entropy)

```python
def loss(y_pred, y, X):
    n = len(y)
    J = -1/n * np.sum(y*np.log(y_pred) + (1-y) * np.log(1-y_pred))
    return J
```

### Explanation:

This directly implements the cross-entropy cost function derived in the notes:

    J = -(1/n) Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

- `y*np.log(y_pred)` → active only when the true label is 1 (penalizes low predicted probability)
- `(1-y) * np.log(1-y_pred)` → active only when the true label is 0 (penalizes high predicted probability)
- Summing over all samples and dividing by `n` gives the **average** loss across the dataset
- The leading `-1/n` flips this from a maximization (log-likelihood) into a minimization problem, which is what gradient descent needs

---

## 🧩 Cell 9: Gradient Function

```python
def gradient(y_pred, y, X):
    n = len(y)
    dw = 1/n * np.dot(X.T,(y_pred - y))
    db = 1/n * np.sum(y_pred - y)
    return dw, db
```

### Explanation:

This implements the gradients derived analytically in the notes:

    ∂J/∂w = (1/m) Σ xᵢ(ŷᵢ - yᵢ)
    ∂J/∂b = (1/m) Σ (ŷᵢ - yᵢ)

- `y_pred - y` → the error for every sample (how far off each prediction was)
- `np.dot(X.T, (y_pred - y))` → this single line computes the weighted sum of errors across *all* features and *all* samples simultaneously (equivalent to summing `xᵢ(ŷᵢ-yᵢ)` for each feature, over all `i`)
- `np.sum(y_pred - y)` → the plain sum of errors, for the bias gradient
- Both divided by `n` to get the average, matching the loss function's averaging

This is the same beautifully simple `ŷ - y` result derived from the chain rule in the theory notes — it's what makes logistic regression's gradient so clean despite the messy sigmoid/log derivation behind it.

---

## 🧩 Cell 10: Training Loop (Gradient Descent)

```python
learning_rate = 0.002
epochs = 3000
w = np.zeros((X.shape[1], 1))
b = 0

for i in range(epochs):
    y_pred = hypothesis(w, X, b)
    
    model_loss = loss(y_pred, y, X)

    dw, db = gradient(y_pred, y, X)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (i % 100 == 0):
        print(f"Iteration: {i}, Loss: {model_loss}")
```

### Step-by-step breakdown:

1. **Initialize** `w` as a zero vector (one weight per feature) and `b` as 0
2. **Loop** for 3000 epochs — each iteration is one full gradient descent step over the whole dataset (batch gradient descent)
3. **Forward pass**: compute predictions `y_pred = hypothesis(w, X, b)`
4. **Compute loss**: track how wrong the current model is
5. **Compute gradients**: `dw`, `db` tell us which direction reduces the loss
6. **Update parameters**: move `w` and `b` a small step (`learning_rate`) in the direction that reduces loss
7. **Print progress** every 100 iterations, so you can watch the loss decrease as training proceeds

**Learning rate insight:** 0.002 is small and safe. If it were much larger, training could diverge (loss increases or oscillates); if much smaller, training would take far more epochs to converge.

---

## 🧩 Cell 11: Why the raw per-feature plots zigzag (and the fix)

Your model uses **3 features together**: `z = w0*x0 + w1*x1 + w2*x2 + b`.

If you plot predicted probability against just *one* feature (sorted by that feature), the other two features are **not** sorted along with it — they jump around randomly at each x-value. That randomness is what shows up as a zigzag in the curve. **It's not a bug** in the gradient descent or the math — it's a side effect of viewing a multi-feature model through a single-feature slice.

Two ways to get clean curves instead:

1. **Plot vs. the model's own 1D input `z` (the logit).** Since sigmoid is a pure function of `z`, this is *always* a perfectly smooth S-curve, no matter how many features the model has.
2. **Partial dependence per feature** — hold the *other* features fixed at their mean, vary just one feature, and predict. This isolates "what does increasing hours_studied alone do, all else average" — smooth, and keeps a per-feature view.

---

## 🧩 Cell 12: Clean Sigmoid Plot (vs. logit z)

```python
z = np.dot(X, w) + b
y_pred_all = hypothesis(w, X, b)

z_idx = z[:, 0].argsort()
z_sorted = z[z_idx, 0]
pred_sorted = y_pred_all[z_idx, 0]

plt.figure(figsize=(8,6))
plt.scatter(z, y, label="Actual Data", alpha=0.6)
plt.plot(z_sorted, pred_sorted, label="Sigmoid Output", color='red', linewidth=2)

plt.xlabel("z (linear combination = X.w + b)")
plt.ylabel("Probability")
plt.title("Model Output vs. Logit (z) -- always a clean sigmoid")
plt.legend()
plt.show()
```

### Explanation:

- `z = np.dot(X, w) + b` — recompute the logit for every sample using the trained weights
- Sort samples by `z` (not by a raw feature) so the x-axis is monotonic in the model's *actual* single input variable
- Plot predicted probability against `z` → produces a perfectly smooth S-curve, since `ŷ = σ(z)` is a strict one-to-one function of `z`
- Confirms the trained model itself is behaving correctly

---

## 🧩 Cells 13–15: Partial Dependence Plots (one per feature)

```python
feature_idx = 0  # 0=hours_studied, 1=attendance, 2=sleep_hours

x_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 200)
X_pd = np.tile(np.mean(X, axis=0), (len(x_range), 1))
X_pd[:, feature_idx] = x_range

y_pd = hypothesis(w, X_pd, b)

plt.figure(figsize=(8,6))
plt.scatter(X[:, feature_idx], y, label="Actual Data", alpha=0.6)
plt.plot(x_range, y_pd[:, 0], label="Model Output (others held at mean)", color='red', linewidth=2)

plt.xlabel("Feature (standardized)")
plt.ylabel("Probability")
plt.title("Partial Dependence")
plt.legend()
plt.show()
```

### Explanation:

1. `x_range` → 200 evenly-spaced values spanning the chosen feature's full range
2. `X_pd = np.tile(np.mean(X, axis=0), (len(x_range), 1))` → build a synthetic dataset where **every row is the dataset's mean feature vector**, repeated 200 times
3. `X_pd[:, feature_idx] = x_range` → overwrite just the one feature of interest with the sweeping range, while the other two features stay frozen at their mean
4. `hypothesis(w, X_pd, b)` → predict probability across this synthetic sweep

Because only one input is changing at a time (the others are held constant), the resulting curve is a clean, smooth S-shape — showing exactly how that one feature affects the predicted probability, independent of the other two. This is repeated for `hours_studied`, `attendance`, and `sleep_hours`.

---

## 🔍 Final Understanding

This model:
- standardized 3 input features
- learned weights `w` and bias `b` via batch gradient descent, minimizing cross-entropy loss
- correctly implements the sigmoid hypothesis, cross-entropy loss, and gradient formulas derived analytically in the theory notes
- the original zigzag in the per-feature plots was a **visualization artifact**, not a model bug — fixed by plotting against the logit `z` and via partial dependence plots

### Possible next steps / improvements:
- Add **regularization** (L1/L2) to the loss and gradient functions to control overfitting
- Track training and validation loss separately to check for overfitting
- Compute a **confusion matrix**, precision, recall, and F1 score on a held-out test set to evaluate the classifier properly (accuracy alone can be misleading on imbalanced data)
- Try different learning rates / epoch counts and plot the loss curve over iterations

---
