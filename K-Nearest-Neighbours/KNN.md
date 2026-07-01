# 📘 K-Nearest Neighbours (KNN)

---

## 📌 What is K-Nearest Neighbours?

KNN is a **lazy learning algorithm** that predicts based on the **K closest datapoints** to a new point.

- No training phase
- Stores the entire dataset
- Uses distance **at prediction time**, not training time

Because there's no real "learning" step — the model just memorizes the data and does all its work when a prediction is requested — it's called *lazy*.

---

## 🧠 Core Intuition

Whenever a new datapoint arrives, KNN:

1. Computes the distance from the new point to **every** point in the dataset
2. Finds the **K closest** points (the "neighbors")
3. Lets those neighbors **vote**:
   - **Classification** → majority class wins
   - **Regression** → average value of neighbors

The core idea: *similar points tend to be near each other, so a point's neighbors are a good guide to its label.*

---

## 📏 Distance Metrics

KNN depends **entirely** on distance — the metric you choose changes how "closeness" is defined.

### (i) Euclidean Distance

    d = sqrt((x1 - x2)^2 + (y1 - y2)^2)

Just like the standard distance formula — straight-line ("as the crow flies") distance between two points. This is the metric used in your `euclidean()` function.

### (ii) Manhattan Distance

    d = |x1 - x2| + |y1 - y2|

Distance measured by moving **only horizontally and vertically** — like walking city blocks instead of cutting diagonally.

### (iii) Minkowski Distance

    d = (|x1 - x2|^p + |y1 - y2|^p) ^ (1/p)

A generalized distance formula:
- **p = 1** → reduces to Manhattan distance
- **p = 2** → reduces to Euclidean distance

### (iv) Cosine Distance

    cosine similarity = (x · y) / (||x|| ||y||)
    cosine distance = 1 - cosine similarity

Measures the **angle** between two vectors rather than their straight-line distance — magnitude doesn't matter, only direction.

---

## ⚖️ Weighted KNN

In ordinary KNN, every neighbor gets **one equal vote**. But this can be misleading.

**Example:** Suppose K = 5, and the new point's neighbors are:

| Neighbor | Class | Distance |
|----------|-------|----------|
| A        | ●     | 0.2      |
| B        | ●     | 2.5      |
| C        | ●     | 2.8      |
| D        | ●     | 3.0      |
| E        | ●     | 3.2      |

Ordinary KNN counts votes as: class ● = 1, class ● (different shade) = 4 → majority wins, even though the single closest neighbor (A, distance 0.2) is far more relevant than the four distant ones.

**Weighted KNN** fixes this: instead of giving every neighbor one equal vote, each neighbor's vote is weighted **proportional to how close it is**. Closer neighbors should influence the prediction more than distant ones.

### The weighting function

    wi = 1 / di

where `di` is the distance of the i-th neighbor.

**Problem:** if `di = 0` (the point is exactly on top of a training point), this becomes `1/0` — undefined.

**Fix:** add a tiny constant `e` (epsilon) to the denominator so it's never zero:

    wi = 1 / (di + e)        where e = 10^-8 (a tiny number)

Sometimes an alternative is used instead:

    wi = 1 / di^2

This weights closer points even more aggressively than the plain inverse.

> Note: your current implementation uses **plain majority voting** (unweighted), not weighted voting — this section is background theory that explains a natural next improvement to the model.

---

# 📦 CODE WALKTHROUGH

## 🧩 Cell 1: Imports

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

### Explanation:

- NumPy is imported for numerical operations (distance calculations)
- Pandas is imported to load and handle the dataset
- `train_test_split` splits the data into training and testing sets

---

## 🧩 Cell 2: Load Dataset

```python
df = pd.read_csv("dataset.csv")
X = df[["study_hours", "attendance", "sleep_hours"]].values
y = df["label"].values

X.shape, y.shape
```

### What's happening:

- Loads `dataset.csv` into a DataFrame
- `X` is built from three feature columns: `study_hours`, `attendance`, `sleep_hours`
- `y` is the target column, `label`
- Output confirms shape: **138 samples, 3 features**

---

## 🧩 Cell 3: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Explanation:

- Splits data into **80% training** and **20% testing**
- `X_train`/`y_train` → used to store as the "memory" for KNN (remember: no real training happens)
- `X_test`/`y_test` → used to evaluate predictions

---

## 🧩 Cell 4: Euclidean Distance Function

```python
def euclidean(x, y):
    return np.sqrt(np.sum((x-y) ** 2))
```

### Explanation:

This implements the Euclidean distance formula directly:

    d = sqrt(Σ(xi - yi)^2)

- `(x - y) ** 2` → squares the difference for each feature
- `np.sum(...)` → adds them all up
- `np.sqrt(...)` → takes the square root

This is the "closeness" measure KNN uses to decide which points are neighbors.

---

## 🧩 Cell 5: KNN Prediction Function

```python
def knn_predict(X_train, y_train, test_point, k=3):
    distances = []

    for i in range(len(y_train)):
        distance = euclidean(test_point, X_train[i])
        distances.append((distance, y_train[i]))

    distances.sort(key = lambda x: x[0])

    neighbors = distances[:k]

    labels = []

    for _, label in neighbors:
        labels.append(label)

    votes = {}

    for _, label in neighbors:
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1

    prediction = max(votes, key=votes.get)
    return prediction
```

### This is the heart of your KNN implementation.

### Step-by-step breakdown:

#### 1. Compute distance to every training point

```python
for i in range(len(y_train)):
    distance = euclidean(test_point, X_train[i])
    distances.append((distance, y_train[i]))
```

For each point in the training set, calculate the Euclidean distance from the new `test_point`, and store it as a `(distance, label)` pair.

#### 2. Sort by distance

```python
distances.sort(key = lambda x: x[0])
```

Sorts all `(distance, label)` pairs from closest to farthest, using the distance value as the sort key.

#### 3. Take the K closest neighbors

```python
neighbors = distances[:k]
```

Slices out just the top `k` closest points — these are the "voters."

#### 4. Collect the vote

```python
votes = {}

for _, label in neighbors:
    if label in votes:
        votes[label] += 1
    else:
        votes[label] = 1
```

Counts how many times each class label appears among the neighbors — this is **plain majority voting**, where every neighbor's vote counts equally (unlike Weighted KNN, described above).

#### 5. Pick the winner

```python
prediction = max(votes, key=votes.get)
```

Returns whichever label got the most votes.

---

## 🧩 Cell 6: Predict on Test Set

```python
predictions = []

for point in X_test:
    pred = knn_predict(X_train, y_train, point, k=3)
    predictions.append(pred)
```

### Explanation:

Loops through every point in `X_test` and predicts its class using `knn_predict`, with `k=3` neighbors consulted each time. All predictions are collected into a list.

---

## 🧩 Cell 7: Compare Predictions vs Actual

```python
accuracy = np.mean(predictions == y_test)
print("Predictions:", predictions)
print("Actual:", y_test)
```

### Explanation:

- Compares the predicted labels against the true labels element-wise
- `np.mean(...)` on a boolean array gives the **fraction of correct predictions** — this is accuracy

---

## 🧩 Cell 8: Print Accuracy

```python
print("Accuracy: ", accuracy)
```

### Result:

    Accuracy:  0.9285714285714286

The model correctly classified about **92.9%** of the test points using k=3 and plain Euclidean distance.

---

## 🔍 Final Understanding

This model:
- stored the entire training dataset (no actual "training")
- computed Euclidean distance from each test point to every training point
- selected the K nearest neighbors and let them vote by simple majority
- achieved ~92.9% accuracy on the test set

---
