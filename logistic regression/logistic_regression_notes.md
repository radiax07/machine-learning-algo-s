# 📘 Logistic Regression — Theory Notes (In-Depth)

---

## 📌 What is Logistic Regression?

**Example:**

| Study Hours | Result |
|---|---|
| 3 | Fail |
| 4 | Fail |
| 5 | Pass |
| 6 | Pass |
| 1 | Fail |

If you plot this, the result jumps sharply from 0 to 1 around a certain study-hour threshold — the shape looks like a stretched "S".

### Why linear regression fails here

Linear regression gives a **continuous** output (any real number), but here we need:
- Output of exactly **0 or 1**, or
- At least a **probability between 0 and 1**

A straight line can shoot past 1 or below 0, which makes no sense as a probability. So linear regression is the wrong tool for classification.

### What we actually want

- Output should be **between 0 and 1**
- Output should **represent a probability**
- The model should help us **separate classes clearly**

---

## 🧮 Building the Model

### Step 1 — Compute a linear combination

    z = wx + b     (linear, same as linear regression)

### Step 2 — Convert z into a probability (0 to 1)

For this, we use the **sigmoid function**:

    ŷ = σ(z) = 1 / (1 + e^(-z))

The sigmoid function **squashes** any real number into the range (0, 1), and it's what makes the graph smooth (an S-shaped curve) instead of a sharp jump.

**Example:** if ŷ = 0.8 → this means 80% chance of belonging to class 1.

---

## 🎯 From Probability to Class (Decision Rule)

To make a final decision, we convert the probability into a class using a threshold:

    ŷ = 1   if probability ≥ 0.5
    ŷ = 0   if probability < 0.5

### Decision Boundary

The **decision boundary** is the line (or surface, in higher dimensions) where the model switches its prediction from one class to another. On one side of this boundary, the model predicts class 0; on the other side, class 1.

---

## ❌ Loss in Classification

**Loss** = how wrong your model's prediction is, for a single data point.

In classification:
- The model **outputs a probability**
- But in reality, the true label is either **0** (false) or **1** (true) — never in between

So our loss function must:
- **Penalize** wrong, confident predictions **heavily**
- **Reward** correct, confident predictions

This leads us to **Cross-Entropy Loss**.

---

## 🎲 Cross-Entropy Loss — Derivation from Probability

Cross-entropy loss measures **how far the predicted probability distribution is from the actual distribution**.

We define:

    ŷ = P(y = 1 | x)

This means: *assuming x is known, what is the probability that y = 1?*

We want to **maximize** this probability, P(y|x), because a good model should assign high probability to the correct outcome.

### Case 1: y = 1

    P(y|x) = ŷ

### Case 2: y = 0

    P(y|x) = 1 - ŷ

### Combining both cases into one formula

    P(y|x) = ŷ^y * (1 - ŷ)^(1-y)

This clever trick works because:
- When y = 1, the second term becomes (1-ŷ)^0 = 1, leaving just ŷ
- When y = 0, the first term becomes ŷ^0 = 1, leaving just (1-ŷ)

### Total probability of the whole dataset

Assuming the data points are independent, the total probability is the **product of individual probabilities**:

    P(all data) = P₁ · P₂ · P₃ · ... · Pₙ = Π (i=1 to m) ŷᵢ^yᵢ * (1-ŷᵢ)^(1-yᵢ)

**Example:**

| i | y | ŷ |
|---|---|---|
| 1 | 1 | 0.9 |
| 2 | 0 | 0.2 |
| 3 | 1 | 0.8 |

    Total likelihood = 0.9 × (1 - 0.2) × 0.8 = 0.576

**Higher likelihood = better model.**

### Why we take the log

    log(Π Pᵢ) = Σ log(Pᵢ)

Taking the log of a product turns it into a **sum**, which:
- **Prevents underflow** (multiplying many small probabilities together can produce a number too small for a computer to represent)
- **Makes the math easier** (derivatives of sums are much simpler than derivatives of products)

### Log-Likelihood

    L = Σ (i=1 to m) [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]

Instead of **maximizing** the log-likelihood, it's more convenient to **minimize** its negative — this suits well with **gradient descent**, which is built to minimize a function.

### Cross-Entropy Loss (Cost Function)

    J = -(1/n) Σ (i=1 to m) [ yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ) ]

---

## ❓ Why Log? (Intuition)

Log makes **confident wrong predictions** produce a **massive loss**.

    log(0.01) → very large negative → huge loss

### Case 1: y = 1

    L = -log(ŷ)

| ŷ | Loss |
|---|---|
| 0.9 | small |
| 0.5 | medium |
| 0.1 | huge |

### Case 2: y = 0

    L = -log(1 - ŷ)

| ŷ | Loss |
|---|---|
| 0.1 | small |
| 0.5 | medium |
| 0.9 | huge |

👉 The further the prediction is from the true label (while still being confident), the more severely it's punished.

---

## ⚙️ Optimization in Logistic Regression

### Model

    ŷ = σ(z) = σ(wx + b)
    σ(z) = 1 / (1 + e^(-z))

### Loss Function (Cross-Entropy)

    J(w,b) = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

**Goal:** find w, b such that this loss is minimized → **Gradient Descent**.

### Rewriting J(w,b) in terms of z (substituting ŷ = 1/(1+e^-z))

    J(w,b) = -(1/m) Σ [ y log(1/(1+e^-z)) + (1-y) log(e^-z/(1+e^-z)) ]

Simplifying step by step using log rules:

    log(1/(1+e^-z)) = log(1) - log(1+e^-z) = -log(1+e^-z)

    log(e^-z/(1+e^-z)) = log(e^-z) - log(1+e^-z) = -z - log(1+e^-z)

Substituting back:

    J(w,b) = -(1/m) Σ [ -y·log(1+e^-z) + (1-y)(-z - log(1+e^-z)) ]

    = -(1/m) Σ [ -y·log(1+e^-z) - z - log(1+e^-z) + zy + y·log(1+e^-z) ]

    = -(1/m) Σ [ -log(1+e^-z) - z + zy ]

    = (1/m) Σ [ log(1+e^-z) + z(1-y) ]

This rewritten form is more convenient for taking derivatives.

---

## 🧮 Deriving the Gradient (∂J/∂z)

Starting from:

    J(w,b) = (1/m) Σ [ log(1+e^-z) + z(1-y) ]

Differentiate with respect to z:

    ∂J/∂z = (1/m) Σ [ (-e^-z)/(1+e^-z) + (1-y) ]

Multiply numerator and denominator by e^z:

    (-e^-z)/(1+e^-z) × (e^z/e^z) = -1/(1+e^z)

So:

    ∂J/∂z = (1/m) Σ [ -1/(1+e^z) + (1-y) ]

### Key identity: relating 1/(1+e^z) to σ(z)

    σ(z) = 1/(1+e^-z) = e^z/(1+e^z)

    1 - σ(z) = 1 - e^z/(1+e^z) = 1/(1+e^z)

So:

    1/(1+e^z) = 1 - σ(z)

Substituting:

    ∂J/∂z = (1/m) Σ [ -(1-σ) + (1-y) ]

    = (1/m) Σ [ -1 + σ + 1 - y ]

    = (1/m) Σ [ σ(z) - y ]

    = (1/m) Σ [ ŷ - y ]

### This is a beautifully simple result:

    ∂J/∂z = (1/m) Σ (ŷᵢ - yᵢ)

---

## 🧮 Deriving Gradients for w and b (Chain Rule)

We know:

    z = wx + b   →   ∂z/∂w = x

Using the chain rule:

    ∂J/∂w = (1/m) Σ [ xᵢ (ŷᵢ - yᵢ) ]

    ∂J/∂b = (1/m) Σ [ ŷᵢ - yᵢ ]

### Gradient Descent Update Rule

    w = w - learning_rate × ∂J/∂w
    b = b - learning_rate × ∂J/∂b

**Goal:**
- Minimize w → converges to the best-fit weight
- Minimize b → converges to the best-fit bias

---

## 📊 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### (i) Accuracy

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

*Out of all predictions, how many were correct.*

### (ii) Precision

    Precision = TP / (TP + FP)

*Out of all predicted positives, how many were actually correct.*
> "Can I trust my positive predictions?"

### (iii) Recall

    Recall = TP / (TP + FN)

*Out of all actual positives, how many did we correctly catch.*
> "How many real positives did I NOT miss?"

### (iv) F1 Score

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

The **harmonic mean** of precision and recall.

> You can't maximize both Precision and Recall at the same time — there's a tradeoff.
- **High precision** → fewer false positives
- **High recall** → fewer false negatives (catches more actual positives, at the risk of more false alarms)

---

## 🛡️ Regularization

Regularization is a technique to **prevent your model from overfitting** by discouraging it from becoming too complex.

    Minimize error + Keep model simple

We **penalize large weights**:

    Loss = Error + λ · Penalty

Here, **λ (lambda)** is the **control knob** — it controls how much the model is allowed to overfit.

| λ value | Effect |
|---|---|
| λ = 0 | No regularization |
| λ = small | Little control |
| λ = large | Strong control |

### Why regularize? — The problem without it

We had, for linear/logistic regression:

    Minimize: Σ (y - ŷ)²   → MSE

**Problem:**
- Model can choose very large weights
- Fits noise in the data
- Becomes unstable

We want to fit the data **BUT** keep weights small.

---

### Types of Regularization

#### L2 (Ridge)

    Penalty = λ Σ w²

- Makes weights **small** (shrinks them, but rarely exactly zero)

**Ridge Loss Function:**

    J = (1/n) Σ (y - ŷ)² + λ Σ w²

    ∂J/∂w = (1/n) Σ (y - ŷ)² · x + 2λw

> The "2" in front of λ can be ignored — it's just a scaling factor (often absorbed into λ itself).

#### L1 (Lasso)

    Penalty = λ Σ |w|

- Makes some weights **exactly zero** — effectively performing feature selection.

**Lasso Loss Function:**

    J = (1/n) Σ (y - ŷ)² + λ Σ |w|

    ∂J/∂w = (1/n) Σ (y - ŷ)² · x + λ · sign(w)

Where the **sign function** is:

    sign(w) =  +1   if w > 0
                0   if w = 0
               -1   if w < 0

**Example:**
- sign(5) = +1
- sign(-3) = -1
- sign(0) = 0

### Ridge vs Lasso — Summary

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Penalty | λ Σ w² | λ Σ \|w\| |
| Effect on weights | Makes weights small | Makes some weights exactly zero |
| Use case | General shrinkage | Feature selection |

---
