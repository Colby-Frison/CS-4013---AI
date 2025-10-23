# Module 6: Traditional ML Methods

## Table of Contents
1. [Regression](#regression)
2. [Clustering and Nearest Neighbors](#clustering-and-nearest-neighbors)
3. [Decision Trees](#decision-trees)
4. [Ensembles](#ensembles)
5. [Ethics](#ethics)

---

## Regression

Regression predicts a continuous numerical value from input features. One of the most fundamental ML tasks.

### Linear Regression

The simplest and most widely used regression method.

#### The Model

**Hypothesis**: Output is a linear function of inputs.

$$h(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T x$$

Where:
- $x = [1, x_1, x_2, ..., x_n]^T$: Feature vector (1 for bias term)
- $\theta = [\theta_0, \theta_1, ..., \theta_n]^T$: Parameters (weights)
- $\theta_0$: Bias/intercept
- $\theta_i$: Weight for feature $x_i$

**Geometric Interpretation**: Hyperplane in feature space.

#### Training: Least Squares

**Objective**: Minimize sum of squared errors.

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$: Number of training examples
- $(x^{(i)}, y^{(i)})$: $i$-th training example

**Why Squared Error?**:
- Penalizes large errors more heavily
- Differentiable (easy to optimize)
- Has closed-form solution
- Assumes Gaussian noise (maximum likelihood under this assumption)

#### Closed-Form Solution: Normal Equation

For linear regression, can solve analytically:

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- $X$: Design matrix ($m \times n$), each row is example
- $y$: Vector of target values ($m \times 1$)

**Advantages**:
- Exact solution (no iterations)
- No learning rate to tune

**Disadvantages**:
- Matrix inversion is $O(n^3)$ (slow for large $n$)
- Doesn't work if $X^T X$ is singular
- Doesn't extend to other models

#### Gradient Descent Approach

**Alternative**: Iterative optimization.

**Update Rule**:
$$\theta_j \leftarrow \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

For all parameters $j$ simultaneously.

**Advantages**:
- Scales to large datasets
- Works even if $X^T X$ is singular
- Generalizes to other models (logistic regression, neural networks)

#### Example: House Price Prediction

**Features**:
- $x_1$: Square footage
- $x_2$: Number of bedrooms
- $x_3$: Age of house
- $x_4$: Distance to city center

**Model**:
$$\text{Price} = \theta_0 + \theta_1 \cdot \text{sqft} + \theta_2 \cdot \text{bedrooms} + \theta_3 \cdot \text{age} + \theta_4 \cdot \text{distance}$$

**Training**: Fit $\theta$ to historical sales data.

**Prediction**: Apply model to new houses.

### Polynomial Regression

**Problem**: What if relationship isn't linear?

**Solution**: Use polynomial features.

**Example**: Instead of $h(x) = \theta_0 + \theta_1 x$

Use: $h(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3$

**Implementation**:
- Create new features: $x_1 = x, x_2 = x^2, x_3 = x^3$
- Apply linear regression to these features
- Still linear in parameters!

**Caution**: High-degree polynomials can overfit.

### Feature Normalization

**Problem**: Features have different scales (e.g., age in years vs. price in dollars).

**Consequence**: Gradient descent converges slowly, some weights dominate.

**Solution**: Normalize features to similar ranges.

**Methods**:

**1. Standardization (Z-score normalization)**:
$$x' = \frac{x - \mu}{\sigma}$$
- Mean = 0, standard deviation = 1
- Preserves outliers

**2. Min-Max Scaling**:
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$
- Range: [0, 1]
- Sensitive to outliers

**Always apply same transformation to training and test data!**

### Regularization in Regression

Prevent overfitting by penalizing large weights.

#### Ridge Regression (L2)

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$

**Effect**:
- Shrinks all weights toward zero
- Smooth model
- All features retained

**When to Use**: Many correlated features, want smooth predictions.

#### Lasso Regression (L1)

$$L(\theta) = \frac{1}{2m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^n |\theta_j|$$

**Effect**:
- Sparse solutions (many weights exactly zero)
- Automatic feature selection
- Interpretable

**When to Use**: Many irrelevant features, want interpretable model.

### Logistic Regression (Classification)

Despite the name, logistic regression is for **classification**, not regression.

#### The Model

**Goal**: Predict probability that example belongs to positive class.

**Hypothesis**:
$$h(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the **sigmoid (logistic) function**.

**Properties of Sigmoid**:
- Maps any real number to (0, 1)
- $\sigma(0) = 0.5$
- Smooth, differentiable
- $\sigma(-z) = 1 - \sigma(z)$

**Interpretation**: $h(x)$ is probability that $y=1$ given $x$.

**Decision Rule**: 
- Predict $y=1$ if $h(x) \geq 0.5$ (i.e., $\theta^T x \geq 0$)
- Predict $y=0$ otherwise

#### Training: Maximum Likelihood

**Loss Function** (Cross-Entropy):
$$L(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log h(x^{(i)}) + (1-y^{(i)}) \log(1 - h(x^{(i)}))]$$

**Why This Loss?**:
- Derived from maximum likelihood
- Convex (has unique global minimum)
- Heavily penalizes confident wrong predictions

**Optimization**: Gradient descent (no closed-form solution).

**Update Rule**:
$$\theta_j \leftarrow \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

(Looks like linear regression, but $h$ is different!)

#### Multi-Class Classification

**One-vs-All (One-vs-Rest)**:
- Train $k$ binary classifiers (one per class)
- Classifier $i$ distinguishes class $i$ from all others
- Predict class with highest probability

**Softmax Regression (Multinomial Logistic Regression)**:
- Direct multi-class generalization
- Outputs probability distribution over classes
- $P(y=k|x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^K e^{\theta_j^T x}}$

### When to Use Regression

**Linear/Logistic Regression**:
- **Advantages**: Simple, fast, interpretable, works well with limited data
- **Disadvantages**: Limited expressiveness (linear decision boundaries)
- **Best For**: When relationship is approximately linear, need interpretability, quick baseline

**Don't Use**:
- Highly non-linear relationships (without feature engineering)
- Complex interactions between features
- When interpretability not important and have sufficient data for complex models

---

## Clustering and Nearest Neighbors

### K-Means Clustering

Unsupervised learning: Group similar examples together.

#### Algorithm

**Goal**: Partition $n$ examples into $k$ clusters.

```
function K-MEANS(data, k):
    # Initialization
    centroids = randomly select k examples as initial centroids
    
    repeat:
        # Assignment Step
        for each example x:
            assign x to nearest centroid
        
        # Update Step
        for each cluster:
            recompute centroid as mean of assigned examples
    
    until convergence (centroids don't change)
    
    return cluster assignments
```

#### Detailed Steps

**1. Initialization**:
- Choose $k$ (number of clusters)
- Randomly initialize $k$ cluster centroids $\mu_1, ..., \mu_k$

**2. Assignment**:
- For each example $x^{(i)}$, assign to nearest centroid:
  $$c^{(i)} = \arg\min_j ||x^{(i)} - \mu_j||^2$$

**3. Update**:
- For each cluster $j$, recompute centroid:
  $$\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x^{(i)}$$
  (Mean of all examples in cluster $j$)

**4. Repeat 2-3 until convergence**.

#### Objective Function

**Minimizes within-cluster sum of squares**:
$$J = \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$$

- Assignment step: Minimize $J$ w.r.t. cluster assignments $c$
- Update step: Minimize $J$ w.r.t. centroids $\mu$
- Algorithm guaranteed to converge (to local optimum)

#### Choosing K

**Problem**: How many clusters?

**Elbow Method**:
1. Run K-means for different values of $k$
2. Plot $k$ vs. objective value $J$
3. Look for "elbow" where decrease slows

**Silhouette Score**:
- Measures how similar example is to its cluster vs. other clusters
- Range: [-1, 1]
- Higher is better

**Domain Knowledge**: Sometimes $k$ is determined by application.

#### Limitations

**Local Optima**: Sensitive to initialization.
- **Solution**: Run multiple times with different initializations, keep best result.

**Spherical Clusters**: Assumes clusters are roughly spherical and similar size.
- Fails for elongated or irregularly-shaped clusters

**Fixed K**: Must specify number of clusters in advance.

**Outliers**: Sensitive to outliers (they pull centroids).

#### When to Use K-Means

**Advantages**: Simple, fast, scalable.

**Best For**:
- Well-separated, spherical clusters
- Large datasets (efficient)
- Need fast clustering

**Alternatives**:
- **DBSCAN**: Arbitrary-shaped clusters, handles noise
- **Hierarchical Clustering**: Creates tree of clusters, don't need to specify $k$
- **Gaussian Mixture Models**: Probabilistic, allows soft assignments

### K-Nearest Neighbors (K-NN)

Supervised learning: Classify based on nearest training examples.

#### Algorithm

**Training**: Store all training examples (lazy learning—no explicit training phase).

**Prediction**:
```
function K-NN-PREDICT(x, training_data, k):
    # Find k nearest neighbors
    neighbors = k closest training examples to x
    
    # Classification
    return majority class among neighbors
    
    # Regression
    return average value among neighbors
```

**Distance Metric**: Usually Euclidean distance:
$$d(x, x') = ||x - x'|| = \sqrt{\sum_i (x_i - x_i')^2}$$

#### Choosing K

**Small K** (e.g., $k=1$):
- Sensitive to noise
- Complex decision boundaries
- Overfitting

**Large K**:
- Smooth decision boundaries
- More robust to noise
- Underfitting (if too large)

**Rule of Thumb**: 
- Try $k = \sqrt{n}$ as starting point
- Use cross-validation to select best $k$
- Use odd $k$ for binary classification (avoid ties)

#### Example

**Classification**:
```
Training Data:
  (1, 1) → Class A
  (2, 2) → Class A
  (3, 1) → Class B
  (2, 3) → Class B

Query: (2, 1.5)
K = 3

Distances:
  to (1, 1): 1.12 ← 
  to (2, 2): 0.50 ← 
  to (3, 1): 1.12 ← 
  to (2, 3): 1.50

3 Nearest: (2, 2) → A, (1, 1) → A, (3, 1) → B

Prediction: Class A (majority vote)
```

#### Distance-Weighted Voting

**Problem**: Far neighbors shouldn't influence as much as close ones.

**Solution**: Weight votes by inverse distance.

$$\text{Weight} = \frac{1}{d(x, x_i)^2 + \epsilon}$$

($\epsilon$ prevents division by zero)

#### Properties

**Decision Boundary**: 
- Non-linear, arbitrary complexity
- Can fit any pattern given enough data

**Bayes Error**: As $n \to \infty$ and $k \to \infty$ (with $k/n \to 0$), K-NN approaches Bayes optimal classifier.

#### Computational Complexity

**Training**: $O(1)$ (just store data)
**Prediction**: $O(mn)$ where $m$ = test examples, $n$ = training examples
- Must compute distance to all training examples
- Slow for large datasets

**Optimization**:
- **K-D Trees**: $O(\log n)$ search (works well in low dimensions)
- **Ball Trees**: Better for higher dimensions
- **Approximate Nearest Neighbors**: Trade accuracy for speed

#### Curse of Dimensionality

**Problem**: In high dimensions, all points are far apart.

**Consequence**: "Nearest" neighbors aren't very near.
- Distances become less meaningful
- Need exponentially more data

**Mitigation**:
- Dimensionality reduction (PCA, etc.)
- Feature selection
- Distance metric learning

#### When to Use K-NN

**Advantages**:
- Simple, intuitive
- No training phase
- Non-linear decision boundaries
- Multi-class naturally handled

**Disadvantages**:
- Slow prediction
- Memory intensive (store all training data)
- Sensitive to irrelevant features and scale
- Curse of dimensionality

**Best For**:
- Small to medium datasets
- Low to moderate dimensions
- Irregular decision boundaries
- Need interpretability (can show nearest neighbors as explanation)

---

## Decision Trees

Learn hierarchy of if-then rules to make decisions.

### Structure

**Decision Tree Components**:

**Internal Node**: Test on an attribute
- Example: "Age < 30?"

**Branch**: Outcome of test
- Example: "Yes" or "No"

**Leaf Node**: Prediction
- Classification: Class label
- Regression: Numerical value

**Example**:
```
           Age < 30?
          /         \
        Yes          No
        /              \
  Income < 50k?    Education?
   /        \       /    |    \
 Yes        No   HS  BS   PhD
  |          |    |   |    |
Deny      Approve |  |    |
              Deny |  Approve
                Approve
```

### Building Decision Trees

#### ID3/C4.5/CART Algorithms

**Basic Idea**: Recursively split data to maximize "purity" of resulting subsets.

```
function BUILD-TREE(examples, attributes):
    if all examples have same label:
        return Leaf(label)
    
    if attributes is empty:
        return Leaf(majority label)
    
    # Choose best attribute to split on
    best_attr = SELECT-ATTRIBUTE(examples, attributes)
    
    tree = Node(best_attr)
    
    for each value v of best_attr:
        subset = examples where best_attr = v
        if subset is empty:
            tree.add_child(v, Leaf(majority label))
        else:
            tree.add_child(v, BUILD-TREE(subset, attributes - {best_attr}))
    
    return tree
```

**Key Question**: How to select best attribute?

### Attribute Selection

**Goal**: Choose attribute that best separates classes.

#### Entropy

**Measure of impurity/disorder**:

$$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$

Where:
- $S$: Set of examples
- $c$: Number of classes
- $p_i$: Proportion of examples in class $i$

**Interpretation**:
- $H = 0$: Pure (all examples same class)
- $H = 1$ (binary): Maximum impurity (50-50 split)
- Higher entropy = more disorder

**Example**:
- 9 positive, 5 negative examples:
  $$H = -\frac{9}{14}\log_2(\frac{9}{14}) - \frac{5}{14}\log_2(\frac{5}{14}) = 0.940$$

- 10 positive, 0 negative:
  $$H = -1\log_2(1) - 0\log_2(0) = 0$$ (pure)

#### Information Gain

**Reduction in entropy from splitting on attribute**:

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $A$: Attribute
- $S_v$: Subset of $S$ where $A = v$

**Select attribute with highest information gain**.

**Example**:
```
Dataset: 14 examples (9 positive, 5 negative)
Attribute "Outlook": {Sunny, Overcast, Rainy}

Sunny: 2 positive, 3 negative → H = 0.971
Overcast: 4 positive, 0 negative → H = 0
Rainy: 3 positive, 2 negative → H = 0.971

IG = 0.940 - (5/14 × 0.971 + 4/14 × 0 + 5/14 × 0.971)
   = 0.940 - 0.693 = 0.247
```

#### Gini Impurity

**Alternative to entropy** (used by CART):

$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

**Interpretation**: Probability of misclassifying randomly chosen example.

**Similar to entropy but computationally simpler**.

### Continuous Attributes

**Problem**: What if attribute is continuous (e.g., age, income)?

**Solution**: Create binary split.

**Process**:
1. Sort examples by attribute value
2. Consider split points between adjacent values
3. Evaluate each split point
4. Choose best split

**Example**: Age = {22, 25, 27, 30, 35, 40}
- Try splits: Age < 23.5, Age < 26, Age < 28.5, etc.
- Choose split with highest information gain

### Pruning

**Problem**: Full tree overfits training data.

**Solution**: Prune tree to reduce complexity.

#### Pre-Pruning (Early Stopping)

Stop growing tree early:
- Maximum depth reached
- Too few examples to split
- No significant information gain
- Statistical test (e.g., chi-square) shows split not significant

**Advantage**: Fast, prevents overfitting during construction.
**Disadvantage**: May stop too early (miss good splits).

#### Post-Pruning

Build full tree, then prune:
1. Build complete tree
2. For each node, consider removing subtree
3. Evaluate on validation set
4. Keep pruned version if improves validation performance

**Advantage**: More accurate (considers full tree).
**Disadvantage**: Slower (build full tree first).

### Regression Trees

**Adaptation for regression**:
- Leaf nodes contain numerical values (not class labels)
- Split criterion: Minimize variance (instead of entropy)
- Prediction: Mean value of examples in leaf

**Variance Reduction**:
$$VR(S, A) = Var(S) - \sum_{v} \frac{|S_v|}{|S|} Var(S_v)$$

### Advantages of Decision Trees

1. **Interpretable**: Easy to understand and explain
2. **Handles Mixed Data**: Categorical and numerical attributes
3. **Non-Linear**: Can capture complex patterns
4. **No Feature Scaling**: Don't need to normalize features
5. **Feature Importance**: Can identify most informative features
6. **Missing Values**: Can handle with surrogate splits

### Limitations

1. **Overfitting**: Can create overly complex trees
2. **Instability**: Small changes in data → very different tree
3. **Greedy**: Locally optimal splits (not globally optimal)
4. **Biased**: Favors attributes with many values
5. **Axis-Aligned**: Splits perpendicular to axes (can't learn diagonal boundaries easily)

### When to Use Decision Trees

**Best For**:
- Need interpretability
- Mixed data types
- Non-linear relationships
- Feature importance analysis
- Quick baseline

**Consider Alternatives**:
- If need high accuracy (use ensembles—next section)
- If data is high-dimensional and sparse (linear models may work better)
- If interpretability not critical (neural networks may outperform)

---

## Ensembles

Combine multiple models for better performance. Often the most effective ML technique.

### Why Ensembles Work

**Key Idea**: "Wisdom of crowds"—combining multiple imperfect models can be better than any single model.

**Conditions**:
1. **Diverse**: Models make different errors
2. **Better than Random**: Each model better than guessing

**If models are independent and have accuracy > 0.5, ensemble accuracy increases with size**.

**Example**: 3 classifiers, each 70% accurate (independent):
- All correct: 34.3%
- Majority correct: 78.4%
- Ensemble (majority vote): 78.4% accuracy

### Bagging (Bootstrap Aggregating)

**Idea**: Train multiple models on different random samples of data, average predictions.

#### Algorithm

```
function BAGGING(data, base_learner, T):
    models = []
    
    for t = 1 to T:
        # Bootstrap sample (sample with replacement)
        sample = randomly draw n examples from data (with replacement)
        
        # Train model on sample
        model_t = train base_learner on sample
        models.append(model_t)
    
    return models

function PREDICT(models, x):
    # Regression: average predictions
    # Classification: majority vote
    return aggregate(model.predict(x) for model in models)
```

**Bootstrap Sample**: Randomly draw $n$ examples from dataset of size $n$ (with replacement).
- On average, each sample contains ~63% unique examples
- Each model sees slightly different data → diversity

#### Random Forests

**Enhanced bagging for decision trees**.

**Additional Randomness**:
- When splitting node, consider only random subset of features (not all features)
- Typical: $\sqrt{n}$ features for classification, $n/3$ for regression
- Forces trees to be more diverse

**Algorithm**:
```
function RANDOM-FOREST(data, num_trees):
    forest = []
    
    for i = 1 to num_trees:
        # Bootstrap sample
        sample = bootstrap(data)
        
        # Build tree with random feature subsets at each split
        tree = build_tree_with_random_features(sample)
        forest.append(tree)
    
    return forest
```

**Advantages**:
- Very high accuracy
- Robust to overfitting
- Handles large datasets and high dimensions
- Feature importance estimates
- Parallelizable

**Disadvantages**:
- Less interpretable than single tree
- Slower prediction than single model
- More memory (stores multiple trees)

**State-of-the-Art**: Random Forests consistently perform well across many problems.

### Boosting

**Idea**: Train models sequentially, each focusing on examples previous models got wrong.

**Key Difference from Bagging**: 
- Bagging: Parallel, independent models
- Boosting: Sequential, dependent models

#### AdaBoost (Adaptive Boosting)

**Algorithm**:
```
function ADABOOST(data, base_learner, T):
    # Initialize weights uniformly
    w[i] = 1/n for all examples i
    
    models = []
    alphas = []
    
    for t = 1 to T:
        # Train model on weighted data
        model_t = train base_learner with weights w
        
        # Compute error
        error = sum of w[i] for misclassified examples i
        
        # Compute model weight
        alpha_t = 0.5 × log((1 - error) / error)
        
        # Update example weights
        for each example i:
            if correctly classified:
                w[i] = w[i] × exp(-alpha_t)
            else:
                w[i] = w[i] × exp(alpha_t)
        
        # Normalize weights
        w = w / sum(w)
        
        models.append(model_t)
        alphas.append(alpha_t)
    
    return models, alphas

function PREDICT(models, alphas, x):
    # Weighted majority vote
    return sign(sum(alpha_t × model_t.predict(x)))
```

**Key Steps**:
1. Weight examples (initially uniform)
2. Train model on weighted data
3. Increase weights of misclassified examples
4. Repeat—next model focuses on hard examples

**Advantages**:
- Often more accurate than bagging
- Adaptive to difficult examples
- Can use simple base learners (weak learners)

**Disadvantages**:
- More prone to overfitting (especially with noise)
- Sensitive to outliers (they get high weights)
- Sequential (can't parallelize)

#### Gradient Boosting

**Modern, powerful boosting technique**.

**Idea**: Each model fits the residuals (errors) of previous models.

**Algorithm**:
```
function GRADIENT-BOOSTING(data, loss, T, learning_rate):
    # Initialize with simple model (e.g., mean)
    F_0(x) = initial_prediction
    
    for t = 1 to T:
        # Compute residuals (negative gradient of loss)
        r[i] = -∂loss(y[i], F_{t-1}(x[i])) / ∂F_{t-1}(x[i])
        
        # Fit model to residuals
        h_t = train base_learner to predict r
        
        # Update ensemble
        F_t(x) = F_{t-1}(x) + learning_rate × h_t(x)
    
    return F_T
```

**Key Insight**: Adding model that predicts residuals is equivalent to gradient descent in function space.

**Hyperparameters**:
- **Learning Rate**: How much each tree contributes (smaller = more robust, need more trees)
- **Number of Trees**: More trees = better fit (but slower, can overfit)
- **Tree Depth**: Typically shallow trees (2-8 levels)

**Variants**:
- **XGBoost**: Optimized implementation, regularization, handles missing values
- **LightGBM**: Fast, efficient, good for large datasets
- **CatBoost**: Handles categorical features well

**State-of-the-Art**: Gradient boosting (especially XGBoost, LightGBM) wins many Kaggle competitions.

### Stacking

**Idea**: Use model to combine predictions of other models.

**Algorithm**:
```
function STACKING(data, base_learners, meta_learner):
    # Split data
    train, validation = split(data)
    
    # Train base models
    base_models = []
    for learner in base_learners:
        model = train learner on train
        base_models.append(model)
    
    # Generate meta-features
    meta_train = [model.predict(x) for model in base_models for x in validation]
    
    # Train meta-model
    meta_model = train meta_learner on meta_train
    
    return base_models, meta_model

function PREDICT(base_models, meta_model, x):
    # Get predictions from base models
    base_preds = [model.predict(x) for model in base_models]
    
    # Meta-model combines base predictions
    return meta_model.predict(base_preds)
```

**Advantages**:
- Can combine diverse models (trees, linear models, neural networks)
- Often achieves best performance

**Disadvantages**:
- Complex
- Risk of overfitting (meta-model learns validation set)
- Computationally expensive

### When to Use Ensembles

**Random Forests**:
- General-purpose, first try for many problems
- Robust, easy to use
- Want feature importance

**Gradient Boosting**:
- Need maximum accuracy
- Structured/tabular data
- Willing to tune hyperparameters

**AdaBoost**:
- Want simple ensemble
- Weak base learners
- Less common now (gradient boosting usually better)

**Stacking**:
- Competitions, maximum performance
- Have diverse strong models
- Computational resources available

**Don't Use Ensembles**:
- Need interpretability (single tree better)
- Computational constraints (single model faster)
- Very small datasets (ensembles may overfit)

---

## Ethics

Traditional ML methods raise ethical concerns, especially when applied to high-stakes decisions.

### Interpretability vs. Accuracy Trade-off

**Issue**: Most accurate models are often least interpretable.

**Accuracy Hierarchy** (approximate):
- Single decision tree < Random forest < Gradient boosting < Neural networks

**Interpretability Hierarchy** (approximate):
- Neural networks < Gradient boosting < Random forest < Single decision tree

**Dilemma**: In high-stakes applications (medicine, criminal justice, lending), do we sacrifice accuracy for interpretability or vice versa?

**Perspectives**:
- **Interpretability Advocates**: "Black box models shouldn't make life-altering decisions"
- **Accuracy Advocates**: "More accurate models save more lives/reduce more harm"

**Balance**: 
- Use interpretable models when possible
- For complex models, use interpretation techniques (SHAP, LIME)
- Provide explanations even if model itself is opaque
- Human oversight for critical decisions

### Feature Selection and Proxy Variables

**Issue**: Features may encode protected attributes or biases.

**Direct Discrimination**: Using protected attributes (race, gender, etc.) as features.
- Usually illegal and unethical
- Easy to avoid: don't include these features

**Proxy Discrimination**: Using features correlated with protected attributes.
- Zip code → race
- First name → gender, ethnicity
- University attended → socioeconomic status

**Example - Credit Scoring**:
- Can't use race directly
- But zip code, type of phone, shopping patterns all correlate with race
- Model achieves similar discrimination indirectly

**Challenges**:
- Almost any feature can be a proxy
- Removing features may not eliminate bias (bias in labels, other features)
- May hurt accuracy for everyone

**Approaches**:
- Careful feature auditing
- Fairness-aware feature selection
- Adversarial debiasing
- Regular bias testing

### Algorithmic Accountability

**Issue**: Who is responsible when algorithm causes harm?

**Example Scenarios**:
- Loan denial due to ML model
- Criminal sentenced based on risk assessment algorithm
- Job applicant rejected by resume screening AI

**Questions**:
- Can decision be explained?
- Can decision be appealed?
- Who is accountable (developer, deployer, user)?
- How to remedy harm?

**Requirements**:
- Ability to explain decisions (especially for affected individuals)
- Human review process for high-stakes decisions
- Clear accountability structures
- Audit trails (logging of decisions and reasoning)

### Predictive Policing and Criminal Justice

**Application**: ML for crime prediction, risk assessment.

**Concerns**:

**1. Biased Training Data**:
- Historical arrest data reflects policing patterns, not just crime
- Over-policing in certain neighborhoods creates biased data
- Model learns to predict police behavior, not crime

**2. Feedback Loops**:
- Model predicts high risk in area → more policing → more arrests → reinforces prediction
- Self-fulfilling prophecy
- Perpetuates and amplifies existing biases

**3. Lack of Transparency**:
- Proprietary algorithms (e.g., COMPAS)
- Defendants can't challenge predictions
- Difficult to audit for bias

**4. Impact on Communities**:
- Systematic over-surveillance of minority communities
- Presumption of guilt
- Erosion of trust

**Better Practices**:
- Transparency (open algorithms)
- Independent auditing
- Community input
- Focus on harm reduction, not prediction
- Human oversight

### Credit Scoring and Financial Services

**Application**: ML for loan approval, insurance pricing.

**Benefits**:
- More efficient than manual review
- Can identify good risks that traditional methods miss
- Consistent application of criteria

**Concerns**:

**1. Fairness**:
- Disparate impact on protected groups
- Alternative data (social media, browsing) may encode biases
- Reinforces historical inequalities

**2. Transparency**:
- Complex models hard to explain
- Applicants have right to know why they were denied
- "Computer says no" is insufficient

**3. Access to Credit**:
- Those denied credit have fewer opportunities to improve
- Can trap people in poverty
- Economic mobility reduced

**Regulations**:
- Fair Credit Reporting Act (FCRA)
- Equal Credit Opportunity Act (ECOA)
- Require explanations for adverse decisions

### Healthcare Applications

**Application**: ML for diagnosis, treatment recommendations, risk prediction.

**Potential Benefits**:
- Improved accuracy over human doctors (in some tasks)
- Faster diagnosis
- Personalized treatment
- Reduce healthcare costs

**Concerns**:

**1. Training Data Representation**:
- Medical data often biased toward certain demographics
- Model may perform poorly on under-represented groups
- Different presentation of symptoms across demographics

**2. Life-and-Death Stakes**:
- Errors can be fatal
- Need extremely high reliability
- Liability questions

**3. Trust and Adoption**:
- Doctors and patients must trust recommendations
- Need explanations for medical decisions
- Balance between AI and human judgment

**4. Privacy**:
- Health data extremely sensitive
- Risk of re-identification
- Insurance discrimination

**Best Practices**:
- Rigorous testing on diverse populations
- Human physician oversight
- Clear communication of uncertainty
- Privacy protection (HIPAA compliance)
- Ongoing monitoring of performance

### Employment Decisions

**Application**: ML for resume screening, interview evaluation, performance prediction.

**Benefits**:
- Process large volumes of applications
- Reduce human bias
- Focus on relevant qualifications

**Concerns**:

**1. Historical Bias**:
- Training on past hires perpetuates existing biases
- If field historically male/white dominated, model learns this pattern
- Amazon hiring tool: discriminated against women (trained on past resumes)

**2. Proxy Variables**:
- University attended, extracurricular activities, gap in employment
- May correlate with protected characteristics
- Disadvantage certain groups

**3. Lack of Context**:
- Can't account for unique circumstances
- Non-traditional paths undervalued
- Reduces people to features

**4. Legal Issues**:
- Equal Employment Opportunity laws
- Requirement to justify hiring criteria
- Right to explanation

**Better Practices**:
- Regular bias audits
- Human review of decisions
- Diverse training data
- Avoid proxy variables
- Transparency about process

---

## Summary

**Module 6 covers traditional ML algorithms**:

### Key Algorithms

1. **Linear/Logistic Regression**: Simple, interpretable, fast
   - Foundation for many techniques
   - Works well when assumptions met
   
2. **K-Means Clustering**: Unsupervised grouping
   - Fast, scalable
   - Limited to spherical clusters

3. **K-Nearest Neighbors**: Non-parametric classification/regression
   - Simple, flexible
   - Slow, curse of dimensionality

4. **Decision Trees**: Interpretable rule-based models
   - Handles mixed data types
   - Prone to overfitting

5. **Ensembles**: Combining multiple models
   - Random Forests: General purpose, robust
   - Gradient Boosting: State-of-the-art accuracy
   - Often best performing methods

### Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Need interpretability | Linear regression, logistic regression, single decision tree |
| Tabular data, want accuracy | Gradient boosting (XGBoost, LightGBM) |
| General-purpose, robust | Random Forest |
| Non-linear, small dataset | K-NN, kernel methods |
| Unsupervised clustering | K-Means, DBSCAN |
| Mixed data types | Decision trees, tree ensembles |
| High-dimensional sparse data | Linear models with L1 regularization |

### Critical Insights

**No Free Lunch**: Different algorithms suit different problems.

**Ensemble Power**: Combining models usually beats single model.

**Interpretability Matters**: Especially in high-stakes applications.

**Ethics is Essential**: Algorithms can perpetuate and amplify bias.

### Looking Ahead

Module 7 introduces **Reinforcement Learning**—learning from interaction rather than static datasets. This enables agents that learn through trial and error.

## Further Reading

- Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning"
- James, Witten, Hastie, Tibshirani, "An Introduction to Statistical Learning"
- Breiman, "Random Forests" (2001)
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)
- Barocas & Selbst, "Big Data's Disparate Impact" (2016)

