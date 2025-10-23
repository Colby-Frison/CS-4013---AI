# Module 5: Machine Learning Basics

## Table of Contents
1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [Training ML](#training-ml)
3. [Evaluating ML](#evaluating-ml)
4. [Training Data and Ethics](#training-data-and-ethics)

---

## Introduction to Machine Learning

Machine Learning (ML) represents a fundamental shift in AI: instead of explicitly programming behavior, we create systems that **learn from data**.

### What is Machine Learning?

**Definition**: Machine Learning is the study of algorithms that improve their performance at some task through experience.

**Formal Definition** (Tom Mitchell):
> "A computer program is said to learn from experience E with respect to some task T and performance measure P, if its performance on T, as measured by P, improves with experience E."

**Components**:
- **Task (T)**: What we want the system to do (classify images, predict prices, etc.)
- **Experience (E)**: Data the system learns from (training examples)
- **Performance (P)**: How we measure success (accuracy, error rate, etc.)

### Why Machine Learning?

**Problems ML Solves**:

**1. Too Complex to Program**
- Recognizing faces: Impossible to write explicit rules for all variations
- Understanding speech: Too many accents, contexts, variations
- Solution: Learn patterns from examples

**2. Adaptivity**
- Environments change over time
- User preferences vary
- Hard-coded rules become outdated
- Solution: Learn and adapt

**3. Discovery**
- Patterns in data humans can't see
- Relationships too subtle or high-dimensional
- Solution: Let algorithms discover patterns

**Examples**:
- **Email Spam Filter**: Learn from examples of spam vs. legitimate email
- **Medical Diagnosis**: Learn from patient records and outcomes
- **Recommendation Systems**: Learn user preferences from behavior
- **Autonomous Driving**: Learn to navigate from driving data

### ML vs. Traditional Programming

**Traditional Programming**:
```
Rules + Data → Computer → Answers
```
Programmer writes explicit rules; computer applies them to data.

**Machine Learning**:
```
Data + Answers → Computer → Rules
```
Computer learns rules from data and correct answers (labels).

**Example - Email Classification**:

**Traditional**:
```python
if "free money" in email or "click here" in email:
    return SPAM
else:
    return NOT_SPAM
```
Brittle, incomplete, easy to evade.

**Machine Learning**:
```python
model = train_on_examples(spam_emails, legitimate_emails)
prediction = model.classify(new_email)
```
Learns patterns from thousands of examples, adapts to new spam tactics.

### Types of Machine Learning

#### 1. Supervised Learning

**Definition**: Learn from labeled examples—input-output pairs.

**Setup**:
- Training data: $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$
- Goal: Learn function $f: X \rightarrow Y$ that maps inputs to outputs
- Use $f$ to predict $y$ for new inputs $x$

**Types**:

**Classification**: Output is discrete (category)
- Examples: spam vs. not spam, cat vs. dog, disease present vs. absent
- Output: Class label

**Regression**: Output is continuous (number)
- Examples: house price, temperature, stock price
- Output: Real-valued number

**Examples**:
- Image Classification: (image → "cat", "dog", "car")
- Email Spam: (email text → spam/not spam)
- Medical Diagnosis: (patient data → disease/no disease)
- House Price Prediction: (features → price)

**When to Use**: Have labeled training data, want to predict outputs for new inputs.

#### 2. Unsupervised Learning

**Definition**: Learn patterns from unlabeled data.

**Setup**:
- Training data: $x_1, x_2, ..., x_n$ (no labels)
- Goal: Discover structure in data
- No "correct answer" to learn from

**Types**:

**Clustering**: Group similar examples together
- Examples: Customer segmentation, document organization
- Find natural groupings in data

**Dimensionality Reduction**: Represent data in fewer dimensions
- Examples: Visualization, compression, noise reduction
- Find compact representation preserving important information

**Density Estimation**: Model probability distribution of data
- Examples: Anomaly detection, generative models
- Learn what "normal" data looks like

**Examples**:
- Customer Segmentation: Group customers by purchasing behavior
- Topic Modeling: Discover topics in document collection
- Anomaly Detection: Find unusual patterns (fraud, defects)
- Data Compression: Reduce dimensions while preserving information

**When to Use**: No labels available, want to discover structure, exploratory data analysis.

#### 3. Reinforcement Learning

**Definition**: Learn from rewards and punishments in interactive environment.

**Setup**:
- Agent takes actions in environment
- Receives rewards or penalties
- Goal: Learn policy (action strategy) that maximizes cumulative reward
- No explicit input-output pairs; must explore to learn

**Components**:
- **State**: Current situation
- **Action**: What agent can do
- **Reward**: Immediate feedback (positive or negative)
- **Policy**: Strategy for choosing actions

**Examples**:
- Game Playing: Learn to win games through self-play
- Robot Control: Learn to walk, grasp objects
- Resource Management: Optimize complex systems
- Personalization: Adapt to user preferences over time

**When to Use**: Sequential decision-making, delayed rewards, interactive environment.

### The Machine Learning Pipeline

**Typical Workflow**:

1. **Problem Definition**
   - What are we trying to predict?
   - What data do we have?
   - How will we measure success?

2. **Data Collection**
   - Gather relevant data
   - Ensure sufficient quantity and quality
   - Consider biases and representativeness

3. **Data Preparation**
   - Clean data (handle missing values, outliers)
   - Feature engineering (create useful representations)
   - Split into training/validation/test sets

4. **Model Selection**
   - Choose appropriate algorithm
   - Consider complexity, interpretability, performance

5. **Training**
   - Fit model to training data
   - Tune hyperparameters
   - Monitor for overfitting

6. **Evaluation**
   - Assess performance on held-out test data
   - Compare to baselines
   - Analyze errors

7. **Deployment**
   - Integrate into production system
   - Monitor performance
   - Update as needed

8. **Iteration**
   - Collect more data
   - Improve features
   - Try different models
   - Refine based on real-world performance

### Key Concepts

#### Generalization

**Goal**: Perform well on **new, unseen** data, not just training data.

**Challenge**: Model must capture underlying patterns, not memorize training examples.

**Test**: Hold out some data during training; evaluate on this test set.

#### Features

**Definition**: Measurable properties or characteristics used as input to model.

**Examples**:
- Email: Word frequencies, sender domain, time sent
- House: Square footage, number of bedrooms, location, age
- Image: Pixel values, edges, textures, colors

**Feature Engineering**: Creating useful features from raw data.
- Critical for traditional ML (less so for deep learning)
- Requires domain expertise
- Can dramatically affect performance

#### Model Capacity

**Capacity**: Ability of model to fit complex patterns.

**Low Capacity**: 
- Simple models (few parameters)
- May underfit (can't capture true pattern)
- High bias

**High Capacity**:
- Complex models (many parameters)
- May overfit (memorize training data)
- High variance

**Sweet Spot**: Match capacity to problem complexity.

#### Inductive Bias

**Definition**: Assumptions a learning algorithm makes about the target function.

**Examples**:
- Linear regression: Assumes linear relationship
- Decision trees: Assumes piecewise constant regions
- Neural networks: Assumes hierarchical features

**Why It Matters**: 
- No bias → can't generalize (must memorize)
- Wrong bias → poor performance
- Right bias → efficient learning

---

## Training ML

Training is the process of fitting a model to data—finding parameter values that minimize error.

### The Learning Objective

**Goal**: Find parameters $\theta$ that minimize loss function $L$.

**Loss Function**: Measures how badly model performs on training data.

$$\theta^* = \arg\min_\theta L(\theta; \text{Data})$$

**Common Loss Functions**:

**Regression (continuous output)**:
- **Mean Squared Error (MSE)**: $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **Mean Absolute Error (MAE)**: $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$

**Classification (discrete output)**:
- **0-1 Loss**: Count of misclassifications (not differentiable)
- **Cross-Entropy Loss**: $-\sum_i y_i \log(\hat{y}_i)$
- **Hinge Loss**: For SVMs

### Gradient Descent

The workhorse optimization algorithm for ML.

#### Core Idea

**Analogy**: Walking downhill in fog.
- Can't see the whole landscape
- Can sense which direction is downhill locally
- Take steps in downhill direction
- Eventually reach bottom (local minimum)

#### Algorithm

```
function GRADIENT-DESCENT(loss_function, initial_θ, learning_rate, iterations):
    θ = initial_θ
    
    for i = 1 to iterations:
        gradient = ∇L(θ)  # Compute gradient of loss w.r.t. θ
        θ = θ - learning_rate × gradient  # Update parameters
    
    return θ
```

**Gradient**: Vector of partial derivatives, points in direction of steepest increase.

**Update Rule**: Move in opposite direction (steepest decrease).

$$\theta \leftarrow \theta - \eta \nabla L(\theta)$$

Where:
- $\theta$: Parameters
- $\eta$: Learning rate (step size)
- $\nabla L(\theta)$: Gradient of loss

#### Learning Rate

**Critical Hyperparameter**: Controls step size.

**Too Large**:
- May overshoot minimum
- Oscillate or diverge
- Unstable training

**Too Small**:
- Slow convergence
- May get stuck in local minima
- Wastes computation

**Typical Values**: 0.001 - 0.1 (problem-dependent)

**Learning Rate Schedules**: Decrease learning rate over time.
- Start with larger steps (fast progress)
- End with smaller steps (fine-tuning)

#### Variants

**Stochastic Gradient Descent (SGD)**:
- Use one example at a time (or small batch)
- Noisy gradient estimate
- Much faster per iteration
- Can escape local minima (due to noise)

**Mini-Batch Gradient Descent**:
- Use small batch of examples (e.g., 32-256)
- Balance between full-batch and SGD
- Parallelizable
- Standard in practice

**Momentum**:
- Add velocity term (accumulate past gradients)
- Smoother updates, faster convergence
- Helps escape local minima

**Adam (Adaptive Moment Estimation)**:
- Adaptive learning rates per parameter
- Combines momentum and RMSProp
- Often works well with default settings
- Very popular

### Overfitting and Underfitting

#### Underfitting (High Bias)

**Problem**: Model too simple to capture true pattern.

**Symptoms**:
- High training error
- High test error
- Model is too rigid

**Example**: Using linear model when relationship is non-linear.

**Solutions**:
- Increase model capacity (more parameters)
- Add features
- Reduce regularization
- Train longer

#### Overfitting (High Variance)

**Problem**: Model memorizes training data, doesn't generalize.

**Symptoms**:
- Low training error
- High test error
- Large gap between training and test performance

**Example**: Complex model fitting noise in training data.

**Solutions**:
- Get more training data
- Reduce model capacity
- Regularization
- Early stopping
- Data augmentation

#### The Bias-Variance Tradeoff

**Bias**: Error from wrong assumptions (underfitting)
**Variance**: Error from sensitivity to training set (overfitting)

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Goal**: Balance bias and variance to minimize total error.

```
       Error
         |
         |      Training Error
         |             Test Error
         |            /
         |          /
         |        /
         |      /___________
         |     /
         |    /
         |   /
         |  /
         |_/______________
         Underfitting  Overfitting
                ↑
           Sweet Spot
```

### Regularization

**Goal**: Prevent overfitting by constraining model complexity.

#### L2 Regularization (Ridge)

Add penalty for large weights:

$$L(\theta) = \text{Loss}(\theta) + \lambda \sum_i \theta_i^2$$

**Effect**: 
- Prefers small weights
- Smooth model
- All features retained (weights shrunk, not eliminated)

**Parameter**: $\lambda$ controls strength (larger = more regularization)

#### L1 Regularization (Lasso)

Add penalty for non-zero weights:

$$L(\theta) = \text{Loss}(\theta) + \lambda \sum_i |\theta_i|$$

**Effect**:
- Sparse solutions (many weights exactly zero)
- Feature selection
- Interpretable models

#### Elastic Net

Combines L1 and L2:

$$L(\theta) = \text{Loss}(\theta) + \lambda_1 \sum_i |\theta_i| + \lambda_2 \sum_i \theta_i^2$$

#### Dropout (Neural Networks)

Randomly deactivate neurons during training.

**Effect**: 
- Prevents co-adaptation
- Ensemble effect
- Reduces overfitting

#### Early Stopping

Stop training when validation error stops decreasing.

**Process**:
1. Monitor validation error during training
2. When validation error increases (while training error decreases)
3. Stop training
4. Use parameters from best validation performance

**Advantages**:
- Simple, effective
- Doesn't require choosing regularization parameter
- Prevents overfitting

### Cross-Validation

**Problem**: Single train/test split might be unrepresentative.

**Solution**: Multiple train/test splits, average performance.

#### K-Fold Cross-Validation

1. Split data into $K$ equal parts (folds)
2. For each fold:
   - Train on other $K-1$ folds
   - Test on this fold
3. Average performance across all $K$ folds

**Typical**: $K = 5$ or $K = 10$

**Advantages**:
- More robust performance estimate
- Uses all data for training and testing
- Reduces variance in estimate

**Disadvantages**:
- $K$ times more expensive
- More complex implementation

#### Leave-One-Out Cross-Validation (LOOCV)

Special case: $K = n$ (number of examples)

**Each fold is single example**.

**Advantages**:
- Nearly unbiased estimate
- Deterministic (no randomness in splits)

**Disadvantages**:
- Very expensive ($n$ training runs)
- High variance in estimate

#### Stratified Cross-Validation

Ensure each fold has similar class distribution.

**Important for**:
- Imbalanced datasets
- Small datasets
- Ensuring representative splits

---

## Evaluating ML

How do we know if our model is good? Different metrics for different tasks.

### Classification Metrics

#### Accuracy

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**When to Use**: Balanced classes, all errors equally important.

**Limitation**: Misleading for imbalanced data.

**Example**: 
- 95% of emails are not spam
- Classifier that always predicts "not spam": 95% accuracy
- But it's useless!

#### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

**Provides detailed breakdown of errors**.

#### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Interpretation**: Of all positive predictions, what fraction were correct?

**When It Matters**: Cost of false positives is high.

**Example**: Medical test—don't want to tell healthy people they're sick.

#### Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Interpretation**: Of all actual positives, what fraction did we find?

**When It Matters**: Cost of false negatives is high.

**Example**: Disease screening—don't want to miss sick people.

#### F1 Score

Harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Use**: Balance precision and recall.

**Generalizations**:
- $F_\beta$ score: Weight recall $\beta$ times as much as precision
- $F_1$ treats them equally

#### ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve**:
- Plot True Positive Rate vs. False Positive Rate
- At different classification thresholds
- Shows trade-off between sensitivity and specificity

**AUC (Area Under Curve)**:
- Single number summarizing ROC curve
- 1.0 = perfect classifier
- 0.5 = random guessing
- Higher is better

**Advantages**:
- Threshold-independent
- Robust to class imbalance
- Comprehensive performance measure

### Regression Metrics

#### Mean Squared Error (MSE)

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Properties**:
- Heavily penalizes large errors (squared)
- Same units as $y^2$
- Differentiable (good for optimization)

#### Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

**Advantage**: Same units as $y$ (more interpretable).

#### Mean Absolute Error (MAE)

$$MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**Properties**:
- Linear penalty for errors
- More robust to outliers than MSE
- Same units as $y$

#### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Interpretation**: 
- Proportion of variance explained by model
- 1.0 = perfect predictions
- 0.0 = as good as predicting mean
- Can be negative (worse than mean)

**Use**: Compare models, assess explanatory power.

### Model Comparison

#### Baselines

Always compare to simple baseline:
- **Classification**: Predict most common class
- **Regression**: Predict mean value
- **Time Series**: Predict last observed value

**If model doesn't beat baseline, something is wrong**.

#### Statistical Significance

**Question**: Is difference in performance real or due to chance?

**Approaches**:
- Paired t-test: Compare performance on same test set
- Cross-validated t-test: Account for cross-validation
- Bootstrap: Resample to estimate confidence intervals

**Don't**: Compare single accuracy numbers and declare winner.
**Do**: Report confidence intervals or statistical tests.

### Practical Considerations

#### Train/Validation/Test Split

**Three Sets**:
1. **Training**: Fit model parameters
2. **Validation**: Tune hyperparameters, select model
3. **Test**: Final evaluation (touch only once!)

**Typical Split**: 60% train, 20% validation, 20% test

**Why Three Sets?**: 
- If we tune on test set, we'll overfit to it
- Validation set allows model selection without biasing test set

#### Hyperparameter Tuning

**Hyperparameters**: Settings chosen before training (not learned from data).

**Examples**:
- Learning rate
- Regularization strength
- Number of layers
- Tree depth

**Methods**:

**Grid Search**: Try all combinations of hyperparameter values.
- Exhaustive but expensive
- Works well for few hyperparameters

**Random Search**: Try random combinations.
- More efficient than grid search
- Often finds good solutions faster

**Bayesian Optimization**: Use probabilistic model to guide search.
- Most sophisticated
- Efficient for expensive training

**Always evaluate on validation set, not training or test!**

---

## Training Data and Ethics

Data is the foundation of ML. Biases and issues in data propagate to models.

### Data Quality Issues

#### 1. Biased Data

**Problem**: Training data doesn't represent true distribution or contains systematic biases.

**Sources**:
- **Historical Bias**: Data reflects past discrimination
- **Sampling Bias**: Some groups over/under-represented
- **Measurement Bias**: How data is collected introduces bias
- **Label Bias**: Human labels reflect biases

**Examples**:
- Hiring AI trained on historical hires (mostly men) discriminates against women
- Face recognition trained mostly on white faces performs poorly on other races
- Crime prediction trained on arrest data reflects policing patterns, not crime patterns

**Consequences**: Model perpetuates or amplifies existing biases.

#### 2. Insufficient Data

**Problem**: Not enough examples to learn robust patterns.

**Symptoms**:
- High variance (overfitting)
- Poor generalization
- Sensitive to outliers

**Solutions**:
- Collect more data
- Data augmentation (synthetic examples)
- Transfer learning (use pre-trained models)
- Simpler models

**How Much Data?**: Rule of thumb: At least 10× as many examples as parameters (very approximate).

#### 3. Imbalanced Data

**Problem**: Some classes much more common than others.

**Example**: Fraud detection (0.1% fraud, 99.9% legitimate)

**Consequences**:
- Model biased toward majority class
- Poor performance on minority class
- Misleading accuracy

**Solutions**:
- Collect more minority class examples
- Resample: Over-sample minority or under-sample majority
- Class weights: Penalize errors on minority class more
- Synthetic examples: SMOTE (Synthetic Minority Over-sampling)
- Appropriate metrics: Precision/recall, F1, AUC (not accuracy)

#### 4. Noisy Labels

**Problem**: Incorrect labels in training data.

**Sources**:
- Human annotation errors
- Ambiguous examples
- Data collection errors
- Adversarial labeling

**Consequences**:
- Ceiling on achievable performance
- Model learns noise

**Solutions**:
- Multiple annotators (consensus)
- Quality control processes
- Noise-robust learning algorithms
- Clean most egregious errors

#### 5. Missing Data

**Problem**: Some features missing for some examples.

**Approaches**:
- **Deletion**: Remove examples with missing data (lose information)
- **Imputation**: Fill in missing values
  - Mean/median/mode
  - Regression-based
  - Model-based
- **Indicator Variables**: Add feature indicating missingness
- **Models That Handle Missing Data**: Some algorithms (e.g., XGBoost) can handle natively

### Data Privacy

#### Personal Data

**Issue**: ML often uses personal information.

**Concerns**:
- Consent: Did people agree to this use?
- Purpose: Is this use aligned with collection purpose?
- Sensitivity: Health, financial, location data
- Retention: How long is data kept?

**Regulations**:
- GDPR (Europe): Strict data protection requirements
- CCPA (California): Consumer privacy rights
- HIPAA (US): Health data protection

**Best Practices**:
- Minimize collection (only what's needed)
- Anonymization where possible
- Clear consent and privacy policies
- Data access and deletion rights
- Secure storage and transmission

#### Re-identification

**Problem**: "Anonymous" data can often be de-anonymized.

**Example**: Netflix Prize dataset
- Released "anonymous" movie ratings
- Researchers re-identified individuals by cross-referencing IMDB

**Lesson**: Removing names isn't enough—combinations of features can identify individuals.

**Approaches**:
- Differential privacy (add noise to protect individuals)
- K-anonymity (ensure each record matches at least k others)
- Careful consideration of re-identification risks

#### Model Inversion and Membership Inference

**Model Inversion**: Reconstruct training data from model.
- Query model, infer information about training examples
- Can reveal sensitive information

**Membership Inference**: Determine if specific example was in training set.
- Exploits overfitting
- Privacy violation (reveals participation)

**Defenses**:
- Regularization (reduce overfitting)
- Differential privacy during training
- Limit model access

### Fairness in Machine Learning

#### Types of Bias

**1. Historical Bias**: Data reflects past discrimination

**2. Representation Bias**: Some groups under-represented in data

**3. Measurement Bias**: Proxy variables that disadvantage groups

**4. Evaluation Bias**: Test data not representative

**5. Aggregation Bias**: One-size-fits-all model performs poorly for subgroups

#### Fairness Definitions

**Demographic Parity**: Equal positive prediction rate across groups.
$$P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1)$$

**Equalized Odds**: Equal TPR and FPR across groups.
$$P(\hat{Y}=1 | Y=1, A=0) = P(\hat{Y}=1 | Y=1, A=1)$$
$$P(\hat{Y}=1 | Y=0, A=0) = P(\hat{Y}=1 | Y=0, A=1)$$

**Calibration**: Equal precision across groups.
$$P(Y=1 | \hat{Y}=1, A=0) = P(Y=1 | \hat{Y}=1, A=1)$$

**Individual Fairness**: Similar individuals treated similarly.

**Challenge**: These definitions can be mutually incompatible!

#### Addressing Bias

**Pre-processing**: Fix training data
- Re-sample to balance groups
- Remove biased features
- Re-label examples

**In-processing**: Modify learning algorithm
- Add fairness constraints
- Adversarial debiasing
- Regularization for fairness

**Post-processing**: Adjust model predictions
- Different thresholds per group
- Calibration adjustments

**Trade-offs**: Fairness often comes at cost of accuracy.

### Consent and Transparency

#### Informed Consent

**Requirements**:
- **Awareness**: People know data is being collected
- **Understanding**: People understand how it will be used
- **Voluntary**: People can refuse without penalty
- **Specific**: Consent for specific uses (not blanket)

**Challenges**:
- Long, unread terms of service
- Secondary uses of data
- Data brokers and aggregation

#### Transparency

**Users Should Know**:
- What data is collected
- How it's used
- What model does
- How decisions are made
- How to contest decisions

**Model Explainability**: Making ML decisions interpretable (see Module 1).

### Data Collection Ethics

#### Representativeness

**Goal**: Data should represent population model will be applied to.

**Risks of Unrepresentative Data**:
- Poor performance on under-represented groups
- Biased outcomes
- Unequal service quality

**Example**: Medical AI trained on one demographic may fail on others.

#### Exploitation

**Issues**:
- Underpaid data labeling (Amazon Mechanical Turk)
- Data scraped without compensation
- Communities as data sources without benefit

**Considerations**:
- Fair compensation for data workers
- Benefit sharing with data sources
- Respect for communities

#### Dual Use

**Problem**: Data collected for beneficial purpose used harmfully.

**Example**:
- Facial recognition data for photo organization → surveillance
- Social media data for connection → manipulation

**Responsibility**: Consider potential misuse during collection and sharing.

---

## Summary

**Module 5 introduces the foundations of machine learning**:

### Key Concepts

1. **Machine Learning Paradigm**: Learn from data rather than explicit programming
2. **Types of Learning**: 
   - Supervised (labeled data)
   - Unsupervised (unlabeled data)
   - Reinforcement (rewards/punishments)
3. **Training**: Optimization process to fit model to data
   - Gradient descent and variants
   - Regularization to prevent overfitting
   - Cross-validation for robust evaluation
4. **Evaluation**: Metrics depend on task
   - Classification: Accuracy, precision, recall, F1, AUC
   - Regression: MSE, RMSE, MAE, R²
   - Always compare to baselines
5. **Data Ethics**: 
   - Bias in data leads to biased models
   - Privacy and consent critical
   - Fairness requires explicit attention

### Critical Insights

**No Free Lunch**: No single algorithm works best for all problems.

**Data is Key**: Model quality limited by data quality.

**Generalization is Goal**: Training performance isn't what matters—test performance is.

**Ethics is Essential**: Technical excellence isn't sufficient—must consider societal impact.

### Looking Ahead

Module 6 covers specific traditional ML algorithms:
- Regression
- Classification
- Clustering
- Decision trees
- Ensembles

These build on the foundations established here.

## Further Reading

- Mitchell, "Machine Learning" (classic textbook)
- Murphy, "Machine Learning: A Probabilistic Perspective"
- Bishop, "Pattern Recognition and Machine Learning"
- Goodfellow, Bengio, Courville, "Deep Learning" (Chapters 1-5 for ML basics)
- Barocas, Hardt, Narayanan, "Fairness and Machine Learning" (free online)

