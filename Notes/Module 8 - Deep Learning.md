# Module 8: Deep Learning

## Table of Contents
1. [Neural Networks](#neural-networks)
2. [Convolutional Networks](#convolutional-networks)
3. [Generative AI](#generative-ai)
4. [Deep Learning Ethics](#deep-learning-ethics)

---

## Neural Networks

Neural networks are the foundation of modern deep learning. They enable learning complex, hierarchical representations from raw data.

### Biological Inspiration

**Artificial neurons loosely inspired by biological neurons**:

**Biological Neuron**:
- Dendrites: Receive signals
- Cell body: Processes signals
- Axon: Transmits output signal
- Synapses: Connections between neurons

**Artificial Neuron (Perceptron)**:
- Inputs: Analogous to dendrites
- Weights: Analogous to synaptic strengths
- Activation function: Analogous to cell body firing
- Output: Analogous to axon signal

**Note**: The analogy is limited—artificial neural networks are highly simplified compared to biological brains.

### The Perceptron

**Simplest artificial neuron** (1950s).

#### Model

**Inputs**: $x_1, x_2, ..., x_n$
**Weights**: $w_1, w_2, ..., w_n$
**Bias**: $b$ (or $w_0$ with $x_0 = 1$)

**Weighted Sum**:
$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b = \mathbf{w}^T \mathbf{x} + b$$

**Activation** (step function):
$$y = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

**Geometric Interpretation**: Defines a hyperplane—points on one side classified as 1, other side as 0.

#### Learning Rule

**Perceptron Learning Algorithm**:
```
Initialize weights randomly

for each training example (x, y_true):
    y_pred = perceptron(x)
    if y_pred ≠ y_true:
        w ← w + α(y_true - y_pred)x
        b ← b + α(y_true - y_pred)
```

**Convergence**: If data is linearly separable, perceptron learning converges to solution.

#### Limitations

**XOR Problem**: Cannot learn XOR (non-linearly separable).

```
x1  x2  |  XOR
0   0   |   0
0   1   |   1
1   0   |   1
1   1   |   0
```

No single line can separate 0s from 1s.

**Solution**: Multi-layer networks (can learn XOR and more complex functions).

### Multi-Layer Neural Networks

**Architecture**: Multiple layers of neurons.

**Structure**:
1. **Input Layer**: Receives features
2. **Hidden Layers**: Intermediate computations
3. **Output Layer**: Produces predictions

**Each neuron**:
- Takes weighted sum of inputs
- Applies activation function
- Passes output to next layer

#### Feedforward Computation

**For each layer $l$**:
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$

Where:
- $\mathbf{a}^{[l-1]}$: Activations from previous layer
- $\mathbf{W}^{[l]}$: Weight matrix for layer $l$
- $\mathbf{b}^{[l]}$: Bias vector for layer $l$
- $g$: Activation function
- $\mathbf{a}^{[l]}$: Activations for layer $l$

**Initial**: $\mathbf{a}^{[0]} = \mathbf{x}$ (input)
**Final**: $\mathbf{a}^{[L]}$ (output)

### Activation Functions

**Purpose**: Introduce non-linearity (without activation, network is just linear transformation).

#### Common Activation Functions

**1. Sigmoid**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
- Range: (0, 1)
- Smooth, differentiable
- Outputs interpretable as probabilities

**Disadvantages**:
- Vanishing gradients (saturates for large |z|)
- Not zero-centered
- Expensive to compute

**Usage**: Output layer for binary classification (historically used in hidden layers, now rare).

**2. Hyperbolic Tangent (tanh)**:
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Properties**:
- Range: (-1, 1)
- Zero-centered
- Similar shape to sigmoid

**Disadvantages**:
- Vanishing gradients

**Usage**: Hidden layers (better than sigmoid, but ReLU often preferred).

**3. ReLU (Rectified Linear Unit)**:
$$\text{ReLU}(z) = \max(0, z)$$

**Properties**:
- Range: [0, ∞)
- Simple, fast to compute
- No vanishing gradient for positive values
- Sparse activation (many neurons output 0)

**Disadvantages**:
- "Dead ReLU" problem: neurons can get stuck at 0
- Not differentiable at 0 (but subgradient used in practice)

**Usage**: Default choice for hidden layers in deep networks.

**4. Leaky ReLU**:
$$\text{LeakyReLU}(z) = \max(\alpha z, z)$$

Where $\alpha$ is small (e.g., 0.01).

**Properties**:
- Addresses dead ReLU problem
- Allows small gradient when $z < 0$

**5. Softmax** (for output layer):
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Properties**:
- Outputs probability distribution (sum to 1)
- Used for multi-class classification

### Universal Approximation Theorem

**Theorem**: A neural network with:
- Single hidden layer
- Finite number of neurons
- Non-linear activation function

Can approximate any continuous function on a compact domain to arbitrary accuracy.

**Interpretation**: Neural networks are extremely expressive—can represent complex functions.

**Caveats**:
- Doesn't say how many neurons needed (might be exponentially many)
- Doesn't say how to learn the weights
- Doesn't guarantee generalization

**Practical Implication**: Deep networks (multiple layers) often learn better representations than wide shallow networks.

### Backpropagation

**The algorithm that makes training deep networks feasible**.

**Goal**: Compute gradient of loss with respect to all weights efficiently.

#### Chain Rule

**Key Insight**: Use chain rule to propagate gradients backward through network.

**For output layer**:
$$\frac{\partial L}{\partial w^{[L]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial w^{[L]}}$$

**For hidden layers** (recursive):
$$\frac{\partial L}{\partial w^{[l]}} = \frac{\partial L}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial w^{[l]}}$$

Where $\frac{\partial L}{\partial a^{[l]}}$ comes from next layer (backpropagated).

#### Algorithm

```
# Forward pass
for l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = g(z[l])

# Compute loss
L = loss(a[L], y_true)

# Backward pass
for l = L down to 1:
    # Compute gradients for layer l
    dL/dz[l] = dL/da[l] * g'(z[l])
    dL/dW[l] = dL/dz[l] @ a[l-1].T
    dL/db[l] = dL/dz[l]
    
    # Propagate gradient to previous layer
    dL/da[l-1] = W[l].T @ dL/dz[l]

# Update weights
for l = 1 to L:
    W[l] ← W[l] - α * dL/dW[l]
    b[l] ← b[l] - α * dL/db[l]
```

**Efficiency**: 
- Computes all gradients in single backward pass
- Time complexity: $O(\text{forward pass})$
- Makes training deep networks feasible

### Training Neural Networks

#### Loss Functions

**Regression**:
- Mean Squared Error: $L = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2$

**Binary Classification**:
- Binary Cross-Entropy: $L = -\frac{1}{n} \sum_i [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$

**Multi-Class Classification**:
- Categorical Cross-Entropy: $L = -\frac{1}{n} \sum_i \sum_c y_{i,c} \log(\hat{y}_{i,c})$

#### Optimization Algorithms

**Stochastic Gradient Descent (SGD)**:
$$w \gets w - \alpha \nabla L(w)$$

**SGD with Momentum**:
$$v \gets \beta v + \nabla L(w)$$
$$w \gets w - \alpha v$$

Smooths updates, accelerates convergence.

**RMSProp**:
Adapts learning rate for each parameter based on recent gradient magnitudes.

**Adam (Adaptive Moment Estimation)**:
Combines momentum and RMSProp.
- Most popular optimizer
- Adaptive learning rates per parameter
- Usually works well with default settings

**Learning Rate Schedules**:
- Step decay: Reduce by factor every N epochs
- Exponential decay: $\alpha_t = \alpha_0 e^{-kt}$
- Cosine annealing: Smooth periodic reduction
- Learning rate warmup: Start low, increase, then decay

#### Regularization Techniques

**1. L2 Regularization (Weight Decay)**:
$$L_{total} = L_{data} + \lambda \sum_i w_i^2$$

Penalizes large weights.

**2. Dropout**:
- Randomly deactivate neurons during training (e.g., 50% probability)
- Forces network to learn redundant representations
- Acts as ensemble method

**3. Batch Normalization**:
- Normalize activations within each mini-batch
- Reduces internal covariate shift
- Allows higher learning rates
- Has regularization effect

**4. Data Augmentation**:
- Create modified versions of training data
- Examples: Rotation, flipping, cropping (images)
- Increases effective dataset size

**5. Early Stopping**:
- Monitor validation loss
- Stop when validation loss stops improving
- Prevents overfitting

#### Initialization

**Poor initialization can prevent learning**.

**Xavier/Glorot Initialization** (for sigmoid/tanh):
$$w \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})$$

**He Initialization** (for ReLU):
$$w \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$$

**Purpose**: Maintain variance of activations and gradients across layers.

#### Challenges

**Vanishing Gradients**:
- Gradients become tiny in early layers
- Network stops learning
- Causes: Deep networks, sigmoid/tanh activations
- Solutions: ReLU, batch normalization, residual connections

**Exploding Gradients**:
- Gradients become huge
- Unstable training
- Solutions: Gradient clipping, better initialization, normalization

**Overfitting**:
- Network memorizes training data
- Poor generalization
- Solutions: Regularization, more data, simpler architecture

### When to Use Neural Networks

**Advantages**:
- Extremely flexible (universal approximation)
- Automatic feature learning
- State-of-the-art performance on many tasks
- Transfer learning (pre-trained models)

**Disadvantages**:
- Require large datasets
- Computationally expensive
- Many hyperparameters to tune
- Black box (hard to interpret)
- Can overfit easily

**Best For**:
- Large datasets available
- Complex patterns (images, text, audio)
- End-to-end learning desired
- Performance more important than interpretability

---

## Convolutional Networks

Specialized neural networks for processing grid-like data, especially images.

### Motivation

**Problem with Fully-Connected Networks for Images**:
- Image: 224×224×3 = 150,528 inputs
- Hidden layer with 1000 neurons: 150 million weights!
- Too many parameters:
  - Overfitting
  - Computational expense
  - Ignores spatial structure

**Solution**: Exploit spatial structure through:
- **Local connectivity**: Neurons connect to small regions
- **Parameter sharing**: Same weights used across image
- **Spatial hierarchy**: Build up from simple to complex features

### Convolutional Layer

**Core building block** of CNNs.

#### How Convolution Works

**Filter (Kernel)**: Small matrix (e.g., 3×3, 5×5) of learnable weights.

**Operation**: Slide filter across image, computing dot product at each position.

**Example** (1D):
```
Input:  [1, 2, 3, 4, 5]
Filter: [1, 0, -1]

Output: (1×1 + 2×0 + 3×-1) = -2
        (2×1 + 3×0 + 4×-1) = -2
        (3×1 + 4×0 + 5×-1) = -2
```

**For images** (2D convolution):
- Filter slides across height and width
- Dot product computed at each position
- Produces feature map (activation map)

**Multiple Filters**:
- Each filter detects different feature
- Stack of feature maps as output
- Next layer's input

#### Properties

**Local Connectivity**:
- Each neuron connects to small spatial region (receptive field)
- Dramatically reduces parameters

**Parameter Sharing**:
- Same filter weights used across entire image
- Assumes features equally useful everywhere
- Further reduces parameters

**Translation Invariance**:
- Feature detected regardless of position in image
- Important for object recognition

#### Hyperparameters

**Filter Size**: Typically 3×3 or 5×5
- Larger: Bigger receptive field, more parameters
- Smaller: More layers needed, fewer parameters

**Number of Filters**: How many features to detect
- More filters: More representational capacity
- Typical: 32, 64, 128, 256, ...

**Stride**: How many pixels to move filter each step
- Stride 1: Move 1 pixel at a time (dense output)
- Stride 2: Move 2 pixels (downsample by 2)

**Padding**: Add zeros around border
- "Valid": No padding (output smaller than input)
- "Same": Pad to keep output same size as input

**Output Size**:
$$\text{Output} = \frac{\text{Input} - \text{Filter} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

### Pooling Layer

**Purpose**: Downsample feature maps.

**Benefits**:
- Reduces spatial dimensions (fewer parameters, less computation)
- Provides translation invariance
- Controls overfitting

#### Max Pooling

**Most common**: Take maximum value in each region.

**Example** (2×2 max pooling):
```
Input:        Output:
1  3  2  4    3  4
5  6  7  8    6  8
2  1  4  3
1  9  2  5
```

**Effect**: Downsamples by factor (e.g., 2×2 pooling → 4× fewer values).

#### Average Pooling

Take average of each region instead of maximum.

**Less common than max pooling** but used in some architectures.

### Typical CNN Architecture

**Pattern**: Convolutional layers + Pooling layers + Fully-connected layers

**Example**:
```
Input Image (224×224×3)
    ↓
Conv Layer 1 (64 filters, 3×3) + ReLU → (224×224×64)
    ↓
Max Pooling (2×2) → (112×112×64)
    ↓
Conv Layer 2 (128 filters, 3×3) + ReLU → (112×112×128)
    ↓
Max Pooling (2×2) → (56×56×128)
    ↓
Conv Layer 3 (256 filters, 3×3) + ReLU → (56×56×256)
    ↓
Max Pooling (2×2) → (28×28×256)
    ↓
Flatten → (200,704 values)
    ↓
Fully-Connected Layer (1024 neurons) + ReLU
    ↓
Dropout (0.5)
    ↓
Fully-Connected Output Layer (1000 classes)
    ↓
Softmax
```

**Progressive**: 
- Early layers: Simple features (edges, colors)
- Middle layers: Textures, patterns
- Late layers: Object parts, complex patterns
- Final layers: High-level concepts

### Famous CNN Architectures

#### LeNet-5 (1998)

**First successful CNN** (digit recognition).
- 7 layers
- ~60K parameters
- Handwritten digit recognition

#### AlexNet (2012)

**Breakthrough in ImageNet competition**.
- 8 layers
- 60M parameters
- ReLU, dropout, data augmentation
- GPU training
- Top-5 error: 15.3% (vs. 26% previous best)

#### VGGNet (2014)

**Very deep network** with simple architecture.
- 16-19 layers
- Only 3×3 convolutions
- 138M parameters
- Demonstrated depth is important

#### GoogLeNet/Inception (2014)

**Inception modules**: Multiple filter sizes in parallel.
- 22 layers
- More efficient than VGGNet (12M parameters)
- Auxiliary classifiers to help gradient flow

#### ResNet (2015)

**Residual connections** allow very deep networks.
- Skip connections: $\mathbf{h}(x) = f(x) + x$
- 50, 101, 152+ layers
- Solves vanishing gradient problem
- Winner of ImageNet 2015

**Why Residual Connections Work**:
- Easy to learn identity function (if layer not useful)
- Gradients flow directly through skip connections
- Enables training very deep networks

#### Modern Architectures

**EfficientNet**: Systematically scales depth, width, resolution.
**Vision Transformers**: Adapt transformer architecture to images.
**ConvNeXt**: Modern convolutions competitive with transformers.

### Transfer Learning

**Key Idea**: Use pre-trained network, adapt to new task.

**Approach**:
1. Take network pre-trained on large dataset (e.g., ImageNet)
2. Remove final layer
3. Add new layer for your task
4. Fine-tune on your data

**Why It Works**:
- Early layers learn general features (edges, textures)
- These features useful for many tasks
- Only need to learn task-specific features

**Benefits**:
- Much less data needed
- Faster training
- Better performance (leverages knowledge from large dataset)

**Common Practice**: 
- Small dataset: Freeze early layers, train only final layers
- Medium dataset: Fine-tune all layers with small learning rate
- Large dataset: Train from scratch or fine-tune with normal learning rate

### Applications of CNNs

**Image Classification**: Assign label to entire image.
**Object Detection**: Locate and classify objects in image (bounding boxes).
**Semantic Segmentation**: Label every pixel with class.
**Instance Segmentation**: Separate individual objects, label pixels.
**Face Recognition**: Identify or verify individuals.
**Medical Imaging**: Detect diseases, tumors, abnormalities.
**Autonomous Vehicles**: Perceive environment.
**Image Generation**: GANs, diffusion models (see next section).

---

## Generative AI

AI that creates new content (text, images, audio, video).

### Generative vs. Discriminative Models

**Discriminative**: Learn $P(y|x)$ (predict output given input).
- Classification, regression
- "What is this?"

**Generative**: Learn $P(x)$ or $P(x, y)$ (model data distribution).
- Can generate new samples
- "Create something similar"

### Generative Adversarial Networks (GANs)

**Core Idea**: Two networks compete—Generator creates fakes, Discriminator distinguishes real from fake.

#### Components

**Generator $G$**:
- Input: Random noise $z$ (latent vector)
- Output: Fake sample $G(z)$
- Goal: Fool discriminator

**Discriminator $D$**:
- Input: Sample $x$ (real or fake)
- Output: Probability that $x$ is real
- Goal: Distinguish real from fake

#### Training Process

**Adversarial Game**:
- Generator tries to maximize $D(G(z))$ (fool discriminator)
- Discriminator tries to maximize $D(x_{real})$ and minimize $D(G(z))$ (correctly classify)

**Training Loop**:
```
for each iteration:
    # Train Discriminator
    Sample real images x_real
    Sample noise z, generate fake images x_fake = G(z)
    
    Discriminator loss:
        L_D = -[log(D(x_real)) + log(1 - D(x_fake))]
    Update D to minimize L_D
    
    # Train Generator
    Sample noise z
    Generate fake images x_fake = G(z)
    
    Generator loss:
        L_G = -log(D(x_fake))
    Update G to minimize L_G (equivalently, maximize D(x_fake))
```

**Nash Equilibrium**: 
- Ideally converges to $D(x) = 0.5$ (can't tell real from fake)
- $G$ generates realistic samples
- In practice, hard to reach stable equilibrium

#### Challenges

**Mode Collapse**:
- Generator produces limited variety
- Collapses to few types of outputs
- Lacks diversity

**Training Instability**:
- Oscillation instead of convergence
- Sensitive to hyperparameters
- Requires careful tuning

**Vanishing Gradients**:
- If D too good, gradients for G vanish
- G stops learning

**Solutions**:
- Improved architectures (DCGAN, StyleGAN)
- Better training objectives (Wasserstein GAN)
- Techniques like spectral normalization, self-attention

#### Applications

**Image Generation**: Create photorealistic images.
**Image-to-Image Translation**: Convert between domains (e.g., sketches → photos).
**Style Transfer**: Apply artistic style to images.
**Super-Resolution**: Enhance low-resolution images.
**Data Augmentation**: Generate synthetic training data.

### Variational Autoencoders (VAEs)

**Different approach** to generation: Learn latent representation, sample from it.

#### Components

**Encoder**: Maps input $x$ to latent distribution parameters ($\mu, \sigma$).

**Latent Space**: Low-dimensional representation (typically Gaussian).

**Decoder**: Maps latent vector $z$ to reconstruction $\hat{x}$.

#### Training

**Objective**: Maximize $P(x)$ (likelihood of data).

**ELBO (Evidence Lower Bound)**:
$$\mathcal{L} = \mathbb{E}[\log P(x|z)] - KL(Q(z|x) || P(z))$$

**Two terms**:
1. **Reconstruction Loss**: How well can we reconstruct input?
2. **KL Divergence**: How close is latent distribution to prior?

**Training**: Maximize ELBO (minimize negative ELBO).

#### Generation

**Sample**: Draw $z$ from prior $P(z)$ (typically standard normal).
**Decode**: $\hat{x} = \text{Decoder}(z)$.

**Advantages over GANs**:
- Stable training
- Principled probabilistic framework
- Interpretable latent space

**Disadvantages**:
- Blurrier outputs (due to reconstruction loss)
- Less realistic than state-of-the-art GANs

### Diffusion Models

**Newest approach**, now state-of-the-art for image generation.

#### Core Idea

**Forward Process** (Diffusion):
- Gradually add noise to data
- After many steps, data becomes pure noise
- $x_0 \to x_1 \to ... \to x_T$ (Gaussian noise)

**Reverse Process** (Denoising):
- Learn to reverse diffusion
- Remove noise step by step
- $x_T \to x_{T-1} \to ... \to x_0$

**Training**: 
- Neural network learns to predict noise at each step
- Train on many examples of (noisy image, noise added)

**Generation**:
- Start with random noise $x_T$
- Iteratively denoise using trained network
- After $T$ steps, have realistic image $x_0$

#### Advantages

- **High-quality outputs**: State-of-the-art image generation
- **Stable training**: More stable than GANs
- **Flexible**: Easy to condition on text, class labels, etc.

**Examples**: 
- DALL-E 2, Stable Diffusion, Midjourney
- Text-to-image generation

### Large Language Models (LLMs)

**Generative models for text**.

#### Transformer Architecture

**Key Innovation**: Self-attention mechanism.

**Attention**: 
- Each word attends to all other words
- Learns which words are relevant to each other
- Captures long-range dependencies

**Architecture**:
- **Encoder**: Processes input text
- **Decoder**: Generates output text
- **Self-Attention Layers**: Core mechanism
- **Feed-Forward Layers**: Process attended information

**Positional Encoding**: Adds position information (since attention is order-invariant).

#### Training

**Pre-training** (Unsupervised):
- **Objective**: Predict next token
- **Data**: Massive text corpora (web, books)
- **Result**: General language understanding

**Fine-tuning** (Supervised):
- **Objective**: Specific task (Q&A, summarization, etc.)
- **Data**: Task-specific datasets
- **Result**: Task-specific performance

**RLHF (Reinforcement Learning from Human Feedback)**:
- Further refine with human preferences
- Used in ChatGPT, Claude, etc.

#### Scale

**Trend**: Bigger models perform better.

**Examples**:
- GPT-2 (2019): 1.5B parameters
- GPT-3 (2020): 175B parameters
- GPT-4 (2023): Rumored >1T parameters

**Emergent Abilities**: 
- Large models show capabilities not seen in smaller models
- Few-shot learning, reasoning, instruction-following

#### Applications

- Text generation (stories, articles, code)
- Translation
- Summarization
- Question answering
- Chatbots (ChatGPT, Claude, Bard)
- Code generation (GitHub Copilot)

### Multimodal Models

**Combine multiple modalities** (text, images, audio, video).

**Examples**:
- **CLIP**: Learns joint embedding of images and text
- **DALL-E**: Text-to-image generation
- **Flamingo**: Visual question answering
- **GPT-4**: Text and images as input

**Applications**:
- Image captioning
- Visual question answering
- Text-to-image/video generation
- Cross-modal retrieval

---

## Deep Learning Ethics

Deep learning's power raises significant ethical concerns.

### Deepfakes and Misinformation

**Problem**: Generative models can create realistic fake content.

**Deepfakes**:
- Fake videos of people saying/doing things they didn't
- Face swaps, voice synthesis
- Indistinguishable from real

**Concerns**:
- **Misinformation**: Fake news, propaganda
- **Fraud**: Impersonation for financial gain
- **Harassment**: Non-consensual intimate images
- **Political Manipulation**: Fake speeches, events

**Example Harms**:
- Fake celebrity videos
- Political deepfakes (fake speeches)
- Revenge porn (face-swapped intimate content)

**Mitigation**:
- **Detection**: Develop deepfake detectors (arms race)
- **Watermarking**: Embed signals in generated content
- **Provenance**: Verify content authenticity
- **Regulation**: Laws against malicious deepfakes
- **Education**: Media literacy, critical evaluation

### Copyright and Training Data

**Problem**: Models trained on copyrighted content.

**Issues**:
- **Artists' Work**: Stable Diffusion trained on billions of images (many copyrighted)
- **Code**: GitHub Copilot trained on public code (various licenses)
- **Text**: LLMs trained on books, articles (copyrighted)

**Questions**:
- Is training on copyrighted data fair use?
- Should creators be compensated?
- Can models reproduce copyrighted content?
- Who owns AI-generated content?

**Legal Uncertainty**:
- Few clear precedents
- Ongoing lawsuits (artists vs. Stable Diffusion, authors vs. OpenAI)
- International variation in laws

**Perspectives**:
- **Pro-training**: Transformative use, benefits society, similar to human learning
- **Pro-creator**: Unfair exploitation, should require permission/compensation

### Bias and Representation

**Problem**: Deep learning models reflect biases in training data.

#### Image Bias

**Examples**:
- Face recognition worse for darker skin tones
- Image search for "CEO" shows mostly white men
- Image generation defaults to stereotypes

**Causes**:
- Unbalanced training data (more images of majority groups)
- Historical bias in image collections
- Labeling bias

**Harms**:
- Perpetuates stereotypes
- Excludes/misrepresents groups
- Unequal service quality

#### Language Bias

**Examples**:
- "Doctor" associated with male pronouns
- Occupation stereotypes (nurse → female, engineer → male)
- Toxic associations with groups

**Causes**:
- Training data reflects societal biases
- Historical text contains discrimination
- Amplification through training

**Harms**:
- Reinforces stereotypes
- Discriminatory outputs
- Harmful associations

#### Mitigation

**Data**:
- Collect diverse, representative data
- Audit datasets for bias
- Balance training data

**Training**:
- Debiasing techniques
- Fairness constraints
- Diverse evaluation

**Deployment**:
- Monitor for bias
- User feedback mechanisms
- Continuous improvement

### Environmental Impact

**Problem**: Training large models consumes enormous energy.

**Scale**:
- GPT-3 training: ~1,300 MWh (equivalent to 120 homes for a year)
- Carbon footprint: Hundreds of tons of CO₂

**Concerns**:
- Climate change contribution
- Unsustainable scaling
- Inequality (only large organizations can afford)

**Mitigation**:
- Efficient architectures
- Better hardware (TPUs, specialized chips)
- Renewable energy for data centers
- Reuse pre-trained models (transfer learning)
- Question whether scale is always necessary

### Job Displacement

**Problem**: AI automates creative and cognitive tasks.

**Affected Fields**:
- Artists, illustrators (generative image models)
- Writers, journalists (LLMs)
- Programmers (code generation)
- Customer service (chatbots)
- Translators (machine translation)

**Concerns**:
- Unemployment, economic disruption
- Devaluation of human skills
- Loss of cultural production diversity

**Nuanced View**:
- AI as tool augmenting humans (not replacing)
- New jobs created (AI trainers, prompt engineers)
- Routine tasks automated, humans focus on creative/strategic work

**Responsibility**:
- Social safety nets
- Retraining programs
- Thoughtful integration of AI

### Accessibility and Inclusivity

**Positive Potential**:
- Assistive technologies (image description for blind users)
- Real-time translation (breaking language barriers)
- Text-to-speech, speech-to-text (accessibility)

**Negative Concerns**:
- Models may perform poorly for minority languages, accents
- Requires high-quality data, which some groups lack
- Access to powerful models limited (expensive, restricted)

**Goal**: Ensure AI benefits everyone, not just privileged groups.

### Consent and Privacy

**Training Data Privacy**:
- Models trained on personal data (social media, emails)
- Can memorize and regurgitate private information
- Difficult to remove data once trained

**Generated Content Privacy**:
- Face generation trained on real faces
- Privacy implications of photorealistic faces

**Mitigation**:
- Data anonymization
- Differential privacy
- Opt-out mechanisms
- Clear data policies

### Autonomous Systems and Accountability

**Problem**: Deep learning systems make autonomous decisions.

**Examples**:
- Self-driving cars
- Medical diagnosis
- Content moderation

**Questions**:
- Who is responsible when system fails?
- How to ensure safe behavior?
- Can decisions be explained?
- How to contest decisions?

**Challenges**:
- Black box nature of deep learning
- Unpredictable failure modes
- Difficulty in proving safety

**Approaches**:
- Rigorous testing
- Interpretability research
- Human oversight
- Clear accountability structures

### Dual Use and Misuse

**Problem**: Deep learning tools can be misused.

**Examples**:
- Surveillance (facial recognition)
- Autonomous weapons
- Social manipulation (targeted propaganda)
- Cyberattacks (automated hacking)

**Responsibility**:
- Developers: Consider potential misuse
- Publishers: Responsible disclosure of research
- Governments: Regulation and oversight
- Community: Norms and ethical guidelines

### Concentration of Power

**Problem**: Only few organizations can train largest models.

**Causes**:
- Requires massive data
- Requires massive compute (expensive)
- Expertise concentrated

**Concerns**:
- Power concentrated in hands of few companies
- Lack of democratic control
- Profit motives may not align with public good
- Exacerbates inequality

**Alternatives**:
- Open-source models
- Public/academic research
- International collaboration
- Regulation of large tech companies

---

## Summary

**Module 8 covers deep learning—the foundation of modern AI**:

### Key Concepts

1. **Neural Networks**: Multi-layer networks with non-linear activations
   - Universal approximation capability
   - Trained via backpropagation and gradient descent
   - Require large data and computational resources
   
2. **Convolutional Networks**: Specialized for images
   - Exploit spatial structure
   - Hierarchical feature learning
   - Transfer learning enables broad applications
   
3. **Generative AI**: Creating new content
   - GANs: Adversarial training
   - VAEs: Probabilistic framework
   - Diffusion Models: Iterative denoising
   - LLMs: Large-scale language models (transformers)
   
4. **Ethics**: Deep learning raises serious concerns
   - Deepfakes, misinformation
   - Bias, fairness, representation
   - Environmental impact
   - Job displacement, accessibility
   - Concentration of power

### Critical Insights

**Power and Versatility**: Deep learning achieves state-of-the-art performance across domains—vision, language, games, science.

**Data Hungry**: Requires large datasets and computational resources—barriers to entry.

**Black Box**: Difficult to interpret—challenges for trust, debugging, and accountability.

**Rapid Progress**: Field evolving extremely quickly—today's state-of-the-art is tomorrow's baseline.

**Dual-Edged**: Enormous potential for benefit and harm—ethical considerations essential.

### Looking Ahead

Module 9 covers advanced topics and AI applications in science and social good—the cutting edge and future directions of AI.

## Further Reading

- Goodfellow, Bengio, Courville, "Deep Learning" (THE deep learning textbook)
- Nielsen, "Neural Networks and Deep Learning" (free online)
- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, 2012)
- Goodfellow et al., "Generative Adversarial Networks" (2014)
- Vaswani et al., "Attention is All You Need" (Transformers, 2017)
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)
- Bender et al., "On the Dangers of Stochastic Parrots" (LLM ethics, 2021)

