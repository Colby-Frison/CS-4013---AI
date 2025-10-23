# Module 3: Search in the Real World

## Table of Contents
1. [Handling Real-World Scenarios](#handling-real-world-scenarios)
2. [Local Search](#local-search)
3. [Evolutionary Search](#evolutionary-search)
4. [Particle Filtering](#particle-filtering)
5. [Ethics for Search in the Real World](#ethics-for-search-in-the-real-world)

---

## Handling Real-World Scenarios

Module 2 covered "idealized" search in perfect conditions. Real-world problems introduce complications that require different approaches.

### Limitations of Classical Search

**Classical search assumptions**:
- Fully observable environment
- Discrete, deterministic actions
- Finite state space
- Path to goal matters
- Static environment

**Real-world challenges**:
- Continuous state and action spaces
- Uncertainty and stochastic effects
- Infinite or enormous state spaces
- Only final state matters (not path)
- Dynamic, changing environments

### Optimization vs. Path-Finding

#### Path-Finding Problems (Classical Search)
**Goal**: Find sequence of actions leading from start to goal.
- **What matters**: The path taken
- **Solution**: Sequence of actions
- **Examples**: Navigation, puzzle-solving, planning

#### Optimization Problems (Local Search)
**Goal**: Find a configuration (state) that maximizes an objective function.
- **What matters**: The final state quality
- **Path**: Irrelevant—we don't care how we got there
- **Solution**: A state (configuration)
- **Examples**: Scheduling, circuit layout, parameter tuning

### When to Use What

| Problem Type | Approach | Example |
|-------------|----------|---------|
| Path matters, small state space | A*, BFS, IDS | 8-Puzzle, maze navigation |
| Path matters, large state space | A* with good heuristic | Route planning |
| Only goal matters, optimization | Local search | Scheduling, configuration |
| Multiple objectives, complex constraints | Evolutionary algorithms | Design optimization |
| Dynamic, uncertain environment | Particle filtering, online search | Robot localization |

---

## Local Search

Local search algorithms operate on **complete states** rather than paths. They don't explore a search tree—instead, they move from state to neighboring state, trying to improve an objective function.

### Key Characteristics

**State Space**:
- Each state is a complete configuration
- States have a quality score (objective function value)
- We're trying to find the best state

**Movement**:
- From current state, move to a neighboring state
- No memory of path taken
- Very memory efficient (only store current state)

**Objective Function**:
- Fitness function, cost function, or evaluation function
- Maps each state to a numerical value
- Goal: Maximize (or minimize) this function

### Hill Climbing

The simplest local search algorithm—like climbing a hill in the fog.

#### How It Works

**Algorithm**:
```
function HILL-CLIMBING(problem):
    current = problem.INITIAL-STATE
    
    loop:
        neighbors = SUCCESSORS(current)
        next = highest-valued neighbor of current
        
        if VALUE(next) ≤ VALUE(current):
            return current  # Local maximum reached
        
        current = next
```

**Analogy**: Standing on a hill in fog, you can only see immediate surroundings. Take steps uphill until you can't go up anymore.

#### Variants

**1. Simple Hill Climbing (Steepest-Ascent)**
- Examine all neighbors
- Move to best neighbor
- Repeat until no improvement

**2. Stochastic Hill Climbing**
- Choose among uphill moves randomly
- Probability can be weighted by steepness
- Adds randomness to escape some local optima

**3. First-Choice Hill Climbing**
- Generate neighbors randomly
- Move to first one that's better
- Efficient when many neighbors exist

**4. Random-Restart Hill Climbing**
- Run hill climbing multiple times from random starting points
- Keep best result found
- Overcomes local optima problem

#### Problems with Hill Climbing

**1. Local Maxima/Minima**
- Peak that's higher than neighbors but not the global optimum
- Hill climbing gets stuck here
- Very common in complex landscapes

```
      *
     /|\     * ← Global maximum
    / | \   /|
   /  |  \ / |
  /   |   *  |
 /    |      |
--------------
     ^
Local maximum
```

**2. Ridges**
- Sequences of local maxima
- Hard to navigate because each step may go downward
- Common in high-dimensional spaces

**3. Plateaus**
- Flat areas where all neighbors have same value
- Random walk in plateau
- May or may not lead to better regions

#### When to Use Hill Climbing

**Advantages**:
- Very simple to implement
- Extremely memory efficient
- Fast for easy problems

**Best suited for**:
- Smooth objective functions
- Few local optima
- Good starting point available
- Quick approximate solution acceptable

**Examples**:
- Finding minimum of smooth mathematical function
- Tuning parameters when close to optimum
- Simple optimization problems

### Simulated Annealing

Inspired by metallurgy—metal is heated then slowly cooled to reach low-energy crystalline state.

#### How It Works

**Key Idea**: Allow "bad" moves with decreasing probability over time.

**Algorithm**:
```
function SIMULATED-ANNEALING(problem, schedule):
    current = problem.INITIAL-STATE
    
    for t = 1 to ∞:
        T = schedule(t)  # Temperature decreases over time
        if T = 0:
            return current
        
        next = random neighbor of current
        ΔE = VALUE(next) - VALUE(current)
        
        if ΔE > 0:  # Improvement
            current = next
        else:  # Worse state
            current = next with probability e^(ΔE/T)
```

**Temperature Schedule**:
- High temperature (early): Accept most moves (exploration)
- Low temperature (late): Accept fewer bad moves (exploitation)

#### Why It Works

**Escaping Local Optima**:
- Random "bad" moves can jump out of local optima
- As temperature decreases, settles into good solution
- Balance exploration (high T) and exploitation (low T)

**Acceptance Probability**:
$P(\text{accept worse move}) = e^{\frac{\Delta E}{T}}$

Where:
- $\Delta E < 0$ (move is worse)
- $T$ is current temperature
- Larger $T$ → higher probability of accepting bad move
- Larger $|\Delta E|$ (worse move) → lower acceptance probability

#### Temperature Schedules

**Exponential Decay**:
$T(t) = T_0 \cdot \alpha^t$ where $0 < \alpha < 1$ (e.g., $\alpha = 0.95$)

**Linear Decay**:
$T(t) = T_0 - \alpha \cdot t$

**Logarithmic Decay**:
$T(t) = \frac{T_0}{\log(1 + t)}$

**Practical Considerations**:
- Start temperature high enough to accept most moves initially
- End temperature low enough to converge
- Cooling rate affects quality vs. speed trade-off

#### Properties

**Completeness**: If temperature decreases slowly enough, guaranteed to find global optimum (but impractically slow).

**Practical Performance**: 
- Better than hill climbing for rugged landscapes
- Often finds very good solutions
- No guarantee of optimality in reasonable time

#### When to Use Simulated Annealing

**Best for**:
- Many local optima
- Large state spaces
- Continuous optimization
- No good heuristic available

**Examples**:
- VLSI circuit layout
- Job scheduling
- Protein folding
- Traveling salesman problem

**Not ideal for**:
- Small state spaces (just use exhaustive search)
- Smooth landscapes with few local optima (hill climbing is faster)
- When guaranteed optimal solution required

### Local Beam Search

Maintains $k$ states instead of just one.

#### How It Works

**Algorithm**:
```
function LOCAL-BEAM-SEARCH(problem, k):
    states = k randomly generated states
    
    loop:
        successors = all successors of all k states
        if any successor is goal:
            return that successor
        
        states = best k successors from successors
```

**Key Difference from Random Restart**:
- Random restart: $k$ independent searches
- Beam search: $k$ searches share information
- Best $k$ states are kept, regardless of which search generated them

#### Stochastic Beam Search

Select $k$ successors randomly, with probability proportional to their value.

**Advantage**: More diversity in search (less likely all $k$ states converge to same region).

#### When to Use

**Advantages**:
- More thorough than single-state search
- Sharing information between searches
- Parallelizable

**Limitations**:
- Can converge to same region (lack of diversity)
- More memory than hill climbing
- May still get stuck in local optima region

---

## Evolutionary Search

Inspired by biological evolution: populations, selection, reproduction, mutation.

### Genetic Algorithms

The most common evolutionary approach.

#### Core Concepts

**Representation**:
- **Individual**: A candidate solution (e.g., bit string, real-valued vector)
- **Gene**: One element of the representation
- **Chromosome**: Complete individual representation
- **Population**: Collection of individuals

**Example Representations**:
- Bit strings: `10110101`
- Real vectors: `[2.3, -1.7, 0.5]`
- Permutations: `[3, 1, 4, 2]` (for ordering problems)
- Trees: For evolving programs (genetic programming)

**Fitness Function**:
Maps each individual to a numerical score (how good is this solution?).

#### Algorithm

```
function GENETIC-ALGORITHM(population, FITNESS):
    repeat:
        # 1. Selection
        parents = SELECT(population, FITNESS)
        
        # 2. Reproduction (Crossover)
        offspring = CROSSOVER(parents)
        
        # 3. Mutation
        offspring = MUTATE(offspring)
        
        # 4. Replacement
        population = REPLACE(population, offspring)
    
    until termination condition
    
    return best individual in population
```

#### Selection

Choose which individuals reproduce, favoring fitter individuals.

**Methods**:

**1. Fitness-Proportionate (Roulette Wheel)**
- Probability of selection proportional to fitness
- $P(\text{select } i) = \frac{f_i}{\sum_j f_j}$
- High-fitness individuals more likely, but low-fitness still possible

**2. Tournament Selection**
- Choose $k$ individuals randomly
- Select the best among them
- Repeat to fill mating pool
- Parameter $k$ controls selection pressure

**3. Rank Selection**
- Sort by fitness, assign selection probability by rank
- Prevents dominant super-fit individuals from taking over

**4. Truncation Selection**
- Keep top $x$% of population
- Discard the rest
- High selection pressure

**Trade-off**: Strong selection → fast convergence but risk premature convergence. Weak selection → maintains diversity but slow progress.

#### Crossover (Recombination)

Combine two parents to create offspring.

**One-Point Crossover** (for bit strings):
```
Parent 1: 1 1 0 | 1 0 1 1 0
Parent 2: 0 1 1 | 0 1 1 0 1
          -------+----------
Child 1:  1 1 0 | 0 1 1 0 1
Child 2:  0 1 1 | 1 0 1 1 0
```

**Two-Point Crossover**:
```
Parent 1: 1 1 | 0 1 0 | 1 1 0
Parent 2: 0 1 | 1 0 1 | 1 0 1
          ----+-------+------
Child 1:  1 1 | 1 0 1 | 1 1 0
Child 2:  0 1 | 0 1 0 | 1 0 1
```

**Uniform Crossover**:
- Each gene from either parent with 50% probability

**Arithmetic Crossover** (for real vectors):
- Child = $\alpha \cdot Parent1 + (1-\alpha) \cdot Parent2$
- E.g., $\alpha = 0.5$: average of parents

**Why Crossover Works**:
- **Building Block Hypothesis**: Good solutions contain substructures (building blocks) that, when combined, create better solutions
- Crossover preserves and combines these building blocks

#### Mutation

Random alteration of individual genes.

**Bit Flip** (for bit strings):
- Flip each bit with small probability $p_m$ (typically 0.01-0.1)
- Example: `10110101` → `10010101` (bit 3 flipped)

**Gaussian Mutation** (for real vectors):
- Add random Gaussian noise: $x' = x + \mathcal{N}(0, \sigma)$
- $\sigma$ controls mutation strength

**Swap Mutation** (for permutations):
- Swap two random positions
- Example: `[3, 1, 4, 2]` → `[3, 4, 1, 2]`

**Why Mutation Matters**:
- Introduces new genetic material
- Prevents premature convergence
- Explores new regions of search space
- Typically low probability per gene

#### Replacement Strategies

**Generational**:
- Entire population replaced by offspring each generation
- Fast turnover

**Steady-State**:
- Replace worst individuals with new offspring
- Population changes gradually
- Elitism: Always keep best individual(s)

**Elitism**:
- Guarantee best solution(s) survive to next generation
- Prevents losing good solutions due to random chance
- Usually keep 1-5% best individuals

#### Parameters

**Population Size**:
- Larger: More diversity, slower per generation
- Smaller: Faster, risk premature convergence
- Typical: 50-500 individuals

**Crossover Rate**:
- Probability that two parents produce offspring via crossover
- Typical: 0.6-0.9
- Too low: slow progress; too high: disrupts good solutions

**Mutation Rate**:
- Probability of mutating each gene
- Typical: 0.01-0.1
- Too low: loss of diversity; too high: random search

**Number of Generations**:
- Run until convergence or maximum generations
- Typical: 100-10,000 generations

#### Why Genetic Algorithms Work

**Exploration vs. Exploitation**:
- Crossover: Exploits known good solutions
- Mutation: Explores new regions
- Selection: Focuses on promising areas

**Implicit Parallelism**:
- Population samples many regions simultaneously
- Good building blocks propagate through population

**No Gradient Required**:
- Works with discrete, discontinuous, or noisy fitness functions
- No need for differentiability

#### When to Use Genetic Algorithms

**Best for**:
- Large, complex search spaces
- Multiple local optima
- No gradient information available
- Discrete or mixed discrete/continuous spaces
- Multi-objective optimization

**Examples**:
- Circuit design
- Neural network architecture search
- Scheduling and timetabling
- Game strategy evolution
- Engineering design optimization

**Not ideal for**:
- Small search spaces (exhaustive search better)
- Simple convex optimization (gradient descent faster)
- When guaranteed optimal solution needed
- Real-time requirements (GAs are often slow)

### Genetic Programming

Evolve computer programs, not just parameters.

**Representation**: Programs as trees
```
    +
   / \
  x   *
     / \
    y   2
```
Represents: $x + (y \times 2)$

**Operations**:
- Crossover: Swap subtrees between programs
- Mutation: Replace subtree with random subtree

**Applications**:
- Automatic program synthesis
- Symbolic regression (find mathematical formula)
- Game AI behavior evolution

### Evolution Strategies

Similar to GAs but:
- Focus on real-valued optimization
- Mutation is primary operator (crossover secondary)
- Often self-adapt mutation parameters

**Notation**: $(\mu + \lambda)$-ES or $(\mu, \lambda)$-ES
- $\mu$: Number of parents
- $\lambda$: Number of offspring
- +: Parents compete with offspring
- ,: Only offspring considered

### Differential Evolution

Specialized for continuous optimization.

**Mutation**: 
$v = x_{r1} + F \cdot (x_{r2} - x_{r3})$

Uses difference between population members to generate new candidates.

**Often very effective for numerical optimization problems.**

---

## Particle Filtering

Also called Sequential Monte Carlo. Used for **tracking** and **state estimation** in dynamic, uncertain environments.

### The Problem: Tracking and State Estimation

**Scenario**: An agent (robot, vehicle, person) moves through an environment. We have:
- Noisy sensors (observations are uncertain)
- Uncertain motion (actions have stochastic effects)
- Need to estimate current state (position, velocity, etc.)

**Examples**:
- Robot localization: Where am I?
- Object tracking: Where is the person/car?
- Autonomous vehicles: Estimate position and environment state

### Challenges

1. **Uncertainty**: Sensors are noisy, motion is imperfect
2. **High-Dimensional**: State may have many variables
3. **Non-Linear**: Motion and sensor models may be non-linear
4. **Real-Time**: Need fast updates as new data arrives

### Core Idea

**Belief State**: Probability distribution over possible states $P(X_t | e_{1:t})$
- $X_t$: True state at time $t$
- $e_{1:t}$: All observations up to time $t$

**Represent belief with particles**: A set of samples from the distribution.

### Particle Filtering Algorithm

**Representation**: $N$ particles, each representing a hypothesis about the state.

#### Algorithm Steps

```
function PARTICLE-FILTER(particles, observation, action):
    # 1. Prediction (Motion Update)
    for each particle:
        particle.state = PREDICT(particle.state, action)
    
    # 2. Update (Sensor Update)
    for each particle:
        particle.weight = P(observation | particle.state)
    
    # 3. Resampling
    particles = RESAMPLE(particles, weights)
    
    return particles
```

#### 1. Prediction Step (Motion Update)

Apply action to each particle, incorporating motion uncertainty.

**Example**: Robot moves forward 1 meter
- Add noise: Each particle moves forward $1 + \mathcal{N}(0, 0.1)$ meters
- Particles spread out (uncertainty increases)

**Purpose**: Account for motion uncertainty.

#### 2. Update Step (Sensor/Observation Update)

Weight each particle by how well it matches the observation.

**Example**: Robot's sensor detects wall at 2 meters
- Particle at position predicting wall at 2.1m: High weight
- Particle predicting wall at 5m: Low weight

**Weight**: $w_i = P(\text{observation} | \text{particle state})$

**Purpose**: Incorporate sensor information.

#### 3. Resampling Step

Draw new particles from current particles, with probability proportional to weights.

**Effect**:
- High-weight particles likely to be duplicated
- Low-weight particles likely to be eliminated
- Maintains $N$ particles, but focused on high-probability regions

**Methods**:
- **Multinomial Resampling**: Sample $N$ times with replacement
- **Systematic Resampling**: Lower variance, more efficient
- **Residual Resampling**: Keep high-weight particles, sample rest

**Purpose**: Focus computational resources on likely hypotheses.

### Why Particle Filters Work

**Approximating Belief**:
- Each particle is a hypothesis about true state
- Density of particles approximates probability
- More particles → better approximation

**Handling Non-Linearity**:
- No assumption of linearity (unlike Kalman filters)
- Can represent multi-modal distributions
- Works with arbitrary motion and sensor models

**Computational Efficiency**:
- Number of particles controls accuracy vs. speed
- Can be parallelized
- Adapts to difficulty (particles concentrate in likely regions)

### When to Use Particle Filters

**Best for**:
- Non-linear dynamics
- Non-Gaussian noise
- Multi-modal distributions (multiple hypotheses)
- Complex sensor models

**Examples**:
- Robot localization (where am I in this map?)
- Visual tracking (track person in video)
- Simultaneous Localization and Mapping (SLAM)
- Financial modeling (tracking market states)

**Alternatives**:
- **Kalman Filter**: Linear Gaussian case (more efficient if applicable)
- **Extended Kalman Filter**: Linearize non-linear models
- **Histogram Filter**: Discretize state space (inefficient in high dimensions)

### Practical Considerations

**Number of Particles**:
- More particles: Better accuracy, more computation
- Typical: 100-10,000 particles
- Depends on dimensionality and complexity

**Particle Deprivation**:
- Problem: All particles may converge to single hypothesis
- Solutions: 
  - Add small random noise during motion update
  - Use enough particles
  - Roughening: Add artificial noise after resampling

**Initial Distribution**:
- Uniform if no prior knowledge
- Focused if good initial estimate available
- Can recover from poor initialization given enough time

### Example: Robot Localization

**Setup**:
- Robot in known map
- Sensors: Range finder (distance to walls)
- Motion: Drive forward with noise

**Particles**: Each represents hypothesis about robot's position and orientation.

**Process**:
1. **Initialize**: Scatter particles uniformly across map
2. **Robot moves**: Predict new position for each particle (with noise)
3. **Sensor reading**: Weight particles by match with expected sensor reading
4. **Resample**: Concentrate particles where sensor matches map
5. **Repeat**: Particles converge to true position

**Outcome**: After several iterations, particles cluster around true robot position—localization achieved!

---

## Ethics for Search in the Real World

### Optimization and Fairness

**Issue**: Optimization inherently favors some outcomes over others.

**Considerations**:
- Who defines the objective function?
- Whose preferences are encoded?
- What gets optimized may not align with all stakeholders' interests

**Example - Scheduling**:
- Optimize for company profit → may overwork employees
- Optimize for efficiency → may create inflexible schedules
- Optimize for worker satisfaction → may reduce company performance

**Best Practices**:
- Involve stakeholders in defining objectives
- Consider multiple objectives (multi-objective optimization)
- Transparency about what's being optimized
- Regular review and adjustment

### Unintended Consequences

**Issue**: Optimization can find unexpected solutions that technically meet criteria but violate intent.

**Examples**:
- Reward hacking: AI finds loopholes in reward function
- Gaming the system: Optimize metric without improving real goal
- Edge cases: Solutions that work on average but fail catastrophically in rare cases

**Real Example - Specification Gaming**:
- Game AI rewarded for scoring points
- Learns to pause game right before losing, keeping high score indefinitely
- Technically optimal, but not desired behavior

**Mitigation**:
- Careful objective function design
- Testing in diverse scenarios
- Human oversight
- Robust to edge cases

### Environmental Impact

**Issue**: Evolutionary algorithms and large-scale optimization are computationally expensive.

**Considerations**:
- Energy consumption of running populations for many generations
- Carbon footprint of computational infrastructure
- Trade-off between solution quality and resource use

**Best Practices**:
- Efficient implementations
- Early stopping when good-enough solution found
- Energy-efficient hardware
- Consider simpler algorithms when sufficient

### Bias in Optimization

**Issue**: Optimization reflects biases in:
- Objective function design
- Training data (for learned fitness functions)
- Constraints and priorities

**Example - Route Optimization**:
- Delivery route optimization may systematically avoid certain neighborhoods
- May perpetuate historical inequities
- Cumulative effects over many decisions

**Mitigation**:
- Audit objective functions for bias
- Test on diverse scenarios
- Include fairness constraints
- Monitor real-world outcomes

### Autonomous Tracking and Privacy

**Issue**: Particle filters and tracking algorithms enable surveillance.

**Applications**:
- Security cameras tracking people
- Vehicle tracking via license plates
- Location tracking via mobile devices

**Concerns**:
- Privacy invasion
- Potential for abuse (stalking, surveillance state)
- Consent and awareness

**Balance**:
- Legitimate uses (self-driving cars tracking pedestrians for safety)
- Need for regulation and oversight
- Privacy-preserving techniques where possible

### Responsibility for Emergent Behavior

**Issue**: Evolutionary algorithms can produce unexpected behaviors.

**Concerns**:
- Solutions may be difficult to understand or explain
- Evolved strategies may exploit unintended features
- Responsibility when evolved system causes harm

**Example**:
- Evolved trading algorithm causes market instability
- Evolved game AI uses strategy considered unfair
- Evolved robot behavior is unsafe in edge cases

**Best Practices**:
- Thorough testing before deployment
- Constraints to prevent dangerous behaviors
- Human oversight of evolved solutions
- Ability to explain or justify solutions

### Dual Use

**Issue**: Optimization and search techniques can be used for harm.

**Examples**:
- Optimizing attack strategies
- Evading detection systems
- Manipulating systems for malicious purposes

**Responsibility**: Consider potential misuse during development and publication.

---

## Summary

**Module 3 extends search to real-world problems**:

### Key Concepts

1. **Optimization vs. Path-Finding**: When path doesn't matter, local search is more efficient
2. **Local Search**: 
   - Hill climbing: Fast but gets stuck in local optima
   - Simulated annealing: Escapes local optima via random moves
   - Both memory-efficient and practical for large spaces
3. **Evolutionary Algorithms**:
   - Population-based search
   - Combines exploration (mutation) and exploitation (selection, crossover)
   - Effective for complex, discontinuous search spaces
4. **Particle Filtering**:
   - Tracking in uncertain, dynamic environments
   - Represents belief as collection of samples
   - Handles non-linear, non-Gaussian problems

### Algorithm Selection Guide

| Scenario | Recommended Approach |
|----------|---------------------|
| Smooth optimization, good initialization | Hill climbing |
| Many local optima | Simulated annealing |
| Complex discrete optimization | Genetic algorithm |
| Real-valued optimization | Evolution strategies, differential evolution |
| Tracking moving object | Particle filter |
| State estimation with uncertainty | Particle filter, Kalman filter |
| Multi-objective optimization | Multi-objective GA (NSGA-II) |

### Common Themes

- **Trade-offs**: Speed vs. quality, exploration vs. exploitation, simplicity vs. sophistication
- **No Free Lunch**: No algorithm is best for all problems
- **Problem Understanding**: Algorithm choice depends deeply on problem characteristics
- **Practical Matters**: Real-world performance depends on parameter tuning and implementation details

### Looking Ahead

Module 4 extends search to **adversarial** environments—games where an opponent actively opposes our goals. This introduces game theory and strategic reasoning.

## Further Reading

- Russell & Norvig, "Artificial Intelligence: A Modern Approach," Chapter 4, 6
- Mitchell, "An Introduction to Genetic Algorithms"
- Thrun, Burgard, & Fox, "Probabilistic Robotics" (for particle filters)
- Luke, "Essentials of Metaheuristics" (free online textbook)

