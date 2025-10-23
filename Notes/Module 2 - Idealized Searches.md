# Module 2: Idealized Searches

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Uninformed Searches](#uninformed-searches)
3. [Informed Searches](#informed-searches)
4. [A* Examples](#a-star-examples)
5. [Creating Effective Heuristics](#creating-effective-heuristics)
6. [Responsible and Ethical AI for Idealized Searches](#responsible-and-ethical-ai-for-idealized-searches)

---

## Problem Formulation

Before we can solve a problem with search, we must first formulate it precisely. This is often one of the most challenging and important steps in applying AI.

### State-Space Search Model

Every search problem can be defined using five components:

#### 1. Initial State
Where the agent starts.

**Example**:
- GPS Navigation: Your current location
- 8-Puzzle: Starting configuration of tiles
- Route Planning: Starting city

#### 2. Actions (Successor Function)
The set of available actions in a given state $s$, denoted as $ACTIONS(s)$.

**Example**:
- GPS Navigation: Turn left, turn right, go straight
- 8-Puzzle: Slide tile up, down, left, or right (if legal)
- Chess: All legal moves for current position

**Key Consideration**: Actions may be state-dependent (what you can do depends on where you are).

#### 3. Transition Model
Describes what each action does. Given state $s$ and action $a$, returns resulting state $s'$, denoted $RESULT(s, a)$.

**Example**:
- GPS: If at intersection A and turn left, arrive at intersection B
- 8-Puzzle: New configuration after sliding a tile

**Together, Initial State + Actions + Transition Model define the state space**: A graph where nodes are states and edges are actions.

#### 4. Goal Test
Determines whether a given state is a goal state.

**Types**:
- **Explicit**: A specific state (e.g., "Be at location X")
- **Implicit**: A property (e.g., "All tiles in correct position")
- **Multiple Goals**: Several acceptable end states

**Example**:
- GPS: Reached destination coordinates
- 8-Puzzle: Configuration matches goal configuration
- Chess: Opponent's king is in checkmate

#### 5. Path Cost
A function that assigns a numeric cost to each path, denoted $c(s, a, s')$ for taking action $a$ from state $s$ to reach $s'$.

**Common Cost Functions**:
- **Uniform Cost**: Each action costs 1 (minimize number of steps)
- **Distance-Based**: Cost equals distance traveled
- **Time-Based**: Cost equals time taken
- **Resource-Based**: Cost equals fuel/energy consumed

**Path Cost**: Sum of individual action costs along a path.

### Example: The 8-Puzzle

**Problem**: Arrange 8 numbered tiles in a 3x3 grid, with one empty space, to match a goal configuration.

```
Initial State:        Goal State:
7 2 4                 1 2 3
5 _ 6                 4 5 6
8 3 1                 7 8 _
```

**Formulation**:
- **States**: All possible configurations of tiles (9!/2 = 181,440 states)
- **Initial State**: A specific starting configuration
- **Actions**: Slide tile up, down, left, or right into empty space
- **Transition Model**: Results in new tile configuration
- **Goal Test**: Does current configuration match goal?
- **Path Cost**: Number of moves (each move costs 1)

### Example: Route Finding

**Problem**: Find a route from city A to city B.

**Formulation**:
- **States**: Cities or junctions
- **Initial State**: Starting city (e.g., Arad)
- **Actions**: Drive to neighboring cities
- **Transition Model**: Arrive at connected city
- **Goal Test**: Current city = destination city (e.g., Bucharest)
- **Path Cost**: Total distance (or time) traveled

### The Search Tree vs. State Space

**State Space**: The graph of all possible states and transitions (represents the problem).

**Search Tree**: The tree of paths explored by a search algorithm (represents the search process).

**Key Difference**:
- State space: Each state appears once
- Search tree: Same state may appear multiple times (in different paths)

**Why This Matters**: 
- We search the tree but want to find a path in the state space
- Must avoid exploring the same state repeatedly (use "closed list" or "explored set")

### Measuring Problem-Solving Performance

We evaluate search algorithms on four dimensions:

#### 1. Completeness
Does it always find a solution if one exists?

**Why It Matters**: An incomplete algorithm might fail on solvable problems.

#### 2. Optimality
Does it find the best (lowest-cost) solution?

**Why It Matters**: May want the shortest path, fastest route, or most efficient solution.

#### 3. Time Complexity
How long does it take?

Usually measured by number of nodes generated/expanded.

#### 4. Space Complexity
How much memory does it require?

Usually measured by maximum number of nodes in memory.

**Complexity Variables**:
- $b$: Branching factor (maximum number of successors of any node)
- $d$: Depth of shallowest goal node
- $m$: Maximum depth of search tree

---

## Uninformed Searches

Also called **blind search** or **brute-force search**. These algorithms have no knowledge about how close a state is to the goal—they only know how to distinguish a goal state from a non-goal state.

### General Tree Search Algorithm

All search algorithms follow this basic pattern:

```
function TREE-SEARCH(problem):
    frontier = {initial state}
    
    while frontier is not empty:
        node = remove node from frontier
        
        if node is goal:
            return solution
        
        expand node, add successors to frontier
    
    return failure
```

**Key Decisions**:
1. Which node to expand next? (search strategy)
2. How to manage frontier? (data structure)
3. How to avoid revisiting states? (closed list)

### Breadth-First Search (BFS)

#### How It Works
Expand shallowest unexpanded node first. Uses a FIFO queue for the frontier.

**Algorithm**:
1. Start with initial state in queue
2. Remove front node from queue
3. If it's the goal, return solution
4. Otherwise, add all its successors to back of queue
5. Repeat until goal found or queue empty

**Visual Representation**:
```
Level 0:    A
           /|\
Level 1:  B C D
         /| |\
Level 2: E F G H
```
Exploration order: A, B, C, D, E, F, G, H

#### Properties

**Complete**: Yes (if $b$ is finite)
- Will eventually explore every node at each depth level
- If solution exists at finite depth, BFS will find it

**Optimal**: Yes (if path cost is non-decreasing with depth)
- Finds shallowest goal first
- If all actions have same cost, shallowest = cheapest

**Time Complexity**: $O(b^d)$
- Must explore all nodes at levels $0, 1, 2, ..., d$
- Total nodes: $1 + b + b^2 + ... + b^d = O(b^d)$

**Space Complexity**: $O(b^d)$
- Must store all nodes at current level in frontier
- At depth $d$: approximately $b^d$ nodes in memory
- This is the major limitation!

#### When to Use
- Want shortest path (in terms of number of steps)
- Graph is not too wide or deep
- Have sufficient memory
- Completeness is critical

#### Example: Finding Path in Graph

```
   A
  /|\
 B C D
 |   |
 E   F (goal)
```

BFS exploration: A → B, C, D → E, F ✓

Finds path: A → D → F (2 steps)

### Depth-First Search (DFS)

#### How It Works
Expand deepest unexpanded node first. Uses a LIFO stack (or recursion) for the frontier.

**Algorithm**:
1. Start with initial state on stack
2. Pop top node from stack
3. If it's the goal, return solution
4. Otherwise, push all its successors onto stack
5. Repeat until goal found or stack empty

**Visual Representation**:
```
Level 0:    A
           /|\
Level 1:  B C D
         /| |\
Level 2: E F G H
```
Exploration order (depends on child order): A, B, E, F, C, G, H, D

#### Properties

**Complete**: No (can get stuck in infinite paths)
- If tree has infinite depth, may never backtrack
- Complete in finite state spaces with cycle detection

**Optimal**: No
- Finds any solution, not necessarily shortest/cheapest
- May find long path even if short path exists

**Time Complexity**: $O(b^m)$
- Worst case: explore entire tree to depth $m$
- If $m >> d$, this is much worse than BFS

**Space Complexity**: $O(bm)$
- Only need to store path from root to current node
- Plus siblings of nodes on path
- Much better than BFS!

#### When to Use
- Memory is limited
- Solutions are plentiful (many paths to goals)
- Need to explore deep paths
- Can prune or limit depth

#### DFS Variants

**Depth-Limited Search (DLS)**:
- DFS with maximum depth limit $l$
- Complete if $l \geq d$
- Not optimal
- Prevents infinite descent

**Iterative Deepening Search (IDS)** (see next section):
- Combines benefits of BFS and DFS

### Iterative Deepening Search (IDS)

#### How It Works
Perform depth-limited search with increasing depth limits: $0, 1, 2, 3, ..., d$

**Algorithm**:
```
function IDS(problem):
    for depth = 0 to infinity:
        result = DLS(problem, depth)
        if result != cutoff:
            return result
```

**Why This Works**:
- Eventually tries depth $d$, so will find shallowest goal (like BFS)
- Only stores current path, like DFS (memory efficient)

#### Redundant Work?
Seems wasteful to repeat searches at shallower depths, but actually not much overhead!

**Node Generations by Depth**:
- Depth 0: 1 node
- Depth 1: 1 + $b$ nodes  
- Depth 2: 1 + $b$ + $b^2$ nodes
- Depth $d$: $1 + b + b^2 + ... + b^d$ nodes

**Total Work**: Most work is at deepest level (nodes grow exponentially with depth).

**Example** ($b = 10, d = 5$):
- Nodes generated at depth 5: 100,000
- Nodes re-generated at shallower depths: ~11,111
- Overhead: ~11%

#### Properties

**Complete**: Yes (like BFS)
**Optimal**: Yes (if path cost non-decreasing with depth)
**Time Complexity**: $O(b^d)$ (like BFS)
**Space Complexity**: $O(bd)$ (like DFS!)

**Best of Both Worlds**: BFS's optimality with DFS's memory efficiency!

#### When to Use
- Want optimal solution
- Don't know how deep solution is
- Limited memory
- Branching factor is large

This is often the preferred uninformed search method.

### Uniform Cost Search (UCS)

#### How It Works
Expand node with lowest path cost $g(n)$ first. Uses a priority queue ordered by path cost.

**Key Difference from BFS**: BFS minimizes depth; UCS minimizes total path cost.

**Algorithm**:
1. Start with initial state in priority queue (priority = 0)
2. Remove node with lowest path cost
3. If it's the goal, return solution
4. Otherwise, add successors with cumulative path costs
5. Repeat

#### Properties

**Complete**: Yes (if step costs are positive)
**Optimal**: Yes (finds lowest-cost path)
**Time & Space**: $O(b^{C^*/\epsilon})$
- Where $C^*$ is cost of optimal solution
- $\epsilon$ is minimum action cost
- Can be much worse than BFS if costs are small

#### When to Use
- Actions have different costs
- Want lowest-cost solution (not just shortest path)
- All action costs are positive

#### Example

```
   A
  / \
 B(1) C(5)
 |     |
 D(3)  E(2)
```

Numbers are edge costs. Goal: E

- BFS would find: A → C → E (cost 7)
- UCS would find: A → B → D → ... (if there's a cheaper path through B)

Actually, UCS exploration order:
1. A (cost 0)
2. B (cost 1)
3. D (cost 4)
4. C (cost 5)
5. E (cost 7)

### Bidirectional Search

#### How It Works
Search forward from initial state and backward from goal state simultaneously. Stop when the two searches meet.

**Complexity**: $O(b^{d/2})$ - huge improvement over $O(b^d)$!

**Example**: If $b = 10, d = 6$:
- Forward only: $10^6$ = 1,000,000 nodes
- Bidirectional: $2 \times 10^3$ = 2,000 nodes

#### Challenges
- Need to be able to generate predecessors (go backward)
- Need to define goal state explicitly (not just test)
- Tricky to implement correctly
- Must check for intersection efficiently

#### When to Use
- Goal state is explicit
- Can compute predecessors
- Memory available for two searches
- Want significant speedup

---

## Informed Searches

Also called **heuristic search**. These algorithms use domain-specific knowledge to guide search toward the goal more efficiently.

### Heuristic Functions

**Definition**: A heuristic function $h(n)$ estimates the cost from node $n$ to the nearest goal.

$h(n)$ = estimated cost of cheapest path from state at node $n$ to a goal state

**Key Properties**:
- $h(n) = 0$ if $n$ is a goal state
- $h(n) \geq 0$ for all $n$
- Provides informed guidance (not just blind search)

**Example Heuristics for 8-Puzzle**:

$h_1(n)$ = number of misplaced tiles

```
7 2 4          1 2 3
5 _ 6    vs.   4 5 6
8 3 1          7 8 _

h₁ = 8 (all tiles except blank are misplaced)
```

$h_2(n)$ = sum of Manhattan distances of tiles from goal positions

Manhattan distance = $|x_1 - x_2| + |y_1 - y_2|$

```
7 is at (0,0), should be at (2,0): distance = 2
2 is at (0,1), should be at (0,1): distance = 0
...
h₂ = 18
```

### Greedy Best-First Search

#### How It Works
Expand node that appears closest to goal, using only $h(n)$. Uses priority queue ordered by $h(n)$.

**Strategy**: Go in the direction that looks best according to heuristic.

#### Properties

**Complete**: No (can get stuck in loops)
**Optimal**: No (doesn't consider path cost so far)
**Time**: $O(b^m)$ (worst case)
**Space**: $O(b^m)$ (worst case)

**Performance depends heavily on quality of heuristic**.

#### When to Use
- Have good heuristic
- Want quick solution (not necessarily optimal)
- Willing to risk incompleteness

#### Example: Romania Route Finding

Goal: Reach Bucharest from Arad

Using $h(n)$ = straight-line distance to Bucharest:
- Greedy might take: Arad → Sibiu → Fagaras → Bucharest (450 km)
- Optimal path: Arad → Sibiu → Rimnicu Vilcea → Pitesti → Bucharest (418 km)

Greedy found a solution quickly but not the best one.

### A* Search

**The gold standard for optimal search with heuristics.**

#### How It Works
Evaluate nodes using: $f(n) = g(n) + h(n)$

Where:
- $g(n)$ = path cost from start to node $n$ (like UCS)
- $h(n)$ = estimated cost from $n$ to goal (heuristic)
- $f(n)$ = estimated total cost of cheapest solution through $n$

**Strategy**: Expand node with lowest estimated total cost.

**Intuition**: 
- UCS expands by cheapest path so far: $g(n)$
- Greedy expands by estimated cost to goal: $h(n)$
- A* combines both: total estimated cost: $g(n) + h(n)$

#### Properties

**Complete**: Yes (with finite branching factor and arc costs > 0)

**Optimal**: Yes, IF heuristic is **admissible** (and **consistent** for graph search)

**Time**: $O(b^d)$ (worst case, but often much better)

**Space**: $O(b^d)$ (stores all generated nodes - main limitation)

#### Admissibility

**Definition**: A heuristic $h(n)$ is admissible if it never overestimates the true cost to reach a goal.

$h(n) \leq h^*(n)$ for all $n$

Where $h^*(n)$ is the true cost to reach nearest goal from $n$.

**Why This Matters**: If $h$ is admissible, A* is guaranteed to find optimal solution.

**Intuition**: 
- If heuristic is too optimistic (underestimates), we might explore more nodes but won't miss optimal path
- If heuristic is too pessimistic (overestimates), we might reject optimal path prematurely

**Example - 8-Puzzle**:
- $h_1$ (misplaced tiles) is admissible: Each tile must move at least once, so $h_1 \leq h^*$
- $h_2$ (Manhattan distance) is admissible: Can't move tiles diagonally, so Manhattan is minimum
- $h_3 = 2 \times h_2$ is NOT admissible: Overestimates true cost

#### Consistency (Monotonicity)

**Definition**: A heuristic $h(n)$ is consistent if, for every node $n$ and successor $n'$ generated by action $a$:

$h(n) \leq c(n, a, n') + h(n')$

**Interpretation**: Estimated cost from $n$ ≤ (step cost to $n'$) + (estimated cost from $n'$)

This is like the triangle inequality: direct path can't be longer than indirect path.

**Relationship to Admissibility**:
- Consistency implies admissibility
- Admissibility doesn't imply consistency
- For graph search, consistency is needed for optimality

**Why Consistency Matters**:
If $h$ is consistent:
- First time A* expands a node, it has found optimal path to that node
- Can use closed list to avoid re-expansion
- Guaranteed optimal solution

#### A* Example

**Problem**: Find shortest path from S to G

```
     2
  S --- A
  |     |
  1     4
  |     |
  B --- G
     1
```

Edge costs shown. Using heuristic (straight-line distance):
- $h(S) = 5$
- $h(A) = 3$  
- $h(B) = 2$
- $h(G) = 0$

**A* Execution**:
1. Expand S: $f(S) = 0 + 5 = 5$
   - Add A: $f(A) = (0+2) + 3 = 5$
   - Add B: $f(B) = (0+1) + 2 = 3$

2. Expand B: $f(B) = 3$
   - Add G: $f(G) = (1+1) + 0 = 2$

3. Expand G: Goal found!

Path: S → B → G, cost = 2

(Path S → A → G has cost 6, so we found optimal)

#### When to Use A*
- Want optimal solution
- Have admissible heuristic
- Have enough memory
- Willing to spend time to get best solution

This is the most widely used optimal search algorithm.

---

## A* Examples

Let's work through detailed examples to build intuition for how A* works.

### Example 1: Grid Navigation

**Problem**: Navigate from Start (S) to Goal (G) on a grid. Can move up, down, left, right (cost 1 per move).

```
S . . . .
. X X X .
. . . . .
. X X X .
. . . . G
```

X = obstacles

**Heuristic**: Manhattan distance to G

$h(n) = |x_n - x_G| + |y_n - y_G|$

**Initial State**:
- S at (0, 0)
- G at (4, 4)
- $h(S) = |0-4| + |0-4| = 8$

**Step-by-Step**:

1. Expand S (0,0): $g=0, h=8, f=8$
   - Right (0,1): $g=1, h=7, f=8$
   - Down (1,0): $g=1, h=7, f=8$

2. Expand Right (0,1): $g=1, h=7, f=8$
   - Right (0,2): $g=2, h=6, f=8$
   - Down (1,1): blocked by X

3. Continue expanding...

Path found: S → right → right → right → right → down → down → down → down → G

**Key Observations**:
- A* doesn't blindly go toward goal (would hit obstacles)
- Balances progress toward goal with actual path cost
- Finds optimal path around obstacles

### Example 2: 8-Puzzle Walkthrough

**Initial State**:
```
7 2 4
5 _ 6
8 3 1
```

**Goal State**:
```
1 2 3
4 5 6
7 8 _
```

**Heuristic**: Manhattan distance

**Initial Calculation**:
- Tile 1: at (2,2), should be at (0,0): 4
- Tile 2: at (0,1), should be at (0,1): 0
- Tile 3: at (2,1), should be at (0,2): 3
- Tile 4: at (0,2), should be at (1,0): 3
- ... etc ...
- Total: $h = 18$

**A* Process**:
1. Start: $g=0, h=18, f=18$
2. Generate moves (slide tiles into blank)
3. Expand node with lowest $f$ value
4. Continue until goal reached

**Solution Found** (example):
Move sequence that reaches goal in optimal number of moves.

---

## Creating Effective Heuristics

The quality of a heuristic dramatically affects A* performance. Here's how to design good heuristics.

### Properties of Good Heuristics

#### 1. Admissibility (Required for Optimality)
Never overestimate true cost.

**How to Ensure**:
- Derive from relaxed problem (remove constraints)
- Use lower bound on true cost
- Formal proof if possible

#### 2. Consistency (Useful for Efficiency)
Satisfies triangle inequality.

**Benefits**:
- Nodes never need re-expansion
- More efficient graph search

#### 3. Informativeness
Closer to true cost is better.

**Dominance**: If $h_2(n) \geq h_1(n)$ for all $n$, and both are admissible, then $h_2$ **dominates** $h_1$.

**Result**: $h_2$ will expand fewer nodes (more efficient).

**Example - 8-Puzzle**:
- $h_1$ (misplaced tiles): $h_1(n) \leq h^*(n)$
- $h_2$ (Manhattan distance): $h_2(n) \leq h^*(n)$
- $h_2(n) \geq h_1(n)$ for all $n$
- Therefore, $h_2$ dominates $h_1$ (A* with $h_2$ is more efficient)

### Deriving Admissible Heuristics from Relaxed Problems

**Relaxed Problem**: Remove some constraints from original problem.

**Key Insight**: Cost of optimal solution to relaxed problem is admissible heuristic for original problem (since relaxed problem is easier or equal).

#### Example: 8-Puzzle

**Original Problem**:
- Move blank
- Can only swap blank with adjacent tile

**Relaxed Problem 1**: Tiles can move to adjacent squares even if occupied
- Optimal cost = Manhattan distance ($h_2$)

**Relaxed Problem 2**: Tiles can move anywhere in one step
- Optimal cost = number of misplaced tiles ($h_1$)

Both relaxed problems are easier than original, so their optimal costs are admissible heuristics.

#### Example: Route Finding

**Original Problem**: Drive on roads between cities

**Relaxed Problem**: Straight-line travel (ignore roads)
- Optimal cost = Euclidean distance (admissible heuristic)

### Combining Heuristics

If you have multiple admissible heuristics $h_1, h_2, ..., h_m$, you can combine them:

$h(n) = \max(h_1(n), h_2(n), ..., h_m(n))$

This is still admissible and dominates each individual heuristic!

**Why It Works**: 
- Each $h_i$ is optimistic (underestimates)
- Maximum is still ≤ true cost
- Provides tightest bound

**Cost**: Must compute all heuristics (slower per node, but fewer nodes expanded).

### Pattern Databases

**Idea**: Pre-compute exact solutions to subproblems, use as heuristic.

**For 8-Puzzle**:
1. Store solutions for subset of tiles (e.g., tiles 1-4)
2. Use database lookup as heuristic
3. Admissible because ignoring other tiles makes problem easier

**Trade-offs**:
- More accurate heuristic
- Requires preprocessing and memory
- Very effective for many domains

### Learning Heuristics

Use machine learning to learn heuristic function from examples:
- Train on solved problem instances
- Neural network predicts cost-to-goal
- Can discover non-obvious patterns

**Challenge**: Ensuring admissibility (learned heuristic might overestimate).

### Practical Guidelines

1. **Start Simple**: Try obvious relaxations first
2. **Validate Admissibility**: Test extensively or prove formally
3. **Measure Performance**: Count nodes expanded
4. **Iterate**: Improve heuristic based on performance
5. **Consider Computation Cost**: Balance accuracy vs. computation time
6. **Domain Knowledge**: Use problem-specific insights

---

## Responsible and Ethical AI for Idealized Searches

Even seemingly neutral search algorithms raise ethical considerations.

### Efficiency and Environmental Impact

**Issue**: Search algorithms consume computational resources.

**Considerations**:
- Energy consumption of large-scale searches
- Carbon footprint of data centers
- Efficiency improvements reduce environmental impact

**Best Practices**:
- Use most efficient algorithm for the task
- Implement good heuristics to reduce search space
- Consider computational cost in algorithm selection

### Fairness in Route Planning

**Issue**: Navigation systems make choices that affect communities.

**Examples**:
- Routing traffic through residential neighborhoods
- Directing commercial vehicles on inappropriate roads
- Unequal service to different areas

**Considerations**:
- Balance individual efficiency with community impact
- Consider cumulative effects of many routing decisions
- Respect local restrictions and preferences

### Transparency in Decision-Making

**Issue**: Users may not understand why system chose particular path.

**Solutions**:
- Explain trade-offs (fastest vs. shortest vs. most fuel-efficient)
- Show alternative routes
- Disclose factors considered (tolls, traffic, etc.)
- Allow user preferences

### Accessibility

**Issue**: Not all users have same abilities or priorities.

**Considerations**:
- Routes accessible to people with disabilities
- Options for pedestrians, cyclists, public transit
- Consider different levels of technical literacy in interface design

### Privacy Concerns

**Issue**: Route searches reveal sensitive information.

**Concerns**:
- Location data reveals personal patterns
- Destination searches may be sensitive (medical facilities, religious sites)
- Historical searches build detailed profile

**Best Practices**:
- Minimize data collection
- Provide privacy controls
- Secure stored data
- Clear retention policies

### Dual Use Concerns

**Issue**: Search algorithms can be used for harmful purposes.

**Examples**:
- Military targeting systems
- Surveillance optimization
- Unauthorized tracking

**Responsibility**: Consider potential misuse during development.

---

## Summary

**Module 2 establishes the foundation for problem-solving through search**:

### Key Concepts

1. **Problem Formulation**: Define states, actions, goals, and costs
2. **Uninformed Search**: Systematic exploration without domain knowledge
   - BFS: Optimal for uniform costs, memory-intensive
   - DFS: Memory-efficient, not optimal
   - IDS: Best of both worlds for many problems
3. **Informed Search**: Use heuristics to guide search
   - A*: Optimal with admissible heuristics
   - Dominates uninformed methods when good heuristic available
4. **Heuristic Design**: Critical for performance
   - Must be admissible for optimality
   - More informed is better (if admissible)
   - Derive from relaxed problems

### Choosing a Search Algorithm

**Decision Factors**:
- Do you have a good heuristic? → A* or Greedy
- Need optimal solution? → A*, IDS, or BFS
- Memory limited? → IDS or DFS
- Non-uniform costs? → UCS or A*
- Just want any solution quickly? → DFS or Greedy

### Looking Ahead

These "idealized" searches assume:
- Fully observable, deterministic environment
- Known states and actions
- Static environment
- Discrete states and actions

Module 3 will relax these assumptions for real-world problems.

## Further Reading

- Russell & Norvig, "Artificial Intelligence: A Modern Approach," Chapter 3-4
- Pearl, "Heuristics: Intelligent Search Strategies"
- Hart, Nilsson, & Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (original A* paper)

