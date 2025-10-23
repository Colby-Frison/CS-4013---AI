# Module 4: Game Searches

## Table of Contents
1. [Optimal Game Searches](#optimal-game-searches)
2. [Pruning Game Search](#pruning-game-search)
3. [Alpha-Beta Examples](#alpha-beta-examples)
4. [Opponent modeling and play styles](#opponent-modeling-and-play-styles)
5. [Monte Carlo Tree Search](#monte-carlo-tree-search)
6. [Rollouts](#rollouts)
7. [Game Search and Ethics](#game-search-and-ethics)

---

## Optimal Game Searches

Games introduce a fundamental challenge: an **adversarial opponent** actively working against our goals. This requires different search strategies than single-agent problems.

### Game Theory Fundamentals

#### Types of Games

**1. Zero-Sum Games**
- One player's gain is another's loss
- Total reward sums to zero
- Examples: Chess, checkers, tic-tac-toe
- Fully competitive

**2. Non-Zero-Sum Games**
- Players can have mutual benefit or mutual loss
- Examples: Prisoner's dilemma, business negotiations
- Can involve cooperation and competition

**3. Perfect Information Games**
- All players see complete game state
- No hidden information
- Examples: Chess, go, checkers

**4. Imperfect Information Games**
- Some information is hidden
- Examples: Poker, bridge, most real-world scenarios

**5. Deterministic vs. Stochastic**
- Deterministic: No randomness (chess)
- Stochastic: Dice, cards, random events (backgammon)

This module focuses primarily on **two-player, zero-sum, deterministic, perfect information games** (like chess).

### Formalizing Games

A game can be defined by:

**1. Initial State**: Starting position (e.g., standard chess setup)

**2. Players**: $PLAYER(s)$ returns whose turn it is in state $s$

**3. Actions**: $ACTIONS(s)$ returns legal moves in state $s$

**4. Transition Model**: $RESULT(s, a)$ returns resulting state after action $a$ in state $s$

**5. Terminal Test**: $TERMINAL-TEST(s)$ checks if game is over

**6. Utility Function**: $UTILITY(s, p)$ returns numerical value of terminal state $s$ for player $p$
- Examples: +1 for win, 0 for draw, -1 for loss

### Game Trees

**Game Tree**: Tree where:
- **Nodes**: Game states
- **Edges**: Legal moves
- **Levels alternate**: MAX's turn, MIN's turn, MAX's turn, ...
- **Leaves**: Terminal states with utility values

**Example: Tic-Tac-Toe (partial tree)**
```
        X to move (MAX)
       / | \
      /  |  \
   O turn  O turn  O turn
   / | \   / | \   / | \
  ...     ...     ...
```

**Complexity**:
- **Tic-Tac-Toe**: ~10^5 nodes
- **Chess**: ~10^47 nodes in typical game
- **Go**: ~10^170 nodes

Most interesting games are too large to fully enumerate!

### Minimax Algorithm

The foundational algorithm for optimal play in two-player zero-sum games.

#### Core Idea

**MAX Player**: Wants to maximize utility
**MIN Player**: Wants to minimize MAX's utility (which maximizes MIN's utility in zero-sum game)

**Assumption**: Both players play optimally.

**Minimax Value**: The best achievable utility for MAX against optimal play from MIN.

#### Recursive Definition

```
MINIMAX-VALUE(s) = 
    if TERMINAL(s):
        return UTILITY(s)
    
    if PLAYER(s) = MAX:
        return max over a in ACTIONS(s) of MINIMAX-VALUE(RESULT(s, a))
    
    if PLAYER(s) = MIN:
        return min over a in ACTIONS(s) of MINIMAX-VALUE(RESULT(s, a))
```

**Interpretation**:
- At MAX nodes: Choose move leading to highest value
- At MIN nodes: Assume opponent chooses move leading to lowest value (for MAX)
- At terminal nodes: Return utility

#### Algorithm

```
function MINIMAX-DECISION(state):
    return arg max over a in ACTIONS(state) of MIN-VALUE(RESULT(state, a))

function MAX-VALUE(state):
    if TERMINAL-TEST(state):
        return UTILITY(state)
    v = -∞
    for each action in ACTIONS(state):
        v = max(v, MIN-VALUE(RESULT(state, action)))
    return v

function MIN-VALUE(state):
    if TERMINAL-TEST(state):
        return UTILITY(state)
    v = +∞
    for each action in ACTIONS(state):
        v = min(v, MAX-VALUE(RESULT(state, action)))
    return v
```

#### Example: Simple Game Tree

```
         MAX
        / | \
       3  12  8    (MIN level)
      /|  |\ |\
     3 12 8 4 2 1  (Terminal values)
```

**Bottom-up evaluation**:
1. MIN chooses minimum at each MIN node:
   - Left MIN node: min(3, 12) = 3
   - Middle MIN node: min(8, 4) = 4  
   - Right MIN node: min(2, 1) = 1

2. MAX chooses maximum at MAX node:
   - max(3, 4, 1) = 4
   - MAX should choose middle branch

#### Properties

**Completeness**: Yes (if game tree is finite)

**Optimality**: Yes (against optimal opponent)
- Minimax is optimal in the game-theoretic sense
- Finds best possible outcome assuming opponent plays perfectly

**Time Complexity**: $O(b^m)$
- $b$: branching factor (average number of legal moves)
- $m$: maximum depth of tree
- Must explore entire game tree

**Space Complexity**: $O(bm)$ (depth-first exploration)

#### Limitations

**Intractability**: Cannot search entire tree for complex games.

**Solution**: 
1. **Depth-Limited Search**: Search only to depth $d$
2. **Evaluation Function**: Estimate utility for non-terminal states

### Evaluation Functions

Since we can't search to game end, we need to estimate how good a position is.

#### Properties of Good Evaluation Functions

**1. Agrees with Utility on Terminal States**
- If state is win: high value
- If state is loss: low value

**2. Efficient to Compute**
- Must evaluate quickly (thousands of positions per second)

**3. Strongly Correlated with Winning Chances**
- Better positions should have higher values

#### Design Approaches

**Weighted Linear Function**:
$EVAL(s) = w_1 f_1(s) + w_2 f_2(s) + ... + w_n f_n(s)$

Where:
- $f_i(s)$: Features of position $s$
- $w_i$: Weights indicating importance

**Chess Example**:
```
EVAL(s) = 9×(#Queens) + 5×(#Rooks) + 3×(#Bishops) + 3×(#Knights) + 1×(#Pawns)
        + 0.5×(mobility) + 0.5×(king safety) + ...
```

Features include:
- Material: Piece values
- Mobility: Number of legal moves
- King Safety: Pawn shield, open lines near king
- Pawn Structure: Doubled pawns, isolated pawns
- Control of Center
- Development

**Modern Approach**: Use machine learning to learn evaluation function from expert games or self-play.

#### Depth-Limited Minimax

```
function DEPTH-LIMITED-MINIMAX(state, depth):
    if TERMINAL(state) or depth = 0:
        return EVAL(state)
    
    if PLAYER(state) = MAX:
        return max over actions of DEPTH-LIMITED-MINIMAX(RESULT(state, a), depth-1)
    else:
        return min over actions of DEPTH-LIMITED-MINIMAX(RESULT(state, a), depth-1)
```

**Horizon Problem**: Important events just beyond search depth are invisible.

**Solution**: 
- **Quiescence Search**: Continue searching until position is "quiet" (no captures, checks, threats)
- Avoids evaluating in middle of tactical sequence

---

## Pruning Game Search

Minimax explores many unnecessary branches. **Alpha-Beta Pruning** eliminates them without affecting result.

### Alpha-Beta Pruning

**Key Insight**: If we know that a move is worse than a previously examined move, we don't need to fully explore it.

#### Key Concepts

**Alpha ($\alpha$)**: Best value MAX can guarantee so far (lower bound on MAX's value)
- Highest value found for MAX on path to root
- Starts at $-\infty$

**Beta ($\beta$)**: Best value MIN can guarantee so far (upper bound on MAX's value)
- Lowest value found for MIN on path to root  
- Starts at $+\infty$

**Pruning Condition**:
- At MAX node: If value $\geq \beta$, prune (MIN won't allow this path)
- At MIN node: If value $\leq \alpha$, prune (MAX has better option elsewhere)

#### Algorithm

```
function ALPHA-BETA-SEARCH(state):
    return MAX-VALUE(state, -∞, +∞)

function MAX-VALUE(state, α, β):
    if TERMINAL(state):
        return UTILITY(state)
    v = -∞
    for each action in ACTIONS(state):
        v = max(v, MIN-VALUE(RESULT(state, action), α, β))
        if v ≥ β:
            return v  # β cutoff (prune)
        α = max(α, v)
    return v

function MIN-VALUE(state, α, β):
    if TERMINAL(state):
        return UTILITY(state)
    v = +∞
    for each action in ACTIONS(state):
        v = min(v, MAX-VALUE(RESULT(state, action), α, β))
        if v ≤ α:
            return v  # α cutoff (prune)
        β = min(β, v)
    return v
```

#### How Pruning Works

**Example**:
```
         MAX (α=-∞, β=+∞)
        / | \
       /  |  \
   MIN3   MIN? MIN?
   / \    
  3   12  
```

1. Explore left MIN node: finds value 3
2. MAX updates α = 3 ("I can guarantee at least 3")
3. Explore middle MIN node:
   - First child returns 2
   - MIN will choose ≤ 2
   - But MAX already has 3 (α = 3)
   - No need to explore rest of middle MIN node—prune!

**Intuition**: 
- If MIN finds a value ≤ α, MAX won't choose this branch (has better option)
- If MAX finds a value ≥ β, MIN won't allow this path (has better option)

#### Properties

**Correctness**: Returns same result as minimax (no loss of information)

**Pruning Effectiveness**: Depends on move ordering

**Best Case** (perfect ordering): $O(b^{d/2})$
- Can search twice as deep in same time!
- Effectively doubles search depth

**Worst Case** (poor ordering): $O(b^d)$ (same as minimax)
- No pruning occurs

**Practical**: With decent move ordering, approximately $O(b^{3d/4})$

### Understanding Alpha-Beta Pruning: Intuitive Explanation

This section provides a deeper intuition for how alpha-beta pruning works and why it's safe.

#### The Core Minimax Insight

**Minimax is about finding "the highest of the lowest scores"**:

```
You pick a move → Creates a position for opponent
                ↓
Opponent responds optimally → Gives you the "lowest" score they can
                ↓
You choose the move → Where that "lowest" is highest

This is MINI-MAX: Maximize the MINimized score
```

**In other words**: You're choosing which position to put the opponent in, knowing they'll respond optimally to hurt you as much as possible. You pick the position where even their best response isn't too bad for you.

#### What Alpha and Beta Really Mean

**Alpha (α)**: 
- Your **security level** - the best score you can guarantee yourself
- "I've already found a move that guarantees me at least α"
- Updates as you explore (only increases)
- Think: "My current best guarantee"

**Beta (β)**:
- Opponent's **limitation on you** - upper bound on what they'll let you have
- "Opponent has already found a way to limit me to at most β"
- Updates as you explore (only decreases)
- Think: "The ceiling opponent has imposed"

#### Why Pruning Is Safe

**The key insight**: If a branch can't affect your final decision, skip it!

**Example scenario:**
```
You've explored branch A: Opponent's best response gives you 3
α = 3 ("I'm guaranteed at least 3")

Now exploring branch B:
  - First child shows: 0
  - MIN (opponent) will choose ≤ 0
  - You won't pick branch B (0 < 3)
  - Remaining children of B are IRRELEVANT
  - Safe to prune them!
```

**The logic:**
1. After exploring branch A, you know you can get at least 3
2. Branch B shows opponent can limit you to 0 (or worse)
3. You'll never choose branch B over branch A
4. Whatever else is in branch B doesn't matter
5. Skip it!

#### Complete Example: Tree Walkthrough

```
                    A (MAX, α=-∞, β=+∞)
                    |
        ┌───────────┼───────────┬────────┐
        │           │           │        │
    B (MIN)     C (MIN)     D (MIN)  E (MIN)
       / \         |          / \       / \
      5   3        2         0   ?     6   3
      
Leaf values: F=5, G=3, H=2, I=0, J=?, K=6, L=3
```

**Step-by-step execution (left to right):**

**Step 1: Explore B**
- B sees children {5, 3}
- MIN chooses min(5, 3) = 3
- Return 3 to A
- **A updates: α = 3** ("I can guarantee at least 3")

**Step 2: Explore C**
- C sees child H = 2
- Return 2 to A
- α stays 3 (2 is worse, don't update)

**Step 3: Explore D - PRUNING HAPPENS!**
- D explores first child I = 0
- MIN sees: "I can give MAX at most 0"
- **Check: Is 0 ≤ α(3)? YES!**
- **Reasoning:**
  - "MAX already has guarantee of 3 from branch B"
  - "This branch gives 0"
  - "MAX won't choose 0 when they have 3"
  - "Value of J is irrelevant!"
- **PRUNE J!** ✂️

**Step 4: Explore E**
- E sees children {6, 3}
- MIN chooses min(6, 3) = 3
- Return 3 to A

**Final: A chooses max(3, 2, 0, 3) = 3** (branch B or E)

#### Why J's Value Doesn't Matter

**Critical question**: "What if J was -1 or 100? Did we lose information?"

**Answer**: No! Here's why:

```
D (MIN) has children {I=0, J=?}

Case 1: J = 5
  MIN chooses: min(0, 5) = 0
  A chooses: max(3, 2, 0, 3) = 3 ✓

Case 2: J = -1
  MIN chooses: min(0, -1) = -1
  A chooses: max(3, 2, -1, 3) = 3 ✓

Case 3: J = 100
  MIN chooses: min(0, 100) = 0
  A chooses: max(3, 2, 0, 3) = 3 ✓
```

**In ALL cases, A chooses 3!**

The value of J is **irrelevant to the final decision** because:
1. MIN at D will choose min(0, J) which is ≤ 0
2. MAX already has α = 3
3. MAX will never choose anything ≤ 0 when they have 3
4. Therefore J cannot affect MAX's choice at the root

**This is why alpha-beta returns the same answer as minimax - we only skip nodes that cannot affect the final decision!**

#### The "Highest of the Lowest" Principle

Minimax gives you **the best guaranteed outcome**:

```
Your move options:
  Move A: Opponent limits you to 3 → "lowest from this: 3"
  Move B: Opponent limits you to 1 → "lowest from this: 1"  
  Move C: Opponent limits you to 5 → "lowest from this: 5"

You choose: Move C (highest of the lowest = 5)
```

Alpha-beta uses this to prune:
- If you've found "lowest = 5" from one branch
- And another branch shows "lowest ≤ 2"
- You know you won't pick the second branch
- So you can stop exploring it

#### Reading the Tree for Decisions

**The values at nodes show the best each player can achieve:**

```
         A=3 (MAX will get 3)
         |
    ┌────┼────┬────┐
    3    2    0    3  ← Options MAX sees
    ↑
MAX picks 3 (highest)

At each MIN node:
  B: min(5,3)=3  ← MIN picks lowest
  C: 2
  D: 0
  E: min(6,3)=3  ← MIN picks lowest
```

**To trace optimal play:**
1. Start at root (value 3)
2. Follow branches where child value matches parent's value
3. At MAX nodes: value comes from maximum child
4. At MIN nodes: value comes from minimum child

**Optimal path**: A→B→G (value 3) or A→E→L (value 3)

#### Leaf Values: Where They Come From

The leaf node scores come from an **evaluation function**:

**Simple games (terminal positions):**
```
+1  = MAX wins
 0  = Draw
-1  = MIN wins
```

**Complex games (non-terminal positions):**
```
+5  = MAX ahead by 5 points (material, position, etc.)
 0  = Equal position
-2  = MIN ahead by 2 points (MAX down by 2)
```

**All scores are from MAX's perspective:**
- Positive = good for MAX
- Negative = bad for MAX (good for MIN)
- MIN tries to minimize this value
- MAX tries to maximize this value

#### Move Ordering Impact

**The order you explore moves dramatically affects pruning!**

**Example:**
```
        A (α=-∞)
       / | \
      5  6  2
```

**Left-to-right**: α=5, then α=6, then 2≤6 (no prune)
**Right-to-left**: α=2, then α=6, then 5≤6 (no prune)

But with different values:
```
        A (α=-∞)
       / | \
      5  6  2
       \    / \
            0  ?
```

**Left-to-right**: α=5→α=6, then see 0, prune ?
**Right-to-left**: α=2, then α=6, then explore all of 5

**Exploring best moves first = maximum pruning = can search twice as deep!**

### Move Ordering

Order in which moves are explored dramatically affects pruning.

**Best Order**: 
- Explore likely best moves first
- More pruning occurs

**Techniques**:

**1. Try Captures and Checks First**
- Forcing moves often best

**2. Use Killer Moves**
- Moves that caused cutoffs at same depth in nearby branches
- "If it worked there, might work here"

**3. History Heuristic**
- Track which moves have been good historically
- Try those first

**4. Iterative Deepening**
- Search depth $d-1$ first
- Use best move from shallow search to order moves for depth $d$

**5. Principal Variation**
- Best line of play from previous search
- Explore first in next search

**6. MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)**
- For captures: taking queen with pawn better than taking pawn with queen

---

## Opponent Modeling and Play Styles

### The Minimax Assumption: Perfect Rational Opponent

**What minimax assumes:**
```
Opponent plays OPTIMALLY at every turn
   ↓
They will ALWAYS choose the move that hurts you most
   ↓
You must prepare for their best response
```

This is a **conservative/defensive** strategy - you assume worst-case opponent behavior.

**Example:**
```
        You (MAX)
           |
    ┌──────┼──────┐
    A      B      C
    
Move A: Safe position, guaranteed +3 even against perfect defense
Move B: Risky position, +10 if opponent makes mistake, +1 if optimal
Move C: Moderate, +5 against most defenses

Minimax chooses: A (guaranteed +3)
Aggressive play: B (gamble for +10)
```

**Minimax logic**: "Opponent will play optimally in move B and limit me to +1, so move A is better (+3 > +1)"

### When the Assumption Breaks Down

#### 1. Non-Optimal Opponents

**Reality**: Most opponents (especially humans) make mistakes!

**Problem with minimax**:
- Chooses safe moves that guarantee certain value
- Misses opportunities to exploit opponent weaknesses
- Might avoid complex positions that confuse opponents

**Example scenario:**
```
Position evaluation:
- Move A: Simple endgame, +2 (optimal defense)
- Move B: Complex tactics, +5 (if opponent finds the defense) OR +10 (if they miss it)

Against grandmaster: Choose A (they'll find defense)
Against beginner: Choose B (they'll likely miss it, get +10)

Minimax always: Choose A (assumes perfect defense)
```

#### 2. Different Objectives (Non-Zero-Sum)

**Zero-sum assumption**: Your gain = opponent's loss

**But what if:**
- Opponent is playing for **their own** material (not minimizing yours)?
- Opponent wants a draw (doesn't care about winning)?
- Opponent is playing for time (clock management)?
- Opponent is playing for style points or entertainment?

**Example:**
```
Current position: You +5 material, Opponent +3 material

Option A: Trade pieces, both lose 2 → You +3, Opponent +1
Option B: No trade → You +5, Opponent +3

Zero-sum thinking: Both equal (your advantage is +2 in both)
Material-maximizing opponent: Prefers B (they have more material)
```

In a **non-zero-sum game**, opponent might choose moves that improve their position **even if it improves yours more**.

#### 3. Risk Profiles: Aggressive vs Defensive

**Defensive (Conservative) Play:**
- Minimize maximum loss
- Guaranteed outcomes
- Safe positions
- This is what **minimax implements**

**Aggressive (Risk-Seeking) Play:**
- Maximize maximum gain
- Accept risks for higher rewards
- Complex, sharp positions
- "Chaos favors the prepared"

**Comparison:**
```
Position offers two choices:

Defensive: Solid +2, no risk
Aggressive: 50% chance +6, 50% chance -1

Expected value of aggressive: (0.5)(6) + (0.5)(-1) = +2.5
                                                      ↑
                              Higher expected value BUT risky!

Minimax chooses: Defensive (+2 guaranteed)
Risk-seeker chooses: Aggressive (higher upside)
```

### Alternative Algorithms for Non-Optimal Opponents

#### Expectimax

When opponent behavior is **probabilistic** (not perfectly optimal):

```python
function EXPECTIMAX-VALUE(state, agent):
    if TERMINAL(state):
        return UTILITY(state)
    
    if agent == MAX:
        # You still maximize
        return max(EXPECTIMAX-VALUE(RESULT(state, a), MIN) 
                   for a in ACTIONS(state))
    
    else:  # agent == MIN (opponent)
        # Opponent plays according to probability distribution
        values = [EXPECTIMAX-VALUE(RESULT(state, a), MAX) 
                  for a in ACTIONS(state)]
        return weighted_average(values, probabilities)
```

**Key difference**:
- MAX nodes: Still maximize (you play optimally)
- MIN nodes: **Average** over moves weighted by probability (opponent is probabilistic)

**When to use**:
- Modeling human opponents (who make mistakes)
- Games with randomness (dice, card draws)
- Opponents with known tendencies

**Example:**
```
           You (MAX)
              |
        ┌─────┼─────┐
        A     B     C
        |     |     |
       +3    +1   +10

Perfect opponent (minimax): Choose A (+3)

Imperfect opponent (80% optimal, 20% random):
  Move A: guaranteed +3
  Move B: 80%(+1) + 20%(other) ≈ +1.5
  Move C: 80%(+10*0.2) + 20%(+10*0.8) = +3.6  ← Choose this!
                    ↑            ↑
              They defend    They miss it
```

#### Opponent Modeling with Heuristics

**Instead of assuming optimal play, model specific opponent:**

**Beginner model:**
```python
def beginner_probability(move):
    if is_obvious(move):
        return 0.7  # 70% chance they see obvious move
    elif is_subtle(move):
        return 0.1  # 10% chance they see subtle move
    else:
        return 0.2  # 20% random
```

**Aggressive player model:**
```python
def aggressive_probability(move):
    if is_attacking(move):
        return 0.6  # Heavily favor attacks
    elif is_defensive(move):
        return 0.1  # Rarely defend
    else:
        return 0.3
```

**Adaptive opponent modeling:**
1. Start with uniform probability distribution
2. Observe opponent's moves during game
3. Update probabilities based on their choices
4. Adjust your strategy accordingly

### Real-World Chess Engines and Play Styles

#### Modern Engines: Mostly Minimax-Based

**Stockfish, Leela Chess Zero, etc.:**
- Use minimax/alpha-beta as foundation
- Assume optimal opponent (highest level)
- But add **evaluation function tuning** for different styles

#### Creating "Play Styles" Without Changing Algorithm

**1. Evaluation Function Weights:**

**Aggressive engine:**
```python
score = 1.2 * material       # Standard material
      + 0.5 * king_safety    # Less defensive
      + 0.8 * attack_potential  # More aggressive
      + 0.2 * pawn_structure  # Less positional
```

**Defensive engine:**
```python
score = 1.0 * material
      + 1.5 * king_safety    # Very defensive
      + 0.3 * attack_potential  # Less aggressive  
      + 0.8 * pawn_structure  # More positional
```

**Result**: Same minimax algorithm, but different positions get higher scores!

**2. Search Depth Variation:**
- Beginner: Depth 1-3 (very limited lookahead)
- Intermediate: Depth 5-8
- Expert: Depth 15-25+

Lower depth = more mistakes = seems less optimal

**3. Selective Search:**
- Aggressive: Search forcing moves (checks, captures) deeper
- Defensive: Search quiet, safe positions deeper
- Positional: Extend search in complex strategic positions

**4. Time Management:**
- Aggressive: Spend more time in attack positions
- Defensive: Spend more time calculating opponent threats
- Blitz style: Use less time per move (more mistakes)

#### Difficulty Levels in Games

**Easy mode:**
```python
# Simulate beginner
- Search depth: 2-3 moves
- Randomly ignore good moves 30% of the time
- Miss tactical combinations
- Simple evaluation function
```

**Medium mode:**
```python
# Competent player
- Search depth: 8-10
- Occasionally miss subtle tactics (5% error)
- Good evaluation
```

**Hard mode:**
```python
# Near-perfect play
- Search depth: 20+
- Full minimax with alpha-beta
- Sophisticated evaluation
- All optimizations
```

### Practical Considerations

#### When to Use Minimax (Optimal Opponent Assumption)

✅ **Good for:**
- High-level competition (opponents are strong)
- Games where mistakes are punished severely
- When you need guaranteed outcomes
- Perfect information games (chess, checkers, Go)

#### When to Use Expectimax (Probabilistic Opponent)

✅ **Good for:**
- Modeling human opponents
- Games with randomness (backgammon, poker)
- When you have data on opponent behavior
- Difficulty levels in games
- Exploiting known weaknesses

#### Hybrid Approaches

**Real competitive engines often:**

1. **Start with minimax** (assume optimal)
2. **Track opponent patterns** (collect data)
3. **Adjust evaluation** slightly (if confident in pattern)
4. **Remain conservative** (don't over-commit to assumptions)

**Example:**
```
"This opponent exchanges queens 90% of the time"
→ Slightly increase value of positions leading to queen trades
→ But still prepare for the case where they don't (minimax backup)
```

#### The Risk of Assumptions

**Danger of assuming non-optimal opponent:**

```
You assume opponent will make mistake at move 5
   ↓
You choose risky strategy  
   ↓
They DON'T make the mistake
   ↓
You lose badly!
```

**Minimax's conservatism is a feature, not a bug:**
- Guarantees you won't do worse than the minimax value
- No surprises from opponent playing better than expected
- "Hope for the best, prepare for the worst"

### Summary: Play Styles

| Style | Algorithm | Assumption | Best For |
|-------|-----------|------------|----------|
| **Optimal (Minimax)** | Standard minimax | Perfect opponent | Competitive play, strong opponents |
| **Aggressive** | Weighted evaluation | Favor attacks | Confusing opponents, time pressure |
| **Defensive** | Weighted evaluation | Value safety | Preserving advantage, endgames |
| **Exploitative** | Expectimax | Probabilistic opponent | Known weaknesses, beginners |
| **Adaptive** | Dynamic model | Learning opponent | Long matches, tournaments |

**Bottom line**: 
- **Minimax assumes perfect rational opponent** (conservative/defensive)
- **Real opponents vary** (mistakes, different objectives, risk profiles)
- **Expectimax and opponent modeling** can exploit this
- **Most strong engines stick with minimax** for guaranteed performance
- **Different "styles" come from evaluation tuning**, not algorithm changes

### Transposition Tables

**Problem**: Same position can be reached via different move orders.

```
  Start
   / \
  A   B   (different moves)
   \ /
 Same position
```

**Solution**: Store previously evaluated positions in hash table.

**Entry**: (position hash, depth, value, best move, node type)

**Benefits**:
- Avoid re-searching same position
- Can be huge speedup in practice
- Provides move ordering hint

**Challenge**: Table fills up—need replacement strategy.

### Iterative Deepening in Games

Search to depth 1, then 2, then 3, etc., until time runs out.

**Advantages**:
1. **Time Management**: Can stop at any time with best move found so far
2. **Move Ordering**: Shallower searches inform deeper searches
3. **Transposition Table**: Builds up useful entries

**Cost**: Repeating shallower searches
- Overhead is small (most work at deepest level)
- Benefits outweigh costs

### Forward Pruning

Alpha-beta is "safe" (preserves minimax value). **Forward pruning** is "unsafe" (might miss best move) but saves time.

**Beam Search**: Only explore top $k$ moves at each node

**Probability-Based**: Only explore moves with sufficient probability of being best

**Risky**: Might prune the actual best move!

**Used When**: Branching factor is huge (e.g., Go), can't afford to explore all moves.

---

## Alpha-Beta Examples

Let's work through detailed examples to understand alpha-beta pruning.

### Example 1: Simple Tree

```
             MAX
            / | \
           /  |  \
         MIN1 MIN2 MIN3
         /|   /\   /\
        3 12 8  2 14 5
```

**Minimax values**: MIN1=3, MIN2=2, MIN3=5 → MAX=5

**Alpha-Beta Execution**:

**Initial**: $\alpha = -\infty, \beta = +\infty$

1. **Explore MIN1**:
   - Child 1: 3 → MIN1 will choose ≤ 3
   - Child 2: 12 → MIN1 value = 3
   - MAX updates: $\alpha = 3$ ("I can guarantee 3")

2. **Explore MIN2**:
   - Child 1: 8 → MIN2 will choose ≤ 8
   - Child 2: 2 → MIN2 value = 2
   - MIN2 = 2 < α = 3
   - MAX won't choose this branch (already has better)
   - Actually, we had to explore both children here

3. **Explore MIN3**:
   - Child 1: 14 → MIN3 will choose ≤ 14
   - Child 2: 5 → MIN3 value = 5
   - MAX updates: $\alpha = 5$

**Result**: MAX = 5 (chooses MIN3 branch)

**Pruning**: None in this particular tree (would need deeper tree).

### Example 2: Deeper Tree with Pruning

```
                    MAX (α=-∞, β=+∞)
                   / | \
                  /  |  \
            MIN(α=-∞)  MIN  MIN
            / | \
           /  |  \
        MAX MAX MAX
        /|   |
       3 12  8 ...
```

**Step-by-step**:

1. **First MIN node, first MAX child**:
   - Explores to leaves: 3, 12
   - MAX value = 12
   - MIN updates β = 12

2. **First MIN node, second MAX child**:
   - First leaf: 8
   - MAX needs ≥ 12 to beat previous
   - But already found 8 < 12
   - If second leaf is worse, MAX here ≤ 8
   - Actually, explore second leaf: 2
   - MAX value here = 8
   - MIN updates β = 8

3. **First MIN node, third MAX child**:
   - First leaf: 14
   - MAX value so far = 14 > β = 8
   - **Prune!** MIN won't choose this branch (already has 8)
   - Don't need to explore rest of this MAX node

4. Continue with second and third MIN nodes...

### Example 3: Move Ordering Impact

**Good Ordering**:
```
    MAX
    / \
  MIN3 MIN8
  /     /
 3     8
```
After left branch, α = 3. Right branch first child is 8, MIN will choose ≤ 8, but α = 3 means we already have better. Wait, that doesn't cause pruning because 8 > 3.

Let me reconsider:

**Better Example**:
```
        MAX (α=-∞)
       / | \
    MIN2 MIN4 MIN6
```

If we explore in order MIN2, MIN4, MIN6:
- After MIN2: α = 2
- MIN4 finds first child = 1, will choose ≤ 1, but α = 2, so no pruning
- We need to explore all

If we explore in order MIN6, MIN4, MIN2:
- After MIN6: α = 6
- MIN4 finds first child = 1 ≤ α, can potentially prune more subtrees

**Key Point**: Exploring best moves first (highest for MAX, lowest for MIN) maximizes pruning.

---

## Monte Carlo Tree Search

For games with enormous branching factors (like Go), even alpha-beta pruning isn't enough. MCTS uses a different approach: **statistical sampling**.

### Why MCTS?

**Problem with Traditional Search** (in games like Go):
- Branching factor ~250 (vs. ~35 for chess)
- Hard to design good evaluation function
- Alpha-beta still explores too many nodes

**MCTS Solution**:
- Don't explore all moves
- Use random sampling to evaluate positions
- Gradually build tree of promising moves
- Balance exploration (trying new things) and exploitation (using what works)

### Core Idea

**Key Insight**: Don't need to evaluate position directly—can estimate value by playing out random games.

**Process**:
1. From current position, play random moves until game ends
2. See who won
3. Repeat many times
4. Move with best win rate is likely best

**Advantage**: No evaluation function needed—only terminal states (wins/losses).

### MCTS Algorithm

Four phases repeated iteratively:

```
function MCTS(root, iterations):
    for i = 1 to iterations:
        # 1. Selection
        node = SELECT(root)
        
        # 2. Expansion
        if node is not terminal and not fully expanded:
            node = EXPAND(node)
        
        # 3. Simulation (Rollout)
        result = SIMULATE(node)
        
        # 4. Backpropagation
        BACKPROPAGATE(node, result)
    
    return best child of root
```

#### Phase 1: Selection

Starting from root, descend tree by choosing children until reaching a leaf node (not fully expanded) or terminal state.

**Selection Strategy**: **UCB1 (Upper Confidence Bound 1)**

$UCB1(n) = \frac{w_n}{n_n} + C \sqrt{\frac{\ln N_n}{n_n}}$

Where:
- $w_n$: Number of wins from node $n$
- $n_n$: Number of simulations through node $n$
- $N_n$: Number of simulations through parent of $n$
- $C$: Exploration constant (typically $\sqrt{2}$)

**Interpretation**:
- First term: **Exploitation** (win rate)
- Second term: **Exploration** (favor less-visited nodes)
- Balances trying good moves vs. exploring uncertain moves

**Selection Rule**: Choose child with highest UCB1 value.

#### Phase 2: Expansion

If selected node is not terminal and has unexplored children, add one new child to the tree.

**Strategy**:
- Choose an untried action
- Create new node for resulting state
- Add to tree

**Note**: Expand one node per iteration (gradual tree growth).

#### Phase 3: Simulation (Rollout)

From newly expanded node (or selected terminal/leaf node), play out game to completion using a **rollout policy** (typically random or simple heuristic).

**Process**:
1. From current state, choose move according to rollout policy
2. Apply move, get new state
3. Repeat until game ends
4. Return result (win/loss/draw)

**Rollout Policy Options**:
- **Uniform Random**: Choose any legal move randomly
- **Weighted Random**: Favor certain types of moves (captures, etc.)
- **Simple Heuristic**: Fast pattern-based policy
- Trade-off: Smarter policy = slower but more accurate

#### Phase 4: Backpropagation

Propagate game result back up tree from simulated node to root.

**Update Each Node on Path**:
- Increment visit count: $n_n \gets n_n + 1$
- Update wins: $w_n \gets w_n + \text{result}$
  - result = 1 if win for this node's player, 0 if loss, 0.5 if draw

**Effect**: 
- Nodes leading to wins have higher win rates
- More simulations → more confident in value
- UCB1 adapts: frequently-visited nodes need higher win rate to be selected

### Why MCTS Works

**Statistical Convergence**: As number of simulations → ∞, MCTS converges to minimax value.

**Anytime Algorithm**: Can stop at any time and use best move found so far.

**Selective Expansion**: 
- Focuses computation on promising branches
- Bad moves rarely explored after initial trials
- Good moves explored deeply

**No Evaluation Function**: 
- Uses win/loss from simulations
- Especially valuable when evaluation is hard to design (like Go)

### MCTS vs. Alpha-Beta

| Aspect | Alpha-Beta | MCTS |
|--------|------------|------|
| **Evaluation** | Needs evaluation function | Uses win/loss from simulation |
| **Exploration** | Systematic (all moves if not pruned) | Selective (focuses on promising) |
| **Best For** | Tactical games (chess) | Strategic games (Go) |
| **Branching** | Works with moderate branching | Handles huge branching factors |
| **Determinism** | Deterministic result | Stochastic (varies with random rollouts) |
| **Anytime** | Less anytime (prefers fixed depth) | Strongly anytime (use current tree) |

### Enhancements to Basic MCTS

**RAVE (Rapid Action Value Estimation)**:
- Share information between positions where same move is available
- Speeds up learning

**Progressive Bias**:
- Initialize nodes with prior knowledge (heuristics)
- Combines domain knowledge with MCTS

**UCT (UCB applied to Trees)**:
- Theoretical guarantees on convergence
- Standard selection policy

**Tree Parallelization**:
- Run multiple simulations in parallel
- Combines results

**Neural Network Rollouts**:
- Use trained network for rollout policy and/or evaluation
- AlphaGo used this to great effect

### AlphaGo: MCTS + Deep Learning

Combined MCTS with deep neural networks:
1. **Policy Network**: Suggests good moves (focused exploration)
2. **Value Network**: Evaluates positions (reduces rollout depth)
3. **MCTS**: Combines policy and value to select moves

**Result**: Defeated world champion Go players—previously thought to be decades away.

---

## Rollouts

Rollouts (simulations) are the core of MCTS evaluation. Design choices significantly affect performance.

### Rollout Policies

#### 1. Uniform Random

**Description**: Choose any legal move with equal probability.

**Advantages**:
- Extremely fast
- No domain knowledge needed
- Unbiased

**Disadvantages**:
- Many simulations are nonsensical
- Slow convergence
- Doesn't reflect reasonable play

**When to Use**: Simple games, baseline comparisons, no domain knowledge available.

#### 2. Weighted Random

**Description**: Assign probabilities to moves based on simple features.

**Example (Go)**:
- Moves near recent moves: 2× probability
- Captures: 3× probability
- Patterns matching "good shape": 2× probability

**Advantages**:
- Faster convergence than uniform
- Still fast to compute
- Incorporates some domain knowledge

**Disadvantages**:
- Requires domain expertise
- Still makes unreasonable moves

**When to Use**: Moderate complexity games, some domain knowledge available.

#### 3. Heuristic-Based

**Description**: Use fast heuristic rules to select moves.

**Examples**:
- Avoid obviously losing moves
- Prefer moves that increase material
- Follow opening book for first few moves

**Advantages**:
- More realistic simulations
- Faster convergence

**Disadvantages**:
- Slower than random
- Can introduce bias (miss unexpected good moves)

**When to Use**: Complex games where completely random play is unrealistic.

#### 4. Learned Policy

**Description**: Train a neural network or other model to predict good moves.

**Training**:
- Supervised: Learn from expert games
- Reinforcement Learning: Learn from self-play

**Advantages**:
- High-quality simulations
- Can discover non-obvious patterns
- Adapts to game complexity

**Disadvantages**:
- Requires training data or training time
- Slower rollouts (neural network forward pass)
- Potential bias from training distribution

**When to Use**: Top-level play, sufficient computational resources, training data available.

### Rollout Depth

**Question**: How far should we simulate?

**Options**:

**1. To Terminal State**
- Simulate until game ends
- Provides true result
- Can be very slow for long games

**2. Fixed Depth + Evaluation**
- Simulate $d$ moves, then evaluate
- Combines MCTS with evaluation function
- Faster for long games

**3. Adaptive Depth**
- Continue until position is "stable" or "quiet"
- Variable depth based on position

**Trade-off**: Longer rollouts = more accurate but fewer rollouts per time.

### Simulation Budget

**Practical Constraint**: Limited time per move.

**Decision**: How many simulations to run?

**Factors**:
- Available time
- Rollout speed
- Tree size
- Diminishing returns (more simulations help less over time)

**Typical**: 
- Fast games: 1,000-10,000 simulations
- Complex games (Go): 100,000-1,000,000+ simulations

### Combining MCTS and Traditional Evaluation

**Hybrid Approach**:
1. MCTS for move selection and exploration
2. Evaluation function for rollout cutoff
3. Combines benefits of both

**Example**: AlphaGo
- MCTS search
- Neural network for policy (move probabilities) and value (position evaluation)
- Rollouts use policy network, then value network evaluates, or play to end

### Variance Reduction

**Problem**: Simulations have high variance (win or lose, not gradual).

**Techniques**:

**1. RAVE**: Share information across similar positions

**2. Control Variates**: Use baseline value to reduce variance

**3. Importance Sampling**: Weight simulations by how representative they are

**Goal**: Faster convergence to true value.

---

## Game Search and Ethics

### Fairness in Games

**Issue**: Is it fair for AI to play against humans?

**Considerations**:
- AI has perfect memory, instant calculation
- Humans have intuition, creativity
- Different strengths make comparison difficult

**Example**: Chess
- Computers now vastly superior to humans
- Is human vs. computer competition meaningful?
- Separate competitions for humans vs. humans and computers vs. computers

**Balance**: 
- Handicaps for AI (time constraints, hardware limits)
- Human-computer cooperation (centaur chess)
- Focus on AI as training tool rather than competitor

### AI in Competitive Gaming

**Issue**: AI playing online games against humans without disclosure.

**Concerns**:
- Deception (pretending to be human)
- Unfair advantage
- Ruins experience for human players

**OpenAI Five**: Dota 2 AI
- Disclosed as AI when playing against pros
- Impressive demonstration
- Ethical because transparent

**Bots in Online Games**:
- Cheating if undisclosed
- Can ruin game ecosystem
- Some games explicitly prohibit bots

**Best Practice**: Disclosure and consent.

### Game AI in Gambling

**Issue**: AI in poker, sports betting, casino games.

**Concerns**:
- Unfair advantage if only some players have AI
- Potential for market manipulation
- Addiction exploitation (AI designing addictive game mechanics)

**Example - Poker Bots**:
- Banned on most online poker sites
- Can win consistently against humans
- Detection is ongoing arms race

**Responsible Use**:
- Regulations on AI use in gambling
- Transparency requirements
- Protection of recreational players

### Violence in Games

**Issue**: Many games involve simulated violence or warfare.

**Ethical Considerations**:
- Is it ethical to create AI that's optimized for simulated violence?
- Does game AI research contribute to real-world weapon systems?
- Dual-use concerns (game AI → military AI)

**Perspectives**:
- Games as harmless entertainment and AI research testbed
- Concern about desensitization and skill transfer
- Military interest in game AI techniques

**Researcher Responsibility**: Consider potential applications and set boundaries.

### AI Superhuman Performance

**Issue**: What happens when AI becomes superhuman?

**Consequences**:
- Human interest in game declines (if unwinnable)
- Professional players obsolete
- Changes nature of game (humans learn from AI)

**Examples**:
- Chess: Computers superior since 1997 (Deep Blue)
- Go: AI superior since 2016 (AlphaGo)
- Starcraft: AI superhuman in 2019

**Adaptations**:
- AI as training partner
- Human competitions separate from AI
- New game variants less solved by AI
- Appreciation for human excellence even if not optimal

### Resource Consumption

**Issue**: Training and running game AI consumes significant resources.

**AlphaGo**: 
- Training: Thousands of TPUs for months
- Enormous energy consumption
- Carbon footprint

**Considerations**:
- Is this resource use justified for games?
- Environmental impact
- Opportunity cost (resources could be used elsewhere)

**Perspective**:
- Games as testbed for techniques used in important applications
- Cultural value of games
- Efficiency improvements benefit all AI

### Accessibility and Inclusivity

**Positive Applications**:
- AI can provide adaptive difficulty (fair challenge for all skill levels)
- Assistive AI for players with disabilities
- Teaching tools (AI explains good moves)

**Negative Potential**:
- Pay-to-win AI assistance
- Exclusion of players without AI resources
- Reduction in human interaction

### Intellectual Property

**Issue**: AI trained on human games.

**Questions**:
- Does AI infringe on professional players' style or strategies?
- Should players consent to AI training on their games?
- Who owns strategies discovered by AI?

**Current Status**: 
- Generally, game moves not copyrightable
- But ethical considerations remain

---

## Summary

**Module 4 extends search to adversarial environments**:

### Key Concepts

1. **Game Trees**: Represent all possible game states and moves
2. **Minimax**: Optimal algorithm assuming perfect opponent
   - Complete and optimal
   - Intractable for complex games
3. **Alpha-Beta Pruning**: Eliminate branches without affecting result
   - Can double search depth in best case
   - Move ordering is crucial
4. **Evaluation Functions**: Estimate position value for non-terminal states
   - Essential for depth-limited search
   - Design combines domain knowledge and/or machine learning
5. **MCTS**: Statistical sampling approach
   - Handles enormous branching factors
   - No evaluation function needed
   - Balances exploration and exploitation
   - Powers modern Go programs

### Algorithm Selection

| Game Characteristics | Recommended Approach |
|---------------------|----------------------|
| Moderate branching, good evaluation | Alpha-beta with iterative deepening |
| Huge branching, hard to evaluate | MCTS |
| Stochastic (dice, cards) | Expectiminimax or MCTS |
| Imperfect information (poker) | Information set search or MCTS with belief states |
| Simple game, small tree | Minimax (exhaustive) |
| Real-time constraints | Anytime MCTS or limited-depth alpha-beta |

### Modern Game AI

**State of the Art**: 
- Chess: Alpha-beta with deep search and learned evaluation (Stockfish, Leela)
- Go: MCTS with deep neural networks (AlphaGo, KataGo)
- Poker: Counterfactual regret minimization (Libratus, Pluribus)
- StarCraft: Deep reinforcement learning (AlphaStar)

**Common Themes**:
- Combining search with learning
- Self-play for training
- Specialized techniques for game properties

### Looking Ahead

Game search is a form of planning in adversarial settings. Module 5 transitions to **machine learning**—instead of hand-coding intelligence, we'll learn it from data.

## Further Reading

- Russell & Norvig, "Artificial Intelligence: A Modern Approach," Chapter 5
- Shannon, "Programming a Computer for Playing Chess" (1950)
- Silver et al., "Mastering the Game of Go with Deep Neural Networks and Tree Search" (2016)
- Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- Coulom, "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search" (2006)

