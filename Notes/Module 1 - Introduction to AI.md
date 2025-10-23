# Module 1: Introduction to AI

## Table of Contents
1. [What is AI?](#what-is-ai)
2. [Intelligent Agents](#intelligent-agents)
3. [Introduction to Responsible and Ethical AI](#introduction-to-responsible-and-ethical-ai)

---

## What is AI?

### Defining Artificial Intelligence

Artificial Intelligence is a broad field concerned with creating systems that can perform tasks that typically require human intelligence. The definition and goals of AI have evolved significantly since the field's inception in the 1950s, and can be understood through four different perspectives:

### The Four Approaches to AI

#### 1. Thinking Humanly: The Cognitive Modeling Approach

**Core Idea**: Build systems that think like humans.

**Methodology**:
- Study how humans think through introspection (examining our own thoughts)
- Conduct psychological experiments to understand cognitive processes
- Use brain imaging to observe neural activity during thinking
- Build computational models that replicate human cognitive processes

**Historical Context**:
The cognitive science approach emerged in the 1960s and attempts to understand the mind as an information-processing system. Researchers create theories about internal knowledge representation and test them through:
- Predicting human behavior in various tasks
- Direct identification of neurological data from brain imaging

**Limitations**:
- Human thinking is not always optimal or efficient
- Difficult to model subconscious and intuitive processes
- Computational limitations make exact simulation impractical

#### 2. Acting Humanly: The Turing Test Approach

**Core Idea**: Build systems that behave like humans (even if internal processes differ).

**The Turing Test** (proposed by Alan Turing in 1950):
- A computer passes if a human interrogator cannot distinguish its responses from a human's
- Conducted through text-based conversation to avoid physical appearance issues
- Tests natural language processing, knowledge representation, automated reasoning, and learning

**Required Capabilities**:
- **Natural Language Processing**: Understand and generate human language
- **Knowledge Representation**: Store and organize information
- **Automated Reasoning**: Use stored information to answer questions and draw conclusions
- **Machine Learning**: Adapt to new circumstances and detect patterns

**Total Turing Test** (includes physical interaction):
- Adds **Computer Vision**: Perceive objects
- Adds **Robotics**: Manipulate objects and move

**Limitations**:
- Passing the Turing Test doesn't guarantee genuine intelligence
- Systems can "fake" intelligence through clever tricks
- Human-like behavior isn't always the best measure of intelligence

#### 3. Thinking Rationally: The "Laws of Thought" Approach

**Core Idea**: Use formal logic to build intelligent systems.

**Historical Foundation**:
- Aristotle's syllogisms (c. 384-322 BCE) provided the first formal system for rational thinking
- Example: "All humans are mortal; Socrates is human; Therefore, Socrates is mortal"

**Modern Implementation**:
- Use formal logic (propositional logic, first-order logic) to represent knowledge
- Apply logical inference rules to derive new knowledge
- Solve problems through logical deduction

**Logic Programming**:
Languages like Prolog attempt to encode problems as logical statements and use automated theorem-proving to find solutions.

**Why This Works**:
- Logic provides a precise, unambiguous language for representing knowledge
- Inference rules guarantee correct conclusions from correct premises
- Mathematical rigor ensures reproducible results

**Limitations**:
- Not all intelligent behavior can be expressed in formal logic
- Computational complexity: many logical inference problems are intractable
- Uncertainty and incomplete information are difficult to handle
- Real-world knowledge is often informal and context-dependent

#### 4. Acting Rationally: The Rational Agent Approach

**Core Idea**: Build systems that act to achieve the best expected outcome.

**Why This is the Dominant Paradigm**:
This approach has become the standard in modern AI because it:
- Focuses on outcomes rather than internal processes
- Works with uncertain and incomplete information
- Handles situations where perfect rationality is impossible (bounded rationality)
- Allows mathematical analysis and design principles

**Rationality Defined**:
A rational agent:
- Perceives its environment
- Reasons about its perceptions
- Acts to maximize its expected performance measure
- Does the "right thing" given its knowledge and capabilities

**Key Advantages**:
1. **More General**: Encompasses all forms of intelligence, not just human-like
2. **Scientifically Rigorous**: Allows mathematical analysis and formal methods
3. **Goal-Oriented**: Focuses on achieving objectives efficiently
4. **Flexible**: Adapts to various domains and constraints

**Limited Rationality**:
Perfect rationality (always making the optimal decision) is often impossible due to:
- Computational limitations
- Time constraints
- Incomplete information
- Uncertain outcomes

Therefore, AI often aims for **bounded rationality**: making the best decision possible given available computational resources and time.

---

## Intelligent Agents

### What is an Agent?

**Definition**: An agent is anything that can perceive its **environment** through **sensors** and act upon that environment through **actuators**.

**Key Components**:
- **Environment**: The world in which the agent operates
- **Sensors**: Input devices that perceive the environment (cameras, microphones, keyboards, etc.)
- **Actuators**: Output devices that affect the environment (motors, displays, speakers, etc.)
- **Agent Function**: Maps percept sequences to actions
- **Agent Program**: Concrete implementation running on physical hardware

### The PEAS Framework

To fully specify an agent's task environment, we use the **PEAS** framework:

#### P - Performance Measure
How we judge the agent's success. This must be:
- **Objective**: Measurable and clear
- **Complete**: Captures what we actually want
- **External**: Based on environmental outcomes, not internal agent states

**Examples**:
- Self-driving car: Safety, efficiency, legality, passenger comfort
- Chess program: Win/loss/draw outcomes
- Vacuum cleaner: Amount of dirt cleaned, energy consumed, time taken

**Critical Consideration**: Be careful what you wish for! The agent will optimize for what you measure, which may not be what you actually want (e.g., a cleaning robot might hide dirt rather than clean it if only "visible dirt" is measured).

#### E - Environment
The world in which the agent operates.

**Environment Properties**:

1. **Fully Observable vs. Partially Observable**
   - Fully: Agent can see complete state (e.g., chess board)
   - Partially: Agent sees only part of the state (e.g., poker, real-world navigation)

2. **Deterministic vs. Stochastic**
   - Deterministic: Next state completely determined by current state and action
   - Stochastic: Randomness or uncertainty in outcomes

3. **Episodic vs. Sequential**
   - Episodic: Each action is independent (e.g., image classification)
   - Sequential: Current decisions affect future options (e.g., chess)

4. **Static vs. Dynamic**
   - Static: Environment doesn't change while agent deliberates
   - Dynamic: Environment changes during agent's decision-making

5. **Discrete vs. Continuous**
   - Discrete: Finite, distinct states and actions
   - Continuous: Infinite, gradual variations in state and action

6. **Single-Agent vs. Multi-Agent**
   - Single: Only one agent in the environment
   - Multi-Agent: Multiple agents that may cooperate or compete

**Why This Matters**: Environment properties dramatically affect algorithm choice. For example:
- Partially observable environments require memory (agent must track what it has seen)
- Stochastic environments need probabilistic reasoning
- Multi-agent environments require game theory and strategic thinking

#### A - Actuators
Physical mechanisms through which the agent affects its environment.

**Examples**:
- Robot: Motors, grippers, wheels
- Software agent: Display outputs, network messages, file operations
- Self-driving car: Steering, acceleration, brakes, turn signals
- Game-playing AI: Move selection outputs

#### S - Sensors
Physical devices that provide percepts from the environment.

**Examples**:
- Robot: Cameras, LIDAR, touch sensors, microphones
- Software agent: Keyboard inputs, file contents, network data
- Self-driving car: GPS, cameras, radar, speedometer
- Game-playing AI: Board state, opponent moves

### Example: Self-Driving Car

**Performance Measure**:
- Safety (no accidents, follows traffic laws)
- Efficiency (reach destination quickly, minimize fuel)
- Comfort (smooth acceleration/braking)
- Legality (obey all traffic rules)

**Environment**:
- Roads, other vehicles, pedestrians, traffic signals
- Weather conditions, lighting conditions
- Partially observable (can't see around corners)
- Stochastic (unpredictable human drivers)
- Dynamic (constantly changing)
- Continuous (smooth motion)
- Multi-agent (other drivers)

**Actuators**:
- Steering wheel, accelerator, brake
- Turn signals, lights, horn

**Sensors**:
- Cameras, LIDAR, radar, ultrasonic sensors
- GPS, IMU (inertial measurement unit)
- Speedometer, wheel encoders

### Types of Agents

Agents can be classified by their internal structure and decision-making process, from simplest to most complex:

#### 1. Simple Reflex Agents

**How They Work**:
- Use condition-action rules (if-then rules)
- Choose actions based only on the current percept
- No memory of past percepts

**Structure**:
```
if [condition] then [action]
```

**Example**:
- Thermostat: If temperature < 20°C, turn on heater
- Automatic door: If motion detected, open door

**Advantages**:
- Simple to implement
- Fast decision-making
- Minimal memory requirements

**Limitations**:
- Only works in fully observable environments
- Cannot handle situations requiring memory or planning
- Can get stuck in loops (e.g., if sensor readings oscillate)

**When to Use**:
- Simple, well-defined tasks
- Fully observable environments
- Fast response required
- Low computational resources

#### 2. Model-Based Reflex Agents

**How They Work**:
- Maintain an **internal state** representing aspects of the environment that aren't currently visible
- Update internal state based on:
  - How the world changes over time (world model)
  - How actions affect the world (action model)

**Why This is Better**:
- Handles partially observable environments
- Remembers relevant past information
- Can reason about things not currently perceived

**Example**:
- Robot navigation: Remembers locations it has already explored
- Game AI: Remembers opponent's previous strategies
- Email filter: Remembers patterns from previous emails

**Key Concept - Internal State**:
The internal state tries to answer: "What is the state of the world I cannot currently see?"

This requires two types of knowledge:
1. **Transition Model**: How does the world change over time?
2. **Sensor Model**: How do my percepts relate to the world state?

**Limitations**:
- Still uses condition-action rules (reactive behavior)
- Doesn't consider future consequences of actions
- Cannot plan ahead

**When to Use**:
- Partially observable environments
- Need to track history or hidden state
- Still want fast, reactive responses

#### 3. Goal-Based Agents

**How They Work**:
- Have explicit **goal** information
- Consider future states resulting from actions
- Choose actions that lead toward goal achievement
- Use **search** and **planning** algorithms

**Key Difference**:
Instead of fixed condition-action rules, the agent asks: "What will happen if I do action X? Will it get me closer to my goal?"

**Example**:
- GPS navigation: Goal is destination; finds path to reach it
- Chess AI: Goal is checkmate; plans moves to achieve it
- Robotic assembly: Goal is assembled product; sequences actions appropriately

**Advantages**:
- Flexible: Can handle new goals without reprogramming
- Plans multiple steps ahead
- Adapts to changing goals

**Challenges**:
- Computationally expensive (must search/plan)
- Requires knowledge of action consequences
- May be too slow for real-time requirements

**When to Use**:
- Complex tasks requiring planning
- Goals change over time
- Path to goal is not obvious
- Have time to deliberate

#### 4. Utility-Based Agents

**How They Work**:
- Have a **utility function** that maps states to "happiness" or "desirability"
- Choose actions that maximize expected utility
- Can reason about trade-offs between conflicting goals

**Why Utility vs. Just Goals?**:
Goals are binary (achieved or not), but utility allows nuanced preferences:
- Multiple ways to achieve a goal (which is better?)
- Multiple conflicting goals (how to balance?)
- Uncertainty in outcomes (which action is safest?)
- Resource constraints (what's the best use of limited resources?)

**Utility Function**:
$U: States \rightarrow \mathbb{R}$

Maps each possible state to a real number representing its desirability.

**Example**:
Self-driving car utility function might consider:
- Safety: -1000 for accidents
- Speed: +1 per km/h average speed
- Fuel: -0.5 per liter used
- Comfort: -5 per harsh brake/acceleration

**Expected Utility**:
In uncertain environments, choose the action that maximizes **expected utility**:

$EU(a) = \sum_s P(s|a) \cdot U(s)$

Where:
- $a$ is an action
- $s$ is a possible resulting state
- $P(s|a)$ is the probability of reaching state $s$ given action $a$
- $U(s)$ is the utility of state $s$

**Advantages**:
- Handles uncertainty and conflicting objectives
- Provides rational basis for decision-making
- Can make complex trade-offs

**Challenges**:
- Designing the right utility function is difficult
- Computing expected utility can be expensive
- May be hard to explain decisions to humans

**When to Use**:
- Multiple, possibly conflicting objectives
- Uncertain outcomes
- Need to make trade-offs
- Want provably optimal decisions

#### 5. Learning Agents

**How They Work**:
All previous agent types can be augmented with **learning**:
- Start with initial knowledge/behavior
- Observe outcomes of actions
- Improve performance over time

**Components of a Learning Agent**:

1. **Learning Element**: Improves the agent's knowledge/behavior
2. **Performance Element**: Selects actions (this is one of the agent types above)
3. **Critic**: Provides feedback on how well the agent is doing
4. **Problem Generator**: Suggests exploratory actions to learn new things

**Why Learning?**:
- Designer may not know complete solution in advance
- Environment may be too complex to fully specify
- Environment may change over time
- Agent can adapt to new situations

**Example**:
- Game-playing AI: Learns winning strategies through self-play
- Recommendation system: Learns user preferences from clicks/ratings
- Robot: Learns walking gait through trial and error

**When to Use**:
- Environment is too complex to program completely
- Environment changes over time
- Agent needs to personalize to specific users/contexts
- Have training data or opportunity for exploration

---

## Introduction to Responsible and Ethical AI

### Why Ethics Matters in AI

AI systems increasingly impact human lives in significant ways:
- Medical diagnosis and treatment recommendations
- Criminal justice (bail decisions, sentencing)
- Financial services (loan approvals, insurance rates)
- Employment (resume screening, performance evaluation)
- Content moderation and information filtering

**Key Insight**: AI systems are not neutral—they reflect the values, biases, and priorities embedded in their design, training data, and deployment.

### Core Ethical Principles

#### 1. Fairness and Non-Discrimination

**The Problem**:
AI systems can perpetuate or amplify existing biases:
- Facial recognition systems that work poorly for certain demographics
- Hiring algorithms that discriminate by gender or race
- Credit scoring that disadvantages certain zip codes
- Medical AI that under-diagnoses certain populations

**Sources of Bias**:
1. **Training Data Bias**: Data reflects historical discrimination
2. **Measurement Bias**: What we measure may not capture what we care about
3. **Representation Bias**: Some groups under-represented in data
4. **Evaluation Bias**: Test sets don't reflect real-world diversity
5. **Deployment Bias**: System used in contexts different from training

**Approaches to Fairness**:
- **Demographic Parity**: Equal outcomes across groups
- **Equal Opportunity**: Equal true positive rates across groups
- **Individual Fairness**: Similar individuals treated similarly
- **Calibration**: Predictions equally accurate across groups

**Challenge**: Different fairness definitions can be mathematically incompatible!

#### 2. Transparency and Explainability

**The Problem**:
Many AI systems (especially deep learning) are "black boxes"—even their creators can't fully explain specific decisions.

**Why Explainability Matters**:
- **Trust**: Users need to understand why a decision was made
- **Debugging**: Developers need to identify and fix errors
- **Legal Compliance**: Some domains require explainable decisions
- **Fairness**: Can't detect discrimination without understanding reasoning
- **Safety**: Must understand failure modes

**Approaches**:
- **Intrinsically Interpretable Models**: Decision trees, linear models
- **Post-hoc Explanation**: LIME, SHAP, attention visualization
- **Example-Based**: Show similar cases
- **Counterfactual**: "Decision would change if X were different"

**Trade-offs**:
- More interpretable models often have lower accuracy
- Perfect explanation may not be possible for complex tasks
- Human-understandable explanations may be oversimplifications

#### 3. Accountability and Responsibility

**The Problem**:
When an AI system causes harm, who is responsible?
- The developer who created the algorithm?
- The company that deployed it?
- The user who applied it?
- The AI system itself?

**Key Questions**:
- Who should be liable for AI mistakes?
- How do we audit AI systems for compliance?
- What oversight mechanisms are appropriate?

**Approaches**:
- Clear documentation of system capabilities and limitations
- Human-in-the-loop for high-stakes decisions
- Regular audits and impact assessments
- Incident reporting and investigation procedures

#### 4. Privacy and Data Protection

**The Problem**:
AI systems often require vast amounts of personal data:
- Training data may contain sensitive information
- Models can memorize training examples
- Inferences can reveal information users didn't consent to share
- Data breaches expose sensitive information

**Risks**:
- **Re-identification**: Anonymous data can be de-anonymized
- **Inference**: AI can infer sensitive attributes (health, beliefs) from innocuous data
- **Linkage**: Combining datasets reveals more than any single dataset
- **Surveillance**: AI enables large-scale monitoring

**Approaches**:
- **Data Minimization**: Collect only what's needed
- **Differential Privacy**: Add noise to protect individuals
- **Federated Learning**: Train without centralizing data
- **Encryption**: Protect data in transit and at rest
- **User Control**: Allow users to access, correct, delete their data

#### 5. Safety and Security

**The Problem**:
AI systems can fail in unexpected ways:
- Adversarial examples: Small, intentional perturbations cause misclassification
- Reward hacking: Agent finds loopholes in reward function
- Specification gaming: Achieves stated goal in unintended way
- Distribution shift: Performance degrades on new data
- Security vulnerabilities: AI systems can be hacked or poisoned

**Examples of Safety Failures**:
- Self-driving car fails to detect pedestrian in unusual context
- Chatbot learns offensive behavior from users
- Recommendation algorithm promotes harmful content
- Trading algorithm causes market crash

**Approaches**:
- **Robustness Testing**: Test on diverse, adversarial examples
- **Formal Verification**: Mathematically prove safety properties
- **Safe Exploration**: Learn without causing harm
- **Human Oversight**: Keep humans in the loop for critical decisions
- **Fail-Safes**: Design systems to fail gracefully

#### 6. Beneficial AI and Value Alignment

**The Problem**:
Ensuring AI systems actually pursue goals that benefit humanity:
- Short-term optimization may harm long-term wellbeing
- Individual optimization may harm collective good
- Specified goals may not capture true human values
- Advanced AI may pursue goals in unexpected ways

**The Alignment Problem**:
How do we ensure AI systems do what we actually want, not just what we say we want?

**Challenges**:
- Human values are complex, context-dependent, and varied
- We may not be able to fully specify what we want
- Our stated preferences may not match our revealed preferences
- Values change over time and across cultures

**Approaches**:
- **Inverse Reinforcement Learning**: Learn values from human behavior
- **Debate/Amplification**: AI systems argue for transparency
- **Cooperative Inverse Reinforcement Learning**: AI is uncertain about goals
- **Human Feedback**: Incorporate ongoing human guidance

### Ethical Frameworks for AI Development

#### Consequentialism
Judge AI systems by their outcomes and consequences.
- **Utilitarianism**: Maximize overall wellbeing
- **Risk Assessment**: Weigh potential benefits against harms

#### Deontology
Judge AI systems by adherence to rules and duties.
- **Rights-Based**: Respect fundamental human rights
- **Professional Ethics**: Follow codes of conduct
- **Legal Compliance**: Obey relevant laws and regulations

#### Virtue Ethics
Focus on the character and intentions of AI developers and deployers.
- **Trustworthiness**: Build systems that earn public trust
- **Responsibility**: Take ownership of systems' impacts
- **Justice**: Commit to fair and equitable AI

### Practical Considerations

#### Throughout the Development Lifecycle

**Design Phase**:
- Consider potential harms and benefits
- Involve diverse stakeholders
- Consider alternative approaches

**Data Collection**:
- Obtain informed consent
- Ensure representative data
- Protect privacy

**Model Development**:
- Test for bias and fairness
- Document decisions and trade-offs
- Consider interpretability requirements

**Deployment**:
- Monitor real-world performance
- Provide clear communication about capabilities/limitations
- Establish accountability mechanisms
- Create feedback channels

**Maintenance**:
- Regular audits and updates
- Respond to incidents and complaints
- Adapt to changing contexts

### Current Debates and Open Questions

1. **Regulation**: Should AI be regulated? If so, how?
2. **Automation**: Which jobs/decisions should remain human?
3. **Dual Use**: How to prevent beneficial AI from being weaponized?
4. **Global Governance**: How to coordinate across nations?
5. **Long-term Risks**: How to prepare for highly advanced AI?

### Key Takeaways

1. **Ethics is Integral**: Not an afterthought—consider from the start
2. **No Perfect Solutions**: Trade-offs are inevitable
3. **Context Matters**: Appropriate approaches vary by domain
4. **Ongoing Process**: Ethics requires continual attention
5. **Interdisciplinary**: Requires technical, philosophical, legal, and social perspectives

---

## Summary

Module 1 establishes the foundations of AI:
- **Definition**: AI as systems that act rationally to achieve goals
- **Agents**: The rational agent framework for designing intelligent systems
- **Ethics**: The responsibility to develop AI that benefits humanity

These concepts underpin all subsequent topics in the course. The rational agent paradigm provides a unifying framework, while ethical considerations must be woven throughout every aspect of AI development.

**Key Insights**:
1. Intelligence is about achieving goals, not mimicking humans
2. Agent design requires careful specification of goals, environment, sensors, and actuators
3. Different agent architectures suit different types of problems
4. Ethics is not optional—it's essential for responsible AI development

## Further Reading

- Russell & Norvig, "Artificial Intelligence: A Modern Approach," Chapters 1-2
- Turing, A. (1950). "Computing Machinery and Intelligence"
- Wallach & Allen, "Moral Machines: Teaching Robots Right from Wrong"
- O'Neil, "Weapons of Math Destruction"
- Eubanks, "Automating Inequality"

