|                | University    | of             | Trieste         |               |           |
| -------------- | ------------- | -------------- | --------------- | ------------- | --------- |
|                | Data Science  | and Artificial | Intelligence    |               |           |
|                | Mathematics,  | Informatics    | and Geosciences |               |           |
|                | Reinforcement |                | Learning        |               |           |
|                |               | Course         | Notes           |               |           |
| Author:        |               |                |                 |               | Lecturer: |
| Ester De Giosa |               |                |                 | Prof. Antonio | Celani    |
Alina Fogar
|     |     |     |     | October | 19, 2026 |
| --- | --- | --- | --- | ------- | -------- |
Riccardo Samaritan
migesolidale@gmail.com

Contents
1 Introduction 3
1.1 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.1.1 Elements of reinforcement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.1.2 Temporal Difference Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
1.1.3 Multi-Armed Bandit Problem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2 Markov Decision Processes 8
2.1 Interface . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2.2 Markov Decision Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.3 Fully Observable Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
2.4 Partially Observable Markov Decision Process (POMDP) . . . . . . . . . . . . . . . . . . . . . . . 10
2.4.1 Observation Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.4.2 Perfect Memory and Information State. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
2.4.3 Belief States and the Belief-MDP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
2.4.4 Bellman Equation in Belief Space . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.5 Goals and Rewards . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.6 Returns and Episodes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
2.7 Unified Notation for Episodic and Continuing Tasks . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.8 Policies and Value Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
2.9 Optimal Policies and Value Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
3 Dynamic Programming 17
3.1 Policy Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
3.2 Policy Improvement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
3.3 Policy Iteration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
3.4 Comparison with Value Iteration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
3.5 Value Iteration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
3.6 Asynchronous DP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
3.7 Generalized Policy Iteration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
3.8 Linear Programming for MDP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
4 Monte Carlo Methods 25
4.1 Monte Carlo Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
4.2 Monte Carlo Estimation of Action Values . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
4.3 Monte Carlo Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
4.4 Monte Carlo Control without Exploring Starts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
4.5 Off-policy Prediction via Importance Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
4.6 Incremental Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
4.7 Off-policy Monte Carlo Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
5 Temporal-Difference Learning 31
5.1 TD Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
5.2 Advantages of TD Prediction Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5.3 Optimality of TD(0) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5.4 Sarsa: On-policy TD Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
5.5 Q-learning: Off-policy TD Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
5.6 Expected Sarsa . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
5.7 Maximization Bias and Double Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
6 n-step Bootstrapping 37
1

6.1 n-step TD Prediction. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
6.2 n-step Sarsa . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
6.3 n-step Off-policy Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
7 Planning and Learning with Tabular Methods 41
7.1 Models and Planning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
7.2 Dyna: Integrated Planning, Acting, and Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
7.3 When the Model Is Wrong . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
7.4 Prioritized Sweeping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
7.5 Expected vs. Sample Updates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
7.6 Trajectory Sampling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
7.7 Real-time Dynamic Programming . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
7.8 Planning at Decision Time. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
7.9 Heuristic Search . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
7.10 Rollout Algorithms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
7.11 Monte Carlo Tree Search. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
8 Approximate Solution Methods 50
8.1 Value-function Approximation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
8.2 The Prediction Objective (VE) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
8.3 Stochastic-gradient and Semi-gradient Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
8.3.1 Stochastic Approximation and Mean-Square Convergence . . . . . . . . . . . . . . . . . . . 52
8.4 Linear Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
8.5 Feature Construction for Linear Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
9 Policy Gradient Methods 56
9.1 Policy Approximation and its Advantages . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
9.2 Policy Gradient Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 56
9.3 REINFORCE: Monte Carlo Policy Gradient . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57
9.3.1 REINFORCE with Baseline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 58
9.3.2 Advantage Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
9.3.3 Actor-Critic Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
10 Advanced Policy Gradient Notes 60
10.1 Derivation of the Policy Gradient Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
10.1.1 Setup and Parameterization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
10.1.2 Matrix Calculus Identity. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
10.1.3 Derivation of the Policy Gradient Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
10.1.4 The Log-Derivative Trick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
10.2 Global Optima in Markov Decision Processes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
10.2.1 Background and Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
10.2.2 The Performance Difference Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 62
10.2.3 Local and Global Optima . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 63
11 Multi-Agent Reinforcement Learning 64
11.1 General Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.1.1 Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.1.2 Common Approaches . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
11.1.3 Relation with Game Theory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
2

Chapter 1
Introduction
The fundamental building blocks of any decision-making system are the agent and the environment. The agent
perceives the world through sensors, which capture both the external state of the environment and the internal
state of the agent itself, and acts upon it through actuators - the components that translate intentions into
consequences.
The agent’s internal state at time t is denoted m , often referred to as memory. Its size may vary and can be
t
decomposed into sub-variables. The environment’s state at time t is denoted S . Together, these variables fully
t
describe the system at any given moment.
Time is treated as discrete (t → t+1), although working with continuous time remains possible. At each step,
the agent takes an action a — an intention that must confront reality. Two consequences follow:
t
• the state of the environment (and possibly the agent) changes;
• an observation y is produced, which is used to update the internal state: m →m .
t t t+1
This process then repeats. The agent’s behavior is governed by a model of the environment — a representation
we must choose carefully, as it encodes what the agent knows or assumes about the world.
1.1 Reinforcement Learning
Reinforcement Learning (RL) is the problem of learning what to do in order to maximize a numerical reward
signal. Theagentinterractswiththeenviromentwithoutbeingtoldwhichactionstotake;instead,itmustdiscover
which actions yield the most reward by trying them out. This distinguishes RL from supervised learning, where
theagentlearnsfromlabeledinput-outputpairs,andfromunsupervisedlearning,wherethegoalistofindpatterns
in data without explicit feedback. RL is therefore considered a third machine learning paradigm.
A central challenge is the trade-off between exploration and exploitation: the agent must explore the environ-
ment to discover rewarding actions, while simultaneously exploiting its current knowledge to maximize reward.
All RL agents have explicit goals, can sense aspects of their environment, and can choose actions to influence it
— making RL an approach that starts with a complete, interactive, goal-seeking agent.
1.1.1 Elements of reinforcement
• Policy π(a|m): A strategy that maps perceived states of the environment to actions. In simple cases it may
be a lookup table; in others it may involve extensive computation. Policies may be stochastic, specifying a
probability distribution over actions.
• Reward signal R : A scalar feedback signal received from the environment after each action. It evaluates
t+1
the effectiveness of the agent’s behavior and is the primary basis for updating the policy.
• Value function: A function that estimates the expected cumulative reward obtainable from a given state (or
state-action pair). Unlike the reward signal - which reflects immediate gain — the value function captures
long-term consequences, guiding the agent toward decisions that maximize expected return.
• Model of the environment p(S′,y |S,a): An internal representation of the environment used to predict the
consequences of actions. It enables planning by simulating future states, though it is not always required.
• Memory update g(m′ |m,a,y): The computation that updates the agent’s internal state given the previous
memory, the action taken, and the observation received.
3

The goal is to find the optimal policy π∗ that maximizes the expected cumulative reward:
|     |     |     | (cid:34) | (cid:35) |     |     |
| --- | --- | --- | -------- | -------- | --- | --- |
T
|     |     |     | E (cid:88) |     |     |     |
| --- | --- | --- | ---------- | --- | --- | --- |
|     |     |     | max        | R   |     |     |
|     |     |     | π          | t+1 |     |     |
t=0
Most of the time the agent acts greedily, selecting the action that leads to the state with the highest estimated
value — i.e., the highest estimated probability of winning. Occasionally, however, it selects randomly from among
other actions. These are called exploratory moves, as they expose the agent to states it might otherwise never
visit.
|     | Taxonomy | of Reinforcement |     | Learning | & Planning |     |
| --- | -------- | ---------------- | --- | -------- | ---------- | --- |
MonteCarlo
full
|     |     | TemporalDifference |     | MDP |     |     |
| --- | --- | ------------------ | --- | --- | --- | --- |
Policygradient
ytilibavresbO
|     |     | Full | RL  | POMDP |     |     |
| --- | --- | ---- | --- | ----- | --- | --- |
none
|                |            |            | epistemic | knowledge |             |     |
| -------------- | ---------- | ---------- | --------- | --------- | ----------- | --- |
|                |            | model-free |           |           | model-based |     |
|                |            | learning   |           | planning  |             |     |
| 1.1.2 Temporal | Difference | Learning   |           |           |             |     |
Let S denote the state before the greedy move, and S the state after. The update to the estimated value of
| t               |        |     | t+1           |       |     |     |
| --------------- | ------ | --- | ------------- | ----- | --- | --- |
| S , denoted V(S | ), is: |     |               |       |     |     |
| t               | t      |     |               |       |     |     |
|                 |        | V(S | )←V(S )+α(V(S | )−V(S | ))  |     |
|                 |        | t   | t             | t+1 t |     |     |
where α is a small positive learning rate. This is known as the temporal difference (TD) learning rule: it
updates value estimates based on the difference between the predicted value of the next state and the current
| estimate, without | waiting | for the final outcome. |     |     |     |     |
| ----------------- | ------- | ---------------------- | --- | --- | --- | --- |
| 1.1.3 Multi-Armed |         | Bandit Problem         |     |     |     |     |
The most important feature distinguishing RL from other types of learning is that it uses training information
that the actions taken rather than by giving correct actions. This is what creates the need for
| evaluates           |        |                     | instructs      |     |     |     |
| ------------------- | ------ | ------------------- | -------------- | --- | --- | --- |
| active exploration, | for an | explicit search for | good behavior. |     |     |     |
Let’s say we have a slot machine with multiple arms (k) and we are forced to repeatedly choose one arm to
pull. After each choice we receive a numerical reward chosen from a stationary (strong assumption!) probability
distribution that depends on the action we selected. Our objective is to maximize the expected total reward over
| some time period, | i.e. time | steps. |     |     |     |     |
| ----------------- | --------- | ------ | --- | --- | --- | --- |
In this k-armed bandit problem, each of the k actions has an expected or mean reward given that that action is
selected; thisisthevalue ofthataction. WedenotetheactionselectedontimesteptasA andthecorresponding
t
reward as R . The value then of an arbitrary action a, denoted q (a), is the expected reward given that a is
|     | t   |     |     | ∗   |     |     |
| --- | --- | --- | --- | --- | --- | --- |
selected:
|     |     |     | q (a)≡E[R | |A =a] |     |     |
| --- | --- | --- | --------- | ------ | --- | --- |
|     |     |     | ∗ t       | t      |     |     |
We assume that we don’t know the action values with certainty, although we may have estimates. We denote the
estimated value of action a at time step t as Q (a). We would like Q (a) to be close to q (a), but we do not know
|     |     |     | t   | t   |     | ∗   |
| --- | --- | --- | --- | --- | --- | --- |
the true values.
4

Figure 1.1: Tic Tac Toe example of a reinforcement learning agent. The agent learns to play the game by exploring
| different | moves and receiving | rewards | based on the outcome | of the game. |
| --------- | ------------------- | ------- | -------------------- | ------------ |
It we maintain estimates of the action values, then at any time step there is at least one action whose estimated
value is greatest. We call these the greedy actions. When we select one of these actions, we say that we are
our current knowledge of the values of the actions. If instead we select one of the nongreedy actions,
exploiting
then we say we are exploring, because this enables us to improve our estimate of the nongreedy action’s value.
Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may
| produce | the greater total | reward in | the long run. |     |
| ------- | ----------------- | --------- | ------------- | --- |
However, most of these methods make strong assumptions about stationarity and prior knowledge that are either
violatedorimpossibletoverifyinmostapplicationsandinthefullreinforcementlearningproblemthatweconsider
insubsequentchapters. Theguaranteesofoptimalityorboundedlossforthesemethodsareoflittlecomfortwhen
| the assumptions | of their | theory do not | apply. |     |
| --------------- | -------- | ------------- | ------ | --- |
| Action-value    | Methods  |               |        |     |
Action-valuemethodsareaclassofreinforcementlearningalgorithmsthatfocusonestimatingthevalueofactions
based on the rewards received from the environment. The key idea is to maintain an estimate of the expected
reward for each action and to use this information to make decisions about which action to take next.
(cid:80)t−1R ·⊮(A
i i =a)
|     |     |     | Q (a)≡ | i=1            |
| --- | --- | --- | ------ | -------------- |
|     |     |     | t      | (cid:80)t−1⊮(A |
i =a)
i=1
We call this the method for estimating action values becuase each estimate is an average of the
sample-average
sample of relevant rewards. To select an action, the simplest approach is to select one of the actions with the
highest estimated value, i.e., one of the greedy actions as defined in the previous section. We write this
greedy
| action selection | method | as  |             |     |
| ---------------- | ------ | --- | ----------- | --- |
|                  |        |     | A ≡argmaxaQ | (a) |
|                  |        |     | t           | t   |
A simple alternative is to behave greedily most of the time, but every once in a while, with a small probability ϵ,
instead select randomly from all the actions with equal probability, independently of the action-value estimates.
| These methods | are named | ϵ-greedy. |     |     |
| ------------- | --------- | --------- | --- | --- |
| Nonstationary | Bandit    | Problems  |     |     |
Previous methods assume that the action values are stationary, meaning that they do not change over time.
However, in many real-world scenarios, the action values can change due to various factors such as changes in the
environment or the agent’s own actions. This leads to nonstationary bandit problems, where the action values
5

Figure 1.2: Comparison between greedy and ϵ-greedy action selection methods. The ϵ-greedy method explores the
environmentbyselectingrandomactionswithasmallprobabilityϵ,whilethegreedymethodalwaysselectstheactionwith
| the highest | estimated value. |     |     |     |     |
| ----------- | ---------------- | --- | --- | --- | --- |
are not fixed and can vary over time. In such cases it makes sense to give more weight to recent rewards than to
long-past rewards. One of the most popular ways of doing this is to use a constant step-size parameter.
|     |     |     | Q (a)=Q | (a)+α(R | −Q (a)) |
| --- | --- | --- | ------- | ------- | ------- |
|     |     |     | n+1     | n       | n+1 n   |
where α ∈ [0,1] is a constant step-size parameter that determines how much weight to give to the most recent
reward. This approach allows the agent to adapt to changes in the environment and to update its action-value
| estimates  | accordingly.   |     |     |     |     |
| ---------- | -------------- | --- | --- | --- | --- |
| Optimistic | Initial Values |     |     |     |     |
Allthemethodswehavediscussedsofararedependenttosomeextentontheinitialaction-valueestimates,Q (a).
1
In the language of statistics, these methods are biased by their initial estimates.
Initial action-values can also be used as a simple way to encourage exploration. Setting the initial action values
to be optimistic, i.e., higher than the true expected rewards, can encourage the agent to explore more. This is
because the agent will initially prefer actions with higher estimated values, leading to more exploration of those
actions until their true values are learned. We regard it as a simple trick that can be quite effective on stationary
problems, but it is far from being a generally useful approach to encouraging exploration.
| Gradient | Bandit Algorithms |     |     |     |     |
| -------- | ----------------- | --- | --- | --- | --- |
Here we consider learning a numerical for each action a, which we denote as (a)∈R. The larger the
preference H t
preference, the more often that action is taken, but the preference has no interpretation in terms of reward. Only
| the relative | preference | of one action | over another | is important. |     |
| ------------ | ---------- | ------------- | ------------ | ------------- | --- |
eHt(a)
|     |     |     | Pr{A =a}= |                 | =π (a) |
| --- | --- | --- | --------- | --------------- | ------ |
|     |     |     | t         | (cid:80) eHt(b) | t      |
b
6

Figure 1.3: Optimistic initial values in action-value methods. The agent starts with high initial estimates for all actions,
| encouraging | exploration until | the true values | are learned. |     |     |     |
| ----------- | ----------------- | --------------- | ------------ | --- | --- | --- |
where π (a) indicates the probability of taking action a at time step t. The action probabilities are normalized so
t
| that they | sum to one. |     |     |     |     |     |
| --------- | ----------- | --- | --- | --- | --- | --- |
There is a natural learning algorithm for soft-max action preferences based on the idea of stochastic gradient
ascent. On each step, after selecting action and receiving the reward , the action preferences are updated
|     |     |     | A   |     | R   |     |
| --- | --- | --- | --- | --- | --- | --- |
|     |     |     | t   |     | t   |     |
by:
|     |     |             |              | −R¯        | and          |     |
| --- | --- | ----------- | ------------ | ---------- | ------------ | --- |
|     |     | H t+1 (a)=H | t (a)+α(R    | t t )(1−π  | t (A t )),   |     |
|     |     | H (A        | )=H (A )−α(R | −R¯ )π (a) | for all a̸=A |     |
|     |     | t+1 t       | t t          | t t t      |              | t   |
where R¯ is the average reward received up to time step t. This update rule adjusts the preferences based on the
t
received reward, encouraging actions that yield higher rewards and discouraging those that yield lower rewards.
Figure1.4: Gradientbanditalgorithmsforactionselection. Theagentlearnspreferencesforactionsbasedonthereceived
| rewards, | using a soft-max | policy to select | actions probabilistically. |     |     |     |
| -------- | ---------------- | ---------------- | -------------------------- | --- | --- | --- |
7

| Chapter | 2   |          |           |     |     |     |     |
| ------- | --- | -------- | --------- | --- | --- | --- | --- |
| Markov  |     | Decision | Processes |     |     |     |     |
A Markov Decision Process (MDP) is a classical formalization of sequential decision-making, where actions in-
fluence not just immediate rewards but also subsequent states and, through those, future rewards. Whereas in
bandit problems we estimated the value q (a) of each action a, in MDPs we estimate either q (s,a) - the value of
|     |     |     | ∗   |     |     | ∗   |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
each action in each state - or - the value of each state under optimal action selection.
v (s)
∗
2.1 Interface
The learner and decision-maker is called the agent. Everything it interacts with, comprising everything outside
the agent, is called the environment. These interact continually: the agent selects actions, and the environment
responds by presenting new situations and emitting rewards - special numerical values that the agent seeks to
| maximize    | over time. |                       |                 |     |       |     |     |
| ----------- | ---------- | --------------------- | --------------- | --- | ----- | --- | --- |
| The general | scheme     | of a Decision Process | is shown below: |     |       |     |     |
|             | state      | S                     |                 |     | S t+1 |     |     |
t
Environment
R
y
|     |     |     | a t | t   | Sensorimotor | Interface |     |
| --- | --- | --- | --- | --- | ------------ | --------- | --- |
g
π
Agent
|     | memory |     |     |     | m   |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- |
|     |        | m t |     |     | t+1 |     |     |
t
|                |                 | t                  |             |                  | t+1 |     |     |
| -------------- | --------------- | ------------------ | ----------- | ---------------- | --- | --- | --- |
| The four       | key components  | of the interface   | are:        |                  |     |     |     |
| • π(a|m):      | policy          |                    |             |                  |     |     |     |
| • R(y):        | reward function |                    |             |                  |     |     |     |
| • p(s′,y|s,a): | model           | of the environment | (transition | and observation) |     |     |     |
| • g(m′|m,a,y): | memory          | update             |             |                  |     |     |     |
Discount factor Furthermore, to incentivize the agent to complete tasks efficiently, we introduce a discount
(or survival probability) γ, with 1. At each time step the agent survives with probability and
| factor |     |     | 0 ≤ γ < |     |     |     | γ   |
| ------ | --- | --- | ------- | --- | --- | --- | --- |
transitions to a terminal absorbing state (a “coffin” state) with probability 1−γ.
| The expected | survival | time is: |     |     |     |     |     |
| ------------ | -------- | -------- | --- | --- | --- | --- | --- |
1
T =
eff 1−γ
8

The goal is to find the optimal policy π∗ that maximizes the expected discounted return:
|     |     |     |     |     |     |     | (cid:34) | (cid:35) |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --- |
∞
|     |     |     |     |     | π∗      | E         | (cid:88) γtR(y     |           |     |
| --- | --- | --- | --- | --- | ------- | --------- | ------------------ | --------- | --- |
|     |     |     |     |     | =argmax |           | t )                |           |     |
|     |     |     |     |     |         | π         | t=0                |           |     |
|     |     |     |     |     |         | (cid:124) | (cid:123)(cid:122) | (cid:125) |     |
ExpectedReturn
| The discount |     | factor |      | ensures | the infinite | sum converges. |     |     |     |
| ------------ | --- | ------ | ---- | ------- | ------------ | -------------- | --- | --- | --- |
|              |     |        | γ <1 |         |              |                |     |     |     |
Specifications:
• Perfect observability → the agent knows the state of the environment (y =S) and p(y|s,a,s′)=∞(y =s′)
Observation:
p(s′,y|s,a)=p(s′|s,a)p(y|s,a,s′)
• Memoryupdate→theagentknowsthestateoftheenvironmentandthememory(M =y)andg(m′|m,a,y)=
∞(m′ =y)
At each time step t, the agent receives some representation of the environment’s state S ∈ S, and on that basis
t
selects an A(s). One time step later, in part as a consequence of its action, the agent receives a
|     | action | A   | ∈   |     |     |     |     |     |     |
| --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
t
| numerical | reward | R   | ∈R⊂R,    |     | and finds itself | in a new | state S . |     |     |
| --------- | ------ | --- | -------- | --- | ---------------- | -------- | --------- | --- | --- |
|           |        |     | t+1      |     |                  |          | t+1       |     |     |
| 2.2       | Markov |     | Decision |     | Process          |          |           |     |     |
At each time step t, the agent receives a representation of the environment’s state S ∈ S, selects an action
t
∈A(s), and receives a scalar reward ∈R⊂R while transitioning to a new state .
| A t        |          |     |            |     | R t+1           |                  |               |             | S t+1  |
| ---------- | -------- | --- | ---------- | --- | --------------- | ---------------- | ------------- | ----------- | ------ |
| A          |          |     |            |     | is a            | fully observable | set of tuples |             | where: |
| Markov     | Decision |     | Process    |     | (MDP)           |                  |               | (S,A,R,P,γ) |        |
| • s∈S:     | a finite | set | of states  |     |                 |                  |               |             |        |
| • a∈A:     | a finite | set | of actions |     |                 |                  |               |             |        |
| • R:S×A→R: |          | the | (expected) |     | reward function |                  |               |             |        |
• →[0,1]: the transition probability function (model of the environment)
P :S×A×S
| • γ ∈[0,1]: | the | discount |       | factor       |     |     |     |     |     |
| ----------- | --- | -------- | ----- | ------------ | --- | --- | --- | --- | --- |
| • p (s):    | the | initial  | state | distribution |     |     |     |     |     |
0
| • π(a|s): | the | policy |     |     |     |     |     |     |     |
| --------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
In a finite MDP, the sets of states, actions, and rewards all have a finite number of elements.
|          |           |            |         |        | Figure   | 2.1: Markov   | Decision Process |                |            |
| -------- | --------- | ---------- | ------- | ------ | -------- | ------------- | ---------------- | -------------- | ---------- |
| 2.3      | Fully     | Observable |         |        | Setting  |               |                  |                |            |
| The most | tractable |            | setting | is the | one with |               | and              |                | knowledge: |
|          |           |            |         |        | perfect  | observability |                  | full epistemic |            |
• Perfect observability: the agent directly observes the environment’s state, i.e., y = S . Formally, p(y |
t t
| s,a,s′)=1(y |     | =s′), | so  | the joint | model factorizes | as:        |           |      |     |
| ----------- | --- | ----- | --- | --------- | ---------------- | ---------- | --------- | ---- | --- |
|             |     |       |     |           | p(s′,y           | |s,a)=p(s′ | |s,a)·1(y | =s′) |     |
Since the observation coincides with the new state, the model simplifies to p(s′ |s,a).
9

• Full epistemic knowledge: the agent has a correct and complete model of the environment — it knows a
priori all transition probabilities p(s′,y |s,a) and the reward r′(a,y) exactly.
• Memory update: since y = s , the agent simply copies the observation into memory: m′ = y, i.e.,
t t+1
g(m′ |m,a,y)=1(m′ =y). The memory at initialization is m =s , and subsequently m =y =s .
0 0 t t−1 t
In this setting (the top right corner of figure - we see everything, we know everything), the agent does not need to
maintain the full history of observations. Knowledge of the current state is sufficient for optimal decision-making.
The goal therefore reduces to finding an optimal stationary policy:
π∗(a|s),
which is a probabilistic mapping from environment states to actions.
In this case, the random variables R and S have well defined discrete probability distributions dependent only
t t
on the preceding state and action.
p(s′,r|s,a)=Pr(S =s′,R =r|S =s,A =a)
t t t−1 t−1
This function defines the dynamics of the MDP, it specifies a probability distribution for each choice of s and a
and has four arguments.
(cid:88) (cid:88)
p(s′,r|s,a)=1 for all s∈S,a∈A
s′∈Sr∈R
The state must include information about all aspects of the past agent-environment interaction that make a
difference for the future. If it does, then the state is said to have the Markov property.
From the four-argument dynamics function, one can obtain:
• the state-transition probabilities p:S×S×A→[0,1]
(cid:88)
p(s′|s,a)=Pr{S =s′|S =s,A =a}= p(s′,r|s,a)
t t−1 t−1
r∈R
• the expected rewards for state-action pairs r :S×A→R:
(cid:88) (cid:88)
r(s,a)=E[R |S =s,A =a]= r p(s′,r|s,a)
t t−1 t−1
r∈R s′∈S
• the expected rewards for state-action-next-state triples r :S×A×S →R:
(cid:88) p(s′,r|s,a)
r(s,a,s′)=E[R |S =s,A =a,S =s′]= r
t t−1 t−1 t p(s′|s,a)
r∈R
In general, actions can be any decisions we want to learn how to make, and states can be anything we can know
thatmightbeusefulinmakingthem. Inparticular, theboundarybetweenagentandenvironmentistypicallynot
the same as the physical boundary of a robot’s or an animal’s body. Usually, the boundary is drawn closer to the
agent for that.
Generally, anythingthatcannotbechangedarbitrarilybytheagentisconsideredtobeoutsideofitandthuspart
of its environment. We do not assume that everything in the environment is unknown to the agent.
The MDP framework proposes that whatever the details of the sensory, memory and control apparatus, and
whatever objective one is trying to achieve, any problem of learning goal-directed behavior can be reduced to
three signals passing back and forth between the agent and its environment:
• actions: choices made by the agent
• states: basis on which the choices are made
• rewards: the agent’s goal
2.4 Partially Observable Markov Decision Process (POMDP)
APartiallyObservableMarkovDecisionProcess(POMDP)generalizestheMDPframeworkforsituations
wheretheagentcannotdirectlyobservetheunderlyingstates . Instead, theagentreceivesanobservation y that
t t
provides partial (and potentially noisy) information about s .
t
In general, because the state is not directly observable, the current observation may not be sufficient for optimal
decision-making.
10

| Example: | Example: |     | 2-State, | 2-Action | MDP |     |     |     |
| -------- | -------- | --- | -------- | -------- | --- | --- | --- | --- |
Consider a system with two states S = {1,2} and two actions A = {1,2}. This example illustrates why partial
| observability | fundamentally |     | changes | the | optimal policy. |     |     |     |
| ------------- | ------------- | --- | ------- | --- | --------------- | --- | --- | --- |
|               |               |     |         |     | 1               |     |     | 1   |
1−δ,1
|              |     |             |             | 1, -1 |      |       |     | δ,−1  |
| ------------ | --- | ----------- | ----------- | ----- | ---- | ----- | --- | ----- |
|              |     |             |             | 1     |      |       |     | 2     |
|              |     |             |             |       | δ,−1 | 1−δ,1 |     | 1, -1 |
|              |     |             |             |       | 2    |       |     | 2     |
| The dynamics |     | are defined | as follows: |       |      |       |     |       |
• State 1: Action 2 is the "correct" move, transitioning to State 2 with probability 1−δ (Reward: +1). Action 1 is
| "wrong", | keeping | the | agent in State | 1   | (Reward: | −1). |     |     |
| -------- | ------- | --- | -------------- | --- | -------- | ---- | --- | --- |
• Action 1 is the "correct" move, transitioning to State 1 with probability (Reward: +1). Action 2 is
| State    | 2:      |     |                |     |          |      |     | 1−δ |
| -------- | ------- | --- | -------------- | --- | -------- | ---- | --- | --- |
| "wrong", | keeping | the | agent in State | 2   | (Reward: | −1). |     |     |
In a setting (MDP), the Bellman Optimality Equation for State 1 is:
| fully | observable |     |     |     |     |     |     |     |
| ----- | ---------- | --- | --- | --- | --- | --- | --- | --- |
 
 
|     |     | V∗(1)=max |     | −1+γV∗(1), |     | (1−δ)(1+γV∗(2))+δ(−1+γV∗(1)) |     |     |
| --- | --- | --------- | --- | ---------- | --- | ---------------------------- | --- | --- |
(cid:124) (cid:125)
|     |     |     |     |     | (cid:123)(cid:122) | (cid:125) (cid:124) |     | (cid:123)(cid:122) |
| --- | --- | --- | --- | --- | ------------------ | ------------------- | --- | ------------------ |
|     |     |     |     |     | Action1            |                     |     | Action2            |
Symmetrically, for V∗(2), the optimal policy is deterministic: π∗(1)=2 and π∗(2)=1.
IftheagentcannotdistinguishbetweenState1andState2(i.e.,observationsareuninformative),
ThePOMDPCase:
anydeterministicpolicy(e.g.,"alwaystakeaction2")performspoorly. IftheagentisactuallyinState2butthinks
it might be in State 1, it will keep receiving R =−1 indefinitely. In such cases, a stochastic policy (e.g., choosing
actions randomly with p = 0.5) can be superior, as it guarantees the agent will eventually switch states and collect
positive rewards, yielding an expected return higher than the "stuck" deterministic policy.
| 2.4.1 | Observation |     | Model |     |     |     |     |     |
| ----- | ----------- | --- | ----- | --- | --- | --- | --- | --- |
To formalize the POMDP, we introduce the f(y|s,a,s′), which defines the probability of
|     |     |     |     |     | observation |     | function |     |
| --- | --- | --- | --- | --- | ----------- | --- | -------- | --- |
receiving observation y given a transition to s′. The joint probability of transitioning and observing is:
p(s′,y|s,a)=p(s′|s,a)f(y|s,a,s′)
In a simplified model where the observation depends only on the current state s, we use an error parameter ϵ:
(cid:40)
|     |     |     |     |     | s,  | with | probability | 1−ϵ |
| --- | --- | --- | --- | --- | --- | ---- | ----------- | --- |
f(y|s)=
|          |        |             |           |     | noise, | with | probability | ϵ   |
| -------- | ------ | ----------- | --------- | --- | ------ | ---- | ----------- | --- |
| Analysis | of the | uncertainty | parameter |     | ϵ:     |      |             |     |
• ϵ=0: Observability. The agent knows the state perfectly; the POMDP reduces to an MDP.
Full
• (in a 2-state system): Uncertainty. The observation is pure noise. The agent has no
| ϵ = 0.5     |         |        |           | Maximum     |             |               |     |         |
| ----------- | ------- | ------ | --------- | ----------- | ----------- | ------------- | --- | ------- |
| information | about   | its    | location, | making      | the problem | significantly |     | harder. |
| 2.4.2       | Perfect | Memory | and       | Information |             | State         |     |         |
In a general POMDP, the agent cannot observe the true state s directly, and therefore the current observation
t
is usually not sufficient for optimal decision-making. To act well, the agent must retain information from the
y t
past. A natural idea is to let the agent use a perfect memory, that is, a memory containing the entire history
| of actions | taken | and observations |     | received. |     |     |     |     |
| ---------- | ----- | ---------------- | --- | --------- | --- | --- | --- | --- |
We denote the memory state at time t by m . Initially, the memory is empty:
t
|     |     |     |     |     |     | m   | =∅. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
0
After that, the memory is updated by concatenating the current action and observation:
|     |     |     |     |     | m t =a | 0 y 1 a 1 | y 2 ···a | t−1 y t . |
| --- | --- | --- | --- | --- | ------ | --------- | -------- | --------- |
11

In this setting, the memory contains the whole interaction history. The update rule is therefore deterministic and
| simply records | what | happened: |     |      |              |     |        |     |     |     |
| -------------- | ---- | --------- | --- | ---- | ------------ | --- | ------ | --- | --- | --- |
|                |      |           |     | g(m′ | |m,a,y)=1(m′ |     | =may). |     |     |     |
If we consider the pair formed by the hidden state and the memory, namely ), then the process becomes
|     |     |     |     |     |     |     |     |     | (s  | t ,m t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
Markovian in the enlarged space S×M. However, the policy still cannot depend directly on the hidden state s ,
t
| because | this state is | not observable. |     | It can only | depend | on  | the memory: |     |     |     |
| ------- | ------------- | --------------- | --- | ----------- | ------ | --- | ----------- | --- | --- | --- |
π(a|m).
| The distribution | over | hidden | states | conditioned | on the | memory, |     |     |     |     |
| ---------------- | ---- | ------ | ------ | ----------- | ------ | ------- | --- | --- | --- | --- |
µ (s|m),
t
summarizeswhattheagentknowsabouttheenvironmentafterobservingthehistorystoredinm. Thisconditional
distribution depends only on the initial distribution and on the sequence of actions and observations contained
in the memory. For this reason, the memory can be interpreted as an information state: it contains all the
information needed to update the agent’s knowledge about the hidden state.
This shows that the agent’s information, rather than the hidden physical state itself, is what matters for decision-
making. Inpractice, however,workingdirectlywiththefullhistoryisinconvenient,whichmotivatestheintroduc-
| tion of belief | states.       |     |         |            |     |     |     |     |     |     |
| -------------- | ------------- | --- | ------- | ---------- | --- | --- | --- | --- | --- | --- |
| 2.4.3          | Belief States |     | and the | Belief-MDP |     |     |     |     |     |     |
Although the full history is sufficient, it is usually inconvenient to work with it directly because its dimension
grows with time. A more compact representation is obtained by considering the state.
belief
| A belief | state is a probability |     | distribution | over | the hidden | states: |     |     |     |     |
| -------- | ---------------------- | --- | ------------ | ---- | ---------- | ------- | --- | --- | --- | --- |
|          |                        |     |              |      | b (s)=P(S  | =s|H    |     | ),  |     |     |
|          |                        |     |              |      | t          | t       | t   |     |     |     |
where H denotes the history of past actions and observations available to the agent up to time t.
t
Thus, instead of storing the entire history explicitly, the agent stores only its current posterior distribution over
states. The belief state is therefore a sufficient statistic for the history: once is known, the full history is no
|               |                      |        |             |         |      |     |     |     | b   | t   |
| ------------- | -------------------- | ------ | ----------- | ------- | ---- | --- | --- | --- | --- | --- |
| longer needed | for decision-making. |        |             |         |      |     |     |     |     |     |
| The set of    | all beliefs          | is the | probability | simplex | over | S:  |     |     |     |     |
b∈∆|S|.
When the agent takes action a and then receives observation y, the belief is updated by Bayes’ rule. The new
| belief b′ | is given by |     |     |     |     |     |     |     |     |     |
| --------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:80)
|     |     |     |     |                   |     |     | p(s′,y | |s,a)b(s) |     |     |
| --- | --- | --- | --- | ----------------- | --- | --- | ------ | --------- | --- | --- |
|     |     |     |     | b′(s′)=Ua,yb(s′)= |     | s   |        |           | .   |     |
(cid:80)
p(y |s,a)b(s)
s
| This update | has a simple | interpretation: |     |     |     |     |     |     |     |     |
| ----------- | ------------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
• the numerator predicts the probability of reaching state s′ and observing y after taking action a;
• the denominator normalizes the result and is equal to the probability of observing under the current belief
y
and action.
More explicitly, the probability of receiving observation under belief and action is
|     |     |     |     |     |     | y   |     | b   |     | a   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|     |     |     |     | P(y | |a,b)= | p(y | |s,a)b(s). |     |     |     |
| --- | --- | --- | --- | --- | ------ | --- | ---------- | --- | --- | --- |
s
In this partially observable setting, we write the reward as a function of the observation, consistently with the
| sensorimotor | interface | introduced | earlier. |     |     |     |     |     |     |     |
| ------------ | --------- | ---------- | -------- | --- | --- | --- | --- | --- | --- | --- |
The expected immediate reward in belief state b when choosing action a is the average reward over the hidden
| states weighted | by the | current | belief: |     |     |     |     |     |     |     |
| --------------- | ------ | ------- | ------- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|     |     |     |     | r(b,a)= |     | r(y)p(y | |s,a)b(s). |     |     |     |
| --- | --- | --- | --- | ------- | --- | ------- | ---------- | --- | --- | --- |
s,y
This allows us to reformulate the POMDP as a standard MDP whose states are beliefs. This is called the belief-
| MDP. Its | components | are: |     |     |     |     |     |     |     |     |
| -------- | ---------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
12

| •   | beliefs | b∈∆|S|; |     |     |     |     |     |     |
| --- | ------- | ------- | --- | --- | --- | --- | --- | --- |
states:
| • actions: | the same | actions |     | as in the | original | POMDP; |     |     |
| ---------- | -------- | ------- | --- | --------- | -------- | ------ | --- | --- |
• transition dynamics: belief updates induced by actions and observations;
| • rewards: | expected  | rewards |         | r(b,a); |          |         |            |     |
| ---------- | --------- | ------- | ------- | ------- | -------- | ------- | ---------- | --- |
| •          | a mapping | from    | beliefs | to      | actions, | written | as π(a|b). |     |
policy:
The key advantage of this formulation is that the problem becomes fully observable again: the agent may not
| know the | true hidden | state,   | but | it does | know   | its current | belief exactly. |     |
| -------- | ----------- | -------- | --- | ------- | ------ | ----------- | --------------- | --- |
| 2.4.4    | Bellman     | Equation |     | in      | Belief | Space       |                 |     |
Once the POMDP has been rewritten as a belief-MDP, dynamic programming ideas can be applied exactly as in
the fully observable case. The only difference is that the state variable is now the belief b rather than the physical
state s.
Letv∗(b)denotetheoptimalvaluefunctioninbeliefspace. Itrepresentsthemaximumexpectedreturnobtainable
when the current belief about the hidden state is b. The Bellman optimality equation becomes
|     |     |     |     |     |     | (cid:34) |     | (cid:35) |
| --- | --- | --- | --- | --- | --- | -------- | --- | -------- |
(cid:88)
|     |     |     |     | v∗(b)=max |     | r(b,a)+γ | P(y |a,b)v∗(Ua,yb) | .   |
| --- | --- | --- | --- | --------- | --- | -------- | ------------------ | --- |
a
y
This equation has the same structure as the Bellman equation for ordinary MDPs:
| • choose   | an action    | a;           |        |             |         |             |     |     |
| ---------- | ------------ | ------------ | ------ | ----------- | ------- | ----------- | --- | --- |
| • collect  | the expected | immediate    |        | reward      | r(b,a); |             |     |     |
| • consider | all possible | observations |        |             | y that  | may follow; |     |     |
| • update   | the belief   | through      | Bayes’ |             | rule;   |             |     |     |
| • add the  | discounted   | value        | of     | the updated |         | belief.     |     |     |
The corresponding Bellman operator B acts on functions of the belief state as
|     |     |     |     |     |     | (cid:34) |     | (cid:35) |
| --- | --- | --- | --- | --- | --- | -------- | --- | -------- |
(cid:88)
|a,b)f(Ua,yb)
|     |     |     |     | (Bf)(b)=max |     | r(b,a)+γ | P(y | .   |
| --- | --- | --- | --- | ----------- | --- | -------- | --- | --- |
a
y
The main difficulty is that the belief space is continuous, even when the original state space is finite. For this
reason, exact dynamic programming in belief space is often computationally hard, and practical methods usually
relyonapproximations. Nevertheless,thebelief-MDPformulationprovidesthefundamentaltheoreticalframework
| for Bayesian | decision-making |     |         | under | partial | observability. |     |     |
| ------------ | --------------- | --- | ------- | ----- | ------- | -------------- | --- | --- |
| 2.5          | Goals           | and | Rewards |       |         |                |     |     |
At each time step, the reward is a simple number R. Informally, the agent’s goal is to maximize the total
R t ∈
amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long
| run. We | can clearly | state | this | informal | idea | as the reward | hypothesis: |     |
| ------- | ----------- | ----- | ---- | -------- | ---- | ------------- | ----------- | --- |
That all of what we mean by goals and purposes can be well thought of as the maximization of the
expected value of the cumulative sum of a received scalar signal (called reward).
In particular, the reward signal is not the place to impart to the agent prior knowledge about to achieve
how
what we want it to do. For example, a chess-playing agent should be rewarded only for actually winning, not
for achieving subgoals such as taking its opponent’s pieces or gaining control of the center of the board. The
reward signal is your way to communicating to the agent what you want to be achieved, not how you want it to
be achieved.
| 2.6 | Returns | and | Episodes |     |     |     |     |     |
| --- | ------- | --- | -------- | --- | --- | --- | --- | --- |
If the sequence of rewards received after time step t is denoted R ,R ,R ,..., then what precise aspect of
|     |     |     |     |     |     |     | t+1 t+2 | t+3 |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- |
this sequence do we wish to maximize? In general, we seek to maximize the return, where the return,
expected
denoted as G is defined as some specific function of the reward sequence (most of the cases the sum of the
t+1
rewards).
Thisapproachmakessenseinapplicationsinwhichthereisanaturalnotionoffinaltimestep,i.e.,whentheagent-
environment interaction breaks naturally into sequencces called episodes. Each one ends in a special state called
13

the terminal state, followed by a reset to a standard starting state or to a sample from a standard distribution
of starting states. The next episode begins independently of how the previous one ended. Tasks with episodes of
this kind are called episodic tasks and in them we need to distinguish the set of nonterminal states (S), from the
set of all states plus the terminal state (S+).
On the other hand, in many cases the agent-environment interaction does not break naturally into identifiable
episodes, but goes on continually without limit. These are the continuing tasks. The return formulation is
problematic for these because the final time step would be T =∞, such as the the return.
The additional concept that we need is that of discounting. According to this approach, the agent tries to select
actionssothatthesumofthediscountedrewardsitreceivesoverthefutureismaximized. Inparticular,itchooses
A to maximize the expected discounted return:
t
∞
(cid:88)
G =R +γR +γ2R +···= R ,
t t+1 t+2 t+3 t+k+1
k=0
where γ is a parameter defined in [0,1] called the discount rate.
It determines the present value of future rewards: a reward received k time steps in the future is worth only γk−1
times what it would be worth if it were received immediately.
Returnsatsuccessivetimestepsarerelatedtoeachotherinawaythatisimportantforthetheoryandalgorithms
of RL:
G =R +γR +γ2R +γ3R +...
t t+1 t+2 t+3 t+4
=R +γG
t+1 t+1
where G =R +γR +γ2R +... and if the reward is a constant +1, the return is finite
t+1 t+2 t+3 t+4
∞
(cid:88) 1
G = γk =
t 1−γ
k=0
2.7 Unified Notation for Episodic and Continuing Tasks
We number the time steps of each episode starting anew from zero, referring thus not to S but to S at time t
t t,i
of episode i. The same for the other parameters. However, it turns out that when we discuss episodic tasks we
almost never have to distinguish between different episodes. In practice we almost always abuse notation slightly
by dropping the explicit reference to episode number, i.e., we write S to refer to S and so on.
t t,i
Thesumsforbothepisodicandcontinuingtaskscanbeunifiedconsideringepisodeterminationtobetheentering
of a special absorbing state that transitions only to itself and that generates only rewards of zero. Consider the
following transition diagram:
Figure 2.2: Transition diagram for an absorbing state
We can alternatively write
T
(cid:88)
G = γk−t−1R ,
t k
k=t+1
including the possibility that T =∞ or γ =1, but not both.
2.8 Policies and Value Functions
Almost all RL algorithms involve estimating value functions, i.e., functions of states that estimate how good it
is for the agent to be in a given state. The notion of "how good" here is deifned in terms of future rewards
that can be expected or in terms of expected return. They are defined with respect to particular ways of acting,
called policies. A policy is a mapping from states to probabilities of selecting each possible action. If the agent is
following policy π at time t, then π(a|s) is the probability that A =a if S =s.
t t
14

The of a state under a policy π, denoted (s), is the expected return when starting in and
| value     | function      |     |           | s   |        |        |            | v   | π   |     |     |     | s   |
| --------- | ------------- | --- | --------- | --- | ------ | ------ | ---------- | --- | --- | --- | --- | --- | --- |
| following | π thereafter. |     | For MDPs, |     | we can | define | v formally |     | by  |     |     |     |     |
π
|     |     |     |     |     |     |     | (cid:34) |     |     | (cid:35) |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | -------- | --- | --- | --- |
∞
|     |     |     | v(t)(s)=E |     |           | =s]=E |     | (cid:88) γkR |               | for | all  |     |     |
| --- | --- | --- | --------- | --- | --------- | ----- | --- | ------------ | ------------- | --- | ---- | --- | --- |
|     |     |     |           | π   | [G t |S t |       | π   |              | t+k+1 |S t =s | ,   | s∈S, |     |     |
π
k=0
where E denotes the expected value of a random variable given that the agent follows policy and is any
|     | [.] |     |     |     |     |     |     |     |     |     |     | π   | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
π
time step.
| We call | the function | v   | the state-value |     | function |     | for policy | π.  |     |     |     |     |     |
| ------- | ------------ | --- | --------------- | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
π
Similarly, we define the value of taking action a in state s under a policy π, denoted q (s,a), as the expected
π
return starting from s, taking the action a, and thereafter following policy π:
|     |     |     |     |     |     |     |     | (cid:34) | ∞   |     |     | (cid:35) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | -------- | --- |
(cid:88)
|     |     | q   | (s,a)=E |     | [G |S | =s,A | =a]=E |     | γkR   | |S =s,A | =a  | .   |     |
| --- | --- | --- | ------- | --- | ----- | ---- | ----- | --- | ----- | ------- | --- | --- | --- |
|     |     | π   |         | π   | t t   |      | t     | π   | t+k+1 | t       | t   |     |     |
k=0
| We call | q the action-value |     | function |     | for policy | π.  |     |     |     |     |     |     |     |
| ------- | ------------------ | --- | -------- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
π
The value functions v and q can be estimated from experience by averaging the returns G observed after each
|     |     | π   |     | π   |     |     |     |     |     |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
time step when the agent is following policy π. This leads to the so called methods.
|     | t   |     |     |     |     |     |     |     |     | Monte | Carlo |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ----- | --- | --- |
Observation:
AfundamentalpropertyofvaluefunctionsusedthroughoutRLanddynamicprogrammingisthattheysatisfyrecursive
relationships similar to that which we have already estabilished for the return. For any policy π and any state s, the
following consistency condition holds between the value of s and the value of its possible successor state:
|     |     |     | v   | (s) =E | [G              | |S =s]   |                  |     |        |         |      |     |     |
| --- | --- | --- | --- | ------ | --------------- | -------- | ---------------- | --- | ------ | ------- | ---- | --- | --- |
|     |     |     | π   |        | π               | t t      |                  |     |        |         |      |     |     |
|     |     |     |     | =E     | [R              | +γG      | |S               | =s] |        |         |      |     |     |
|     |     |     |     |        | π               | t+1      | t+1              | t   |        |         |      |     |     |
|     |     |     |     | =      | (cid:80) π(a|s) | (cid:80) | p(s′,r|s,a)[r+γv |     | (s′)], | for all | s∈S, |     |     |
|     |     |     |     |        | a               |          | s′,r             |     | π      |         |      |     |     |
Note how in the last equation we have merged the two sums, one over all the values of s′ and the other over all
the values of r, into one sum over all possible values of both. For each triple (a,s′,r) we compute its probability
π(a|s)p(s′,r|s,a), weight the quantity in brackets by that probability and then sum over all possibilities to get an
| expected | value. |     |     |     |     |     |     |     |     |     |     |     |     |
| -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
NotethatthelastequationintheobservationisthesocalledBellmanequationforvπ. Itexpressesarelationship
between the value of a state and the values of its successor states, averaging over all the possibilities, weighting
| each by | its probability |          | of occurring. |     |        |       |           |         |     |     |     |     |     |
| ------- | --------------- | -------- | ------------- | --- | ------ | ----- | --------- | ------- | --- | --- | --- | --- | --- |
|         |                 |          |               |     |        |       | Backup    | diagram | for |     |     |     |     |
|         |                 |          |               |     | Figure | 2.3:  |           |         | v   | π   |     |     |     |
| 2.9     | Optimal         | Policies |               |     | and    | Value | Functions |         |     |     |     |     |     |
Solving a RL task means, roughly, finding a policy that achieves a lot of reward over the long run. A policy π
is defined to be better than or equal to a policy π′ if its expected return is greater than or equal to that of π′
for all states. There is always at least one policy that is better than or equal to all other policies, and that is
the optimal policy π . The optimal policies share the same state-value function, called the optimal state-value
∗
| function, | denoted | and | defined | as  |     |     |     |     |     |     |     |     |     |
| --------- | ------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
v ∗
|     |     |     |     |     | v   | (s)=maxv(t)(s), |     |     | for all s∈S. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------- | --- | --- | ------------ | --- | --- | --- | --- |
|     |     |     |     |     | ∗   |                 |     | π   |              |     |     |     |     |
π
Optimal policies also share the same function, denoted and defined as
|     |     |     |     |     | optimal | action-value |     |     |     | q ∗ |     |     |     |
| --- | --- | --- | --- | --- | ------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
15

|     |     |     | q (s,a)=maxq |     | (s,a), | for | all s∈S,a∈A(s). |     |     |     |
| --- | --- | --- | ------------ | --- | ------ | --- | --------------- | --- | --- | --- |
|     |     |     | ∗            |     | π      |     |                 |     |     |     |
π
| Thus, we | can write in | terms | of as       | follows: |     |     |       |      |      |     |
| -------- | ------------ | ----- | ----------- | -------- | --- | --- | ----- | ---- | ---- | --- |
|          | q            |       | v           |          |     |     |       |      |      |     |
|          | ∗            |       | ∗           |          |     |     |       |      |      |     |
|          |              |       | q (s,a)=E[R |          | +γv | (S  | )|S   | =s,A | =a]. |     |
|          |              |       | ∗           |          | t+1 | ∗   | t+1 t |      | t    |     |
Because v is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman
∗
equation for state values. It can be written in a special form without reference to any specific policy. This is the
Bellmanequationforv ,ortheBellman optimality equation,andexpressesthefactthatthevalueofastateunder
∗
an optimal policy must equal the expected return for the best action from that state.
v(t)(s)
|     |     |     | =max |     | q (s,a) |         |     |     |     |     |
| --- | --- | --- | ---- | --- | ------- | ------- | --- | --- | --- | --- |
|     |     |     | ∗    | a   | π∗      |         |     |     |     |     |
|     |     |     | =max |     | E [G    | |S =s,A | =a] |     |     |     |
|     |     |     |      | a   | π∗      | t t     | t   |     |     |     |
E
|             |            |          | =max | a   | π [R                | t+1 +γG  | t+1 |S | t =s,A      | t =a] |           |
| ----------- | ---------- | -------- | ---- | --- | ------------------- | -------- | ------ | ----------- | ----- | --------- |
|             |            |          |      |     | (cid:104)∗          |          |        |             |       | (cid:105) |
|             |            |          |      |     | E                   | +γv(t)(S |        |             |       |           |
|             |            |          | =max | a   | R t+1               | ∗        | t+1    | )|S t =s,A  | t =a  |           |
|             |            |          |      |     | (cid:80)            |          |        | (cid:104)   |       | (cid:105) |
|             |            |          | =max |     | p(s′,r(s,a,s′)|s,a) |          |        | r+γv(t)(s′) |       |           |
|             |            |          |      | a   | s′,r                |          |        |             | ∗     |           |
| The Bellman | optimality | equation | for  | is  |                     |          |        |             |       |           |
q
∗
|     |     |     | =E[R  |     |       |      |          | ,a′)|S |      |       |
| --- | --- | --- | ----- | --- | ----- | ---- | -------- | ------ | ---- | ----- |
|     |     | q ∗ | (s,a) | t+1 | +γmax | a′ q | ∗ (S t+1 | t      | =s,A | t =a] |
(cid:80)
|     |     |     | =   |      | p(s′,r|s,a)[r+γmax |     |     | q (s′,a′)]. |     |     |
| --- | --- | --- | --- | ---- | ------------------ | --- | --- | ----------- | --- | --- |
|     |     |     |     | s′,r |                    |     |     | a′ ∗        |     |     |
For finite MDPs, the Bellman equations for v and q have unique solutions. The optimal policies can be derived
|            |                  |     |        |      | ∗      | ∗       |     |     |     |     |
| ---------- | ---------------- | --- | ------ | ---- | ------ | ------- | --- | --- | --- | --- |
| from these | value functions. |     |        |      |        |         |     |     |     |     |
|            |                  |     |        |      | Backup | diagram | for | and |     |     |
|            |                  |     | Figure | 2.4: |        |         |     | v ∗ | q ∗ |     |
Once one has v , it is relatively easy to determine an optimal policy. For each state s, there will be one or more
∗
actions at which the maximum is obtained in the Bellman optimality equation. Any policy that assigns nonzero
probabilityonlytotheseactionsisanoptimalpolicy. Havingq makeschoosingoptimalactionseveneasier. With
∗
, the agent does not even have to do a one-step-ahead search: for any state s, it can simply find any action that
q ∗
| maximizes | q (s,a), and | that will | be an | optimal | action. |     |     |     |     |     |
| --------- | ------------ | --------- | ----- | ------- | ------- | --- | --- | --- | --- | --- |
∗
Explicitly solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to
solving the RL problem. However, this solution is rarely direcly useful. This solution relies on at least three
| assumptions     | that are rarely    | true | in practice: |             |       |                 |     |     |     |     |
| --------------- | ------------------ | ---- | ------------ | ----------- | ----- | --------------- | --- | --- | --- | --- |
| • the dynamics  | of the environment |      | are          | accurately  | known |                 |     |     |     |     |
| • computational | resources          | are  | sufficient   | to complete |       | the calculation |     |     |     |     |
| • the states    | have the Markov    |      | property     |             |       |                 |     |     |     |     |
16

| Chapter |     | 3   |             |     |     |     |     |     |     |     |
| ------- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic |     |     | Programming |     |     |     |     |     |     |     |
Thetermdynamic programming (DP)referstoacollectionofalgorithmsthatcanbeusedtocomputeoptimal
policies when a perfect model of the environment is available and the problem is formulated as a Markov Decision
Process (MDP). Classical DP algorithms are of limited practical use in reinforcement learning, both because they
assumecompleteknowledgeoftheenvironmentandbecausetheycanbecomputationallyexpensive. Nevertheless,
| they remain | of great | theoretical |     | importance. |     |     |     |     |     |     |
| ----------- | -------- | ----------- | --- | ----------- | --- | --- | --- | --- | --- | --- |
In this chapter, we usually assume that the environment is a finite MDP. This means that the sets of states,
actions, andrewardsarefinite,andthatthedynamicsoftheenvironmentaredescribedbytransitionprobabilities
|     |     |     |     |     |     | p(s′,r | |s,a), |     |     |     |
| --- | --- | --- | --- | --- | --- | ------ | ------ | --- | --- | --- |
for all s∈S, a∈A, ∈R, and s′ ∈S. When the state or action spaces are continuous, a common way to obtain
r
approximate solutions is to discretize them and then apply finite-state DP methods. The central idea of DP is to
| use value | functions | to  | organize | and | guide | the search | for good | policies. |     |     |
| --------- | --------- | --- | -------- | --- | ----- | ---------- | -------- | --------- | --- | --- |
Once the optimal value functions v or q are known, it is straightforward to derive an optimal policy. These
|         |                 |     |         | ∗       | ∗   |            |            |     |     |     |
| ------- | --------------- | --- | ------- | ------- | --- | ---------- | ---------- | --- | --- | --- |
| optimal | value functions |     | satisfy | the     |     |            | equations: |     |     |     |
|         |                 |     |         | Bellman |     | optimality |            |     |     |     |
(cid:88)
|     |     |     |     | v   | (s)=max |     | p(s′,r |s,a)[r+γv |     | (s′)], |     |
| --- | --- | --- | --- | --- | ------- | --- | ----------------- | --- | ------ | --- |
|     |     |     |     | ∗   |         |     |                   |     | ∗      |     |
a
s′,r
|     |     |     |     |          | (cid:88) |        | (cid:104) |         |         | (cid:105) |
| --- | --- | --- | --- | -------- | -------- | ------ | --------- | ------- | ------- | --------- |
|     |     |     |     | q (s,a)= |          | p(s′,r | |s,a)     | r+γmaxq | (s′,a′) | .         |
|     |     |     |     | ∗        |          |        |           |         | ∗       |           |
a′
s′,r
| Equivalently, | the | action-value |     | function | can also | be  | written as |     |     |     |
| ------------- | --- | ------------ | --- | -------- | -------- | --- | ---------- | --- | --- | --- |
(cid:88)
|     |     |     |     |     | q (s,a)= |     | p(s′,r |s,a)[r+γv |     | (s′)]. |     |
| --- | --- | --- | --- | --- | -------- | --- | ----------------- | --- | ------ | --- |
|     |     |     |     |     | ∗        |     |                   |     | ∗      |     |
s′,r
As we shall see, DP algorithms are obtained by turning Bellman equations such as these into assignments, that is,
into update rules that progressively improve approximations of the desired value functions.
| 3.1 | Policy | Evaluation |     |     |     |     |     |     |     |     |
| --- | ------ | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
We begin with the problem of computing the value of a policy π. Given a policy, we want to know how good it is
| to follow | it from | each state. | This | task | is called |     | evaluation. |     |     |     |
| --------- | ------- | ----------- | ---- | ---- | --------- | --- | ----------- | --- | --- | --- |
policy
| The quantity | we  | want to | compute | is  | the         |     |          |     |     |     |
| ------------ | --- | ------- | ------- | --- | ----------- | --- | -------- | --- | --- | --- |
|              |     |         |         |     | state-value |     | function |     |     |     |
v (s),
π
which represents the expected return obtained by starting from state and then following policy π.
s
| Using the | definition | of return, |     | we can | write   |       |        |        |      |     |
| --------- | ---------- | ---------- | --- | ------ | ------- | ----- | ------ | ------ | ---- | --- |
|           |            |            |     |        | v (s)=E | [G    | |S =s] |        |      |     |
|           |            |            |     |        | π       | π t   | t      |        |      |     |
|           |            |            |     |        | =E      | [R    | +γG    | |S     | =s]  |     |
|           |            |            |     |        |         | π t+1 | t+1    | t      |      |     |
|           |            |            |     |        | =E      | [R    | +γv    | (S )|S | =s]. |     |
|           |            |            |     |        |         | π t+1 | π      | t+1    | t    |     |
17

Ifweexpandtheexpectationoverallpossibleactions,nextstates,andrewards,weobtaintheBellmanexpectation
equation:
v (s)= (cid:88) π(a|s) (cid:88) p(s′,r |s,a) (cid:2) r+γv (s′) (cid:3) .
π π
a s′,r
Here:
• π(a|s) is the probability of choosing action a in state s under policy π;
• p(s′,r |s,a) is the probability of moving to state s′ and receiving reward r after taking action a in state s.
Ifthedynamicsoftheenvironmentareknown,thentheseequationsformasystemof|S|linearequationsinthe|S|
unknown values v (s). In principle, this system can be solved exactly. In practice, however, it is often convenient
π
to use an iterative method.
For this reason, we choose an initial approximation v arbitrarily (with terminal states fixed to 0), and then
0
repeatedly update it according to
v (s)=E[R +γv (S )|S =s]
k+1 t+1 k t+1 t
= (cid:88) π(a|s) (cid:88) p(s′,r |s,a) (cid:2) r+γv (s′) (cid:3) .
k
a s′,r
This procedure is called iterative policy evaluation. At each step, the old estimate of the value of state s is
replaced by a new estimate obtained from:
• the immediate rewards that can be received from s;
• the values of the successor states;
• the probabilities of the transitions and actions under the policy.
This is called an expected update, because it uses an expectation over all possible next states and rewards, rather
than a single sampled transition.
Under the usual assumptions, the sequence {v } converges to the true value function v as k →∞.
k π
A complete in-place version of iterative policy evaluation is shown in the pseudocode below.
Figure 3.1: Iterative Policy Evaluation Algorithm
3.2 Policy Improvement
Evaluating a policy is useful because it helps us find a better one. Once we know v , we can ask whether, in some
π
state s, it would be better to choose a different action than the one currently suggested by π.
To answer this, we introduce the state-action value function
q (s,a),
π
which measures the expected return obtained by taking action a in state s first, and then following policy π
afterwards.
Formally,
q (s,a)=E[R +γv (S )|S =s,A =a]
π t+1 π t+1 t t
= (cid:88) p(s′,r |s,a) (cid:2) r+γv (s′) (cid:3) .
π
s′,r
18

This quantity tells us how good action looks if we evaluate it using the current value function .
a v π
Now suppose we define another policy π′. A fundamental result, called the theorem,
policy improvement
| states that | if for | every | state | s   |     |     |     |     |     |     |
| ----------- | ------ | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
(cid:88) π′(a|s)q
|     |     |     |     |     |     |     |     | π (s,a)≥v | π (s), |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | --- |
a
| then the | new policy |     | is at least | as good | as  | the old | one: |     |       |     |
| -------- | ---------- | --- | ----------- | ------- | --- | ------- | ---- | --- | ----- | --- |
|          |            |     |             |         |     | v (s)≥v | (s), |     | ∀s∈S. |     |
|          |            |     |             |         |     | π′      | π    |     |       |     |
So, ifanewpolicychoosesactionswhoseexpectedvalueunderq isneverworsethantheoldpolicy, thenthenew
π
| policy improves |     | on the | old | one. |     |     |     |     |     |     |
| --------------- | --- | ------ | --- | ---- | --- | --- | --- | --- | --- | --- |
The most natural choice is to build a new policy that, in each state, selects the action with the highest q (s,a).
π
| This gives | the |     | policy: |     |     |     |     |     |     |     |
| ---------- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
greedy
|     |     |     |     | π′(s)=argmaxq |     |     | (s,a) |     |     |     |
| --- | --- | --- | --- | ------------- | --- | --- | ----- | --- | --- | --- |
π
a
(cid:88)
|     |     |     |     |     | =argmax |     | p(s′,r | |s,a) | (cid:2) r+γv | (s′) (cid:3) . |
| --- | --- | --- | --- | --- | ------- | --- | ------ | ----- | ------------ | -------------- |
π
a
s′,r
| In deterministic |     | form, | this | can be | written   | as  |           |     |          |     |
| ---------------- | --- | ----- | ---- | ------ | --------- | --- | --------- | --- | -------- | --- |
|                  |     |       |      |        |           |     | (cid:16)  |     | (cid:17) |     |
|                  |     |       |      |        | π′(a|s)=1 |     | a=argmaxq |     | (s,a)    | .   |
π
a
The greedy policy satisfies the condition of the policy improvement theorem because
(cid:88)
|     |     |     |     |     | π′(a|s)q |     | (s,a)=maxq |     | (s,a)≥v | (s). |
| --- | --- | --- | --- | --- | -------- | --- | ---------- | --- | ------- | ---- |
|     |     |     |     |     |          |     | π          |     | π π     |      |
a
a
Therefore, the greedy policy is always at least as good as the original one. Moreover, if the greedy policy is
different from the old policy, then the improvement is strict. This means that changing the policy in this way
| really produces |     | a better | policy, | not | just an | equivalent | one. |     |     |     |
| --------------- | --- | -------- | ------- | --- | ------- | ---------- | ---- | --- | --- | --- |
This step is called improvement: we use the value function of the current policy to construct a better
policy
policy.
| 3.3              | Policy | Iteration   |         |              |              |            |              |           |     |     |
| ---------------- | ------ | ----------- | ------- | ------------ | ------------ | ---------- | ------------ | --------- | --- | --- |
| Policy iteration |        | is obtained |         | by combining |              | the two    | previous     | ideas:    |     |     |
| 1. evaluate      | the    | current     | policy; |              |              |            |              |           |     |     |
| 2. improve       | it by  | making      | it      | greedy       | with respect |            | to its value | function. |     |     |
| Starting         | from   | an initial  | policy  | π ,          | we obtain    | a sequence |              |           |     |     |
0
|         |        |                |     |     |          | E   | I            | E   | I E       |     |
| ------- | ------ | -------------- | --- | --- | -------- | --- | ------------ | --- | --------- | --- |
|         |        |                |     |     | π        | −→v | −→π          | −→v | −→π −→··· |     |
|         |        |                |     |     | 0        | π0  | 1            | π1  | 2         |     |
| where E | stands | for evaluation |     | and | I stands | for | improvement. |     |           |     |
In other words:
| • first we    | compute     | the    | value      | of the | current     | policy;    |     |         |     |     |
| ------------- | ----------- | ------ | ---------- | ------ | ----------- | ---------- | --- | ------- | --- | --- |
| • then        | we replace  | the    | policy     | with   | a greedy    | one;       |     |         |     |     |
| • then        | we evaluate | the    | new        | policy | again;      |            |     |         |     |     |
| • and we      | continue    | until  | the        | policy | no longer   | changes.   |     |         |     |     |
| The algorithm |             | can be | summarized |        | as follows: |            |     |         |     |     |
|               |             |        |            |        |             | initialize | π   | , k ←0, |     |     |
0
repeat
|     |     |     |     |     |     |     | v =Value(π  |     | )   |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --- | --- |
|     |     |     |     |     |     |     | k           |     | k   |     |
|     |     |     |     |     |     |     | π =Greedy(v |     | )   |     |
|     |     |     |     |     |     |     | k+1         |     | k   |     |
k ←k+1
until
|     |     |     |     |     |     |     | π   | =π . |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- |
|     |     |     |     |     |     |     | k+1 | k    |     |     |
19

A key point is that the stopping criterion is based on the policy, not on a numerical tolerance. The algorithm
stops when the improvement step no longer changes the policy.
This is enough to guarantee convergence in a finite MDP. Indeed, the number of deterministic policies is finite
(although possibly very large), and each genuine improvement step produces a strictly better policy. Therefore,
the algorithm cannot continue forever: after a finite number of iterations, it must reach a policy that cannot be
improved further.
At convergence, suppose that for some step n we have
π =π .
n+1 n
Then the corresponding value function v satisfies
n
v (s)=max (cid:88) p(s′|s,a) (cid:0) r(s,a,s′)+γv (s′) (cid:1) ,
n n
a
s′
which is exactly the Bellman optimality equation.
Since this equation has a unique solution, it follows that
v =v and π =π .
n ∗ n ∗
So, the policy obtained when policy iteration stops is an optimal policy.
Figure 3.2: Policy Iteration Algorithm
3.4 Comparison with Value Iteration
Let B be the Bellman optimality operator, and assume that both algorithms start from the same initial guess V .
0
Then, after the same number of steps k, the value produced by policy iteration is better than the one produced
by value iteration:
V >BkV .
k 0
The reason is that policy iteration performs a full policy improvement at each step. Before convergence, one has
V >BV ,
k k−1
and by repeatedly using the monotonicity of the Bellman operator,
U ≤W ⇒ BU ≤BW,
we obtain
V >BV >B2V >···>BkV .
k k−1 k−2 0
20

This means that, in terms of number of iterations, policy iteration improves the value estimate at least as fast
as value iteration. However, each step of policy iteration is more expensive, because the evaluation step requires
| solving a | linear     | system or | running     | iterative   | evaluation. |                  | So the | trade-off | is: |
| --------- | ---------- | --------- | ----------- | ----------- | ----------- | ---------------- | ------ | --------- | --- |
| • policy  | iteration: | fewer     | iterations, | but         | heavier     | steps;           |        |           |     |
| •         | iteration: | lighter   | steps,      | but usually |             | more iterations. |        |           |     |
value
Therefore, policy iteration is often faster in terms of convergence in policy space, even if each individual iteration
costs more.
| 3.5 Value |     | Iteration |     |     |     |     |     |     |     |
| --------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- |
One drawback of policy iteration is that each of its iterations requires a policy evaluation step, which may itself
be a long iterative computation involving several sweeps through the state space. A natural simplification is to
truncate the policy evaluation step. An important special case is obtained when policy evaluation is stopped after
| just one | sweep. | This gives | the algorithm |     | called |     | iteration. |     |     |
| -------- | ------ | ---------- | ------------- | --- | ------ | --- | ---------- | --- | --- |
value
Value iteration can be interpreted as combining, in a single update, one step of policy improvement and one step
| of truncated | policy | evaluation. | Its | update     | rule | is  |        |     |          |
| ------------ | ------ | ----------- | --- | ---------- | ---- | --- | ------ | --- | -------- |
|              |        |             | v   | (s)=maxE[R |      |     | +γv (S | )|S | =s,A =a] |
|              |        |             | k+1 |            |      | t+1 | k      | t+1 | t t      |
a
|     |     |     |     |      |     | (cid:88) p(s′,r |            |     | (s′)]. |
| --- | --- | --- | --- | ---- | --- | --------------- | ---------- | --- | ------ |
|     |     |     |     | =max |     |                 | |s,a)[r+γv |     |        |
|     |     |     |     |      | a   |                 |            |     | k      |
s′,r
| It is often | convenient | to define | the |         |     |            |          |     | as  |
| ----------- | ---------- | --------- | --- | ------- | --- | ---------- | -------- | --- | --- |
|             |            |           |     | Bellman |     | optimality | operator |     | B   |
(cid:88)
|     |     |     | (BW)(s)=max |     |     | p(s′ | |s,a)[r(s,a,s′)+γW(s′)]. |     |     |
| --- | --- | --- | ----------- | --- | --- | ---- | ------------------------ | --- | --- |
a
s′
| With this | notation, | value iteration |     | can be | written | compactly |     | as  |     |
| --------- | --------- | --------------- | --- | ------ | ------- | --------- | --- | --- | --- |
|           |           |                 |     |        |         | v         | =Bv | .   |     |
|           |           |                 |     |        |         | k+1       |     | k   |     |
The optimal value function v is the unique fixed point of this operator, that is,
∗
|     |     |     |     |     |     | Bv  | =v  | .   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | ∗ ∗ |     |     |
Therefore, value iteration can be viewed as a sequence of successive approximations converging to .
v
∗
Formally, exact convergence to requires infinitely many iterations. In practice, the algorithm is stopped when
v
∗
two successive value functions differ by less than a prescribed tolerance in the max norm:
|     |     |     |     |     |     | ∥v k+1 | −v k ∥ ∞ | <δ. |     |
| --- | --- | --- | --- | --- | --- | ------ | -------- | --- | --- |
A key property of the Bellman optimality operator is that it is a in the max norm, with contraction
contraction
factor γ:
|     |     |     |     |     | ∥BU | −BW∥ | ≤γ∥U | −W∥ | .   |
| --- | --- | --- | --- | --- | --- | ---- | ---- | --- | --- |
|     |     |     |     |     |     |      | ∞    |     | ∞   |
As a consequence,
|     |     |     | ∥v  | k+1 −v | ∗ ∥ ∞ | =∥Bv | k −Bv ∗ | ∥ ∞ ≤γ∥v | k −v ∗ ∥ ∞ , |
| --- | --- | --- | --- | ------ | ----- | ---- | ------- | -------- | ------------ |
which shows that value iteration converges linearly to the optimal value function.
|     |     |     |     | Figure |     | 3.3: Value | Iteration | Algorithm |     |
| --- | --- | --- | --- | ------ | --- | ---------- | --------- | --------- | --- |
21

Thus, valueiterationcombinespolicyimprovementandtruncatedpolicyevaluationintoasinglerepeatedupdate.
Compared with policy iteration, each iteration is simpler and cheaper, although more iterations may be needed.
On the other hand, faster convergence is often achieved by performing more than one policy evaluation sweep
before each policy improvement step, leading back toward policy iteration or intermediate methods between the
two.
| 3.6 | Asynchronous |     |     | DP  |     |     |     |     |     |
| --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
AmajordrawbacktotheDPmethodsthatwehavediscussedsofaristhattheyinvolveoperationsovertheentire
state set of the MDP, i.e., they require sweeps of the state set. If the state set is very large, then even a single
| sweep can | be prohibitively |     | expensive. |     |     |     |     |     |     |
| --------- | ---------------- | --- | ---------- | --- | --- | --- | --- | --- | --- |
DPalgorithmsarein-placeiterativeDPalgorithmsthatarenotorganizedintermsofsystematic
Asynchronous
sweeps of the state set. These algorithms update the values of states in any order whatsoever, using whatever
values of other states happen to be available. The values of some states may be updated several times before the
values of others are updated once. To converge correctly, however, an asynchronous algorithm must continue to
update the values of all the states: it cannot ignore any state after some point in the computation. Asynchronous
| DP algorithms |     | allow great | flexibility |     | in selecting | states to update. |     |     |     |
| ------------- | --- | ----------- | ----------- | --- | ------------ | ----------------- | --- | --- | --- |
Avoiding sweeps does not necessarily mean that we can get away with less computation. It just means that an
algorithm does not need to get locked into any hopelessly long sweep before it can make progress improving a
policy. We can try to take advantage of this flexibility by selecting the states to which we apply updates so as to
improvethealgorithm’srateofprogress. Wecantrytoordertheupdatestoletvalueinformationpropagatefrom
state to state in an efficient way. Some states may not need their values updated as often as others.
| 3.7 | Generalized |     | Policy |     | Iteration |     |     |     |     |
| --- | ----------- | --- | ------ | --- | --------- | --- | --- | --- | --- |
Policyiterationconsistsoftwosimultaneous, interactingprocesses: onemakingthevaluefunctionconsistentwith
the current policy (policy evaluation), and the other making the policy greedy with respect to the current value
function(policyimprovement). Inpolicyiteration,thesetwoprocessesalternate,eachcompletingbeforetheother
| begins, even | though | this | is not | really | necessary. |     |     |     |     |
| ------------ | ------ | ---- | ------ | ------ | ---------- | --- | --- | --- | --- |
We use the term to refer to the general idea of letting policy-evaluation
|     |     | generalized |     | policy | iteration | (GPI) |     |     |     |
| --- | --- | ----------- | --- | ------ | --------- | ----- | --- | --- | --- |
and policy-improvement processes interact, independent of the granularity and other details of the two processes.
Almost all RL methods are well described as GPI. That is, all have identifiable policies and value functions, with
the policy always being improved with respect to the value function for the policy. If both the evaluation process
and the improvement process stabilize, then the value function and policy must be optimal.
| 3.8     | Linear   | Programming |      |       | for    | MDP |     |     |     |
| ------- | -------- | ----------- | ---- | ----- | ------ | --- | --- | --- | --- |
| Primal: | optimize | a function  | f(u) | where | u∈R|S| |     |     |     |     |
(cid:88)
|     |     |     |     |     |     | f(u)= p | (s)u(s) |     |     |
| --- | --- | --- | --- | --- | --- | ------- | ------- | --- | --- |
0
s
|     |     |     | (cid:40) |     |     |     |     |     | (cid:41) |
| --- | --- | --- | -------- | --- | --- | --- | --- | --- | -------- |
(cid:88)
|     |     |     | u∈R|S| |     | p(s′ | (cid:0) r(sas′)+γu(s′) | (cid:1) |       |      |
| --- | --- | --- | ------ | --- | ---- | ---------------------- | ------- | ----- | ---- |
|     |     | Ω=  |        | :   | |sa) |                        | −u(s)≤0 | ∀s∈S, | ∀a∈A |
s′
is a polyhedron and a convex set. The objective is to minimize the function over the set Ω.
| Ω   |     |     |     |     |     |     |     | f(u) |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- |
Observation:
| If u∈Ω | we call | it a Feasible |     | Point. |     |     |     |     |     |
| ------ | ------- | ------------- | --- | ------ | --- | --- | --- | --- | --- |
The solution of the primal MDP linear programming = optimal solution of the MDP
)=f(V∗)=minf(u)
|         |           |                 |     |     | maxG  | π (p 0 |     |     |     |
| ------- | --------- | --------------- | --- | --- | ----- | ------ | --- | --- | --- |
|         |           |                 |     |     | π     |        | u∈Ω |     |     |
| • V∗ is | feasible: | V∗ ∈Ω⇒f(V∗)≥min |     |     | f(u). |        |     |     |     |
u∈Ω
Since
|     |          |     |         |     |         | (cid:34) |         |     | (cid:35) |
| --- | -------- | --- | ------- | --- | ------- | -------- | ------- | --- | -------- |
|     | (cid:88) |     | (cid:0) |     | (cid:1) | (cid:88) | (cid:0) |     | (cid:1)  |
p(s′ |sa) r(sas′)+γV∗(s′) ≤max p(s′ |sa) r(sas′)+γV∗(s′) =V∗(s) ∀s,a
a
|     | s′  |     |     |     |     | s′  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
22

• If is feasible, then u≥V∗ component-wise (meaning u(s)≥V∗(s)).
| u   |     |      |        |              |     |          | ∀s,  |                         |     |     |          |     |
| --- | --- | ---- | ------ | ------------ | --- | -------- | ---- | ----------------------- | --- | --- | -------- | --- |
|     |     |      |        |              |     | (cid:34) |      |                         |     |     | (cid:35) |     |
|     | if  |      | choose | a∗(s)=argmax |     | (cid:88) | p(s′ | (cid:0) r(sas′)+γV∗(s′) |     |     | (cid:1)  |     |
|     |     | u∈Ω, |        |              |     |          |      | |sa)                    |     |     |          |     |
a
s′
u(s)−V∗(s)
|     |     |     |     |     |     |     |     | (cid:34) |     |     |     | (cid:35) |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | -------- |
(cid:88) (cid:0) s′ |sa∗(s) (cid:1)(cid:0) r(sa∗(s)s′)+γu(s′) (cid:1) (cid:88) p(s′ (cid:0) r(sas′)+γV∗(s′) (cid:1)
|     |     | ≥ p |     |     |     |     | −max |     |     | |sa) |     |     |
| --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | ---- | --- | --- |
a
|     |     | s′  |     |     |     |     |     |     | s′  |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:88) (cid:0) (cid:1)(cid:0) (cid:1) (cid:88) (cid:0) (cid:1)(cid:0) (cid:1)
= p s′ |sa∗(s) r(sa∗(s)s′)+γu(s′) − p s′ |sa∗(s) r(sa∗(s)s′)+γV∗(s′)
|     |     | s′       |                    |                             |     |     |         | s′  |     |     |     |     |
| --- | --- | -------- | ------------------ | --------------------------- | --- | --- | ------- | --- | --- | --- | --- | --- |
|     |     | (cid:88) | (cid:0) s′ |sa∗(s) | (cid:1)(cid:0) u(s′)−V∗(s′) |     |     | (cid:1) |     |     |     |     |     |
|     |     | =γ p     |                    |                             |     |     |         |     |     |     |     |     |
s′
|     |     | (cid:0) | u(s′)−V∗(s′) |     | (cid:1) |     |     |     |     |     |     |     |
| --- | --- | ------- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
≥γmin
s′
| Taking | the minimum | over | s on | both | sides:   |     |                    |     |            |     |     |     |
| ------ | ----------- | ---- | ---- | ---- | -------- | --- | ------------------ | --- | ---------- | --- | --- | --- |
|        |             |      |      |      | (1−γ)min |     | (cid:0) u(s)−V∗(s) |     | (cid:1) ≥0 |     |     |     |
s
| hence | u(s)≥V∗(s) | for | all s, | and therefore |     |     |     |     |     |     |     |     |
| ----- | ---------- | --- | ------ | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
f(u)≥f(V∗)
Primal problem:
minf(u)
u∈Ω
Can we go from this constrained problem to an unconstrained problem? Yes.
|     |     |     |     |     | minf(u)= |     | min    | [f(u)+h(u)] |     |     |     |     |
| --- | --- | --- | --- | --- | -------- | --- | ------ | ----------- | --- | --- | --- | --- |
|     |     |     |     |     | u∈Ω      |     | u∈R|S| |             |     |     |     |     |
with
|     |     |     |     |     |     |     | (cid:40) | if  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     |     | +∞       | u∈/ | Ω   |     |     |     |
h(u)=
if
|     |     |          |     |          |       |          | 0         | u∈Ω                    |     |               |                  |     |
| --- | --- | -------- | --- | -------- | ----- | -------- | --------- | ---------------------- | --- | ------------- | ---------------- | --- |
|     |     |          |     | (cid:34) |       | (cid:32) |           |                        |     |               | (cid:33)(cid:35) |     |
|     |     |          |     | (cid:88) |       | (cid:88) |           |                        |     |               |                  |     |
|     |     | h(u)=max |     |          | z(sa) |          | p(s′ |sa) | (cid:0) r(sas′)+γu(s′) |     | (cid:1) −u(s) |                  |     |
z≥0
|     |     |     |     |     | sa  | s′  |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thus:
minf(u)=minmaxL(u,z)
|     |     |     |     |     | u∈Ω |     | u   | z≥0 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:32) (cid:33)
|     |         | (cid:88) |           |              |     | (cid:88) |       |        | (cid:88) |            |     | (cid:88) |
| --- | ------- | -------- | --------- | ------------ | --- | -------- | ----- | ------ | -------- | ---------- | --- | -------- |
|     | L(u,z)= |          | z(sa)p(s′ | |sa)r(sas′)+ |     |          | u(s′) | (s′)+γ | p(s′     |            |     | z(s′a′)  |
|     |         |          |           |              |     |          |       | p 0    |          | |sa)z(sa)− |     |          |
|     |         | sas′     |           |              |     | s′       |       |        | sa       |            |     | a′       |
Let us now consider the problem that arises if we swap the minimum and the maximum in the saddle-point
problem:
maxminL(u,z)
|     |     |     |     |     |     | z≥0 | u   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Defining
(cid:88)
|     |     |     |     |     | g(z)= |     | z(sa)p(s′ | |sa)r(sas′) |     |     |     |     |
| --- | --- | --- | --- | --- | ----- | --- | --------- | ----------- | --- | --- | --- | --- |
sas′
we have
|     |     |     |     |     | (cid:34) |     | (cid:32) |     |     |     |     | (cid:33)(cid:35) |
| --- | --- | --- | --- | --- | -------- | --- | -------- | --- | --- | --- | --- | ---------------- |
minL(u,z)=g(z)+min (cid:88) u(s′) (s′)+γ (cid:88) p(s′ (cid:88) z(s′a′)
|     |     |     |     |     |     |     | p 0 |     | |sa)z(sa)− |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- |
|     |     | u   |     |     | u   |     |     |     |            |     |     |     |
|     |     |     |     |     | s′  |     |     | sa  |            |     | a′  |     |
The second term takes value unless all terms in brackets vanish. Therefore we can rephrase the max-min
−∞
| saddle | problem | as a constrained |     | maximization |     | problem: |     |     |     |     |     |     |
| ------ | ------- | ---------------- | --- | ------------ | --- | -------- | --- | --- | --- | --- | --- | --- |
maxminL(u,z)=maxg(z)
|     |     |     |     |     | z≥0 | u   |     | z∈Θ |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
23

where
(cid:40) (cid:41)
|     |     |     |     |     |     | (cid:88) |     | (cid:88) |
| --- | --- | --- | --- | --- | --- | -------- | --- | -------- |
Θ= z ∈R|S||A| :z(sa)≥0, p (s′)+γ p(s′ |sa)z(sa)− z(s′a′)=0 ∀s′ ∈S
0
|                  |         |     |     |     |     | sa  |     | a′  |
| ---------------- | ------- | --- | --- | --- | --- | --- | --- | --- |
| The optimization | problem |     |     |     |     |     |     |     |
maxg(z)
z∈Θ
| is the dual | MDP linear | program. |     |     |     |     |     |     |
| ----------- | ---------- | -------- | --- | --- | --- | --- | --- | --- |
minf(u)=maxg(z)
|           |                    |         |      |     | u∈Ω |     | z∈Θ |     |
| --------- | ------------------ | ------- | ---- | --- | --- | --- | --- | --- |
| First, it | is straightforward | to show | that |     |     |     |     |     |
minf(u)=minmaxL(u,z)≥maxminL(u,z)=maxg(z)
|     |     | u∈Ω |     | u   | z≥0 |     | z≥0 u | z∈Θ |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- |
To prove the converse, let us introduce the expected discounted occupancy of the state-action pair under a
sa
| given policy | π:              |          |        |          |                     |      |                   |               |
| ------------ | --------------- | -------- | ------ | -------- | ------------------- | ---- | ----------------- | ------------- |
|              |                 |          |        | (cid:34) |                     |      | (cid:35)          |               |
|              |                 |          |        | ∞        |                     |      |                   | ∞             |
|              |                 |          |        | (cid:88) |                     |      |                   | (cid:88)      |
|              |                 | ηπ(sa)≡E |        |          | γt1(S               | =s,A | =a) =             | γtρ (s)π(a|s) |
|              |                 |          |        |          |                     | t t  |                   | t             |
|              |                 |          |        | t=0      |                     |      |                   | t=0           |
| and remark   | that the return | of       | policy | π can    | be written          | as   |                   |               |
|              |                 |          | Gπ(p   |          | (cid:88) ηπ(sa)p(s′ |      | |sa)r(sas′)=g(ηπ) |               |
0 )=
sas′
| The key | point here is | to recognize | that | ηπ ∈Θ | for | any policy | π. Indeed, |     |
| ------- | ------------- | ------------ | ---- | ----- | --- | ---------- | ---------- | --- |
ηπ(sa)≥0
∀s,a
| and from | the Kolmogorov-Chapman |     |          | equations  |     |        |                     |     |
| -------- | ---------------------- | --- | -------- | ---------- | --- | ------ | ------------------- | --- |
|          |                        |     | (cid:88) | ηπ(s′a′)=p |     | (s′)+γ | (cid:88) ηπ(sa)p(s′ |     |
|sa)
0
|                |                 |        | a′          |       |      |         | sa        |     |
| -------------- | --------------- | ------ | ----------- | ----- | ---- | ------- | --------- | --- |
| In particular, | for the optimal | policy |             | π∗ we | have |         |           |     |
|                |                 |        | minf(u)=Gπ∗ |       |      | )=g(ηπ∗ |           |     |
|                |                 |        |             |       |      | (p      | )≤maxg(z) |     |
0
|                 |            |      |     | u∈Ω |     |     |     | z∈Θ |
| --------------- | ---------- | ---- | --- | --- | --- | --- | --- | --- |
| which concludes | the proof. |      |     |     |     |     |     |     |
| This result     | also shows | that |     |     |     |     |     |     |
=argmaxg(z)=ηπ∗
z∗
z∈Θ
| and therefore | the optimal | policy | can | be read | from | z∗ as follows: |        |     |
| ------------- | ----------- | ------ | --- | ------- | ---- | -------------- | ------ | --- |
|               |             |        |     |         |      | ηπ∗(sa)        | z∗(sa) |     |
π∗(a|s)=
|     |     |     |     |     |     | (cid:80) | = (cid:80) |        |
| --- | --- | --- | --- | --- | --- | -------- | ---------- | ------ |
|     |     |     |     |     |     | ηπ∗(sa)  |            | z∗(sa) |
|     |     |     |     |     |     | a        |            | a      |
If the denominator is zero for some state s, so is the numerator and the choice of the optimal policy is arbitrary.
| This has | no consequences | because | if  |     |          |     |     |     |
| -------- | --------------- | ------- | --- | --- | -------- | --- | --- | --- |
|          |                 |         |     |     | (cid:88) | ηπ∗ |     |     |
(sa)=0
a
then the state s is never visited under the optimal policy and so it is irrelevant.
24

Chapter 4
Monte Carlo Methods
Here we do not assume complete knowledge of the environment. Monte Carlo methods require only experience,
meaningsamplesequencesofstates,actionsandrewardsfromactualorsimulatedinteractionwithanenvironment.
Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics,
yet can still attain optimal behavior. Learning from simulated experience is also powerful. Altough a morel is
required,themodelneedonlygeneratesampletransitions,notthecompleteprobabilitydistributionsofallpossible
transitions that is required for dynamic programming (DP).
Monte Carlo methods (MC) are ways of solving the RL problem based on averaging sample returns. Here are
defined only for episodic tasks. Only on the completion of an episode are value estimates and policies changed.
Essentially, MC methods sample and average returns for each state-action pair much like the bandit methods
whichsampleandaveragerewardsforeachaction. Themaindifferenceisthatnowtherearemultiplestates, each
acting like a different bandit problem and the different bandit problems are interrelated. That is, the return after
taking an action in one state depends on the actions taken in later states in the same episode. Because all the
actionselectionsareundergoinglearning, theproblembecomesnonstationaryfromthepointofviewoftheearlier
state.
To handle the nonstationarity, we adapt the idea of general policy iteration (GPI) developed earlier for DP.
Whereas there we computed value functions from knowledge of the MDP, here we learn value functions from
sample returns with the MDP.
4.1 Monte Carlo Prediction
We begin by considering MC methods for learning the state-value function for a given policy. An obvious way to
estimate it from experience is simply to average the returns observed after visits to that state. As more returns
are observed, the average should converge to the expected value. Suppose we wish to estimate v (s), the value of
π
astatesunderpolicyπ givenasetofepisodesobtainedbyfollowingπ andpassingthroughs. Eachoccurrenceof
state s in an episode is called a visit to s. The first-visit MC method estimates v (s) as the average of the returns
π
following the first visits to s, whereas the every-visit MC method averages the returns following all visits to s.
Both first-visit and every-visit MC converge to v (s) as the number of visits to s goes to infinity.
π
We can extend the idea of backup diagrams to Monte Carlo algorithms. In general, a backup diagram represents
25

the quantity being updated as the root node at the top, and shows below it the transitions, rewards, and leaf
nodes whose estimated values contribute to that update.
For Monte Carlo estimation of (v ), the root is a state node. Below it, the diagram contains the entire trajectory
π
generated in a single episode, from that state until the terminal state. The return observed along this complete
episode is then used to update the value estimate of the starting state.
A key property of Monte Carlo methods is that state-value estimates are independent in the sense that they do
not bootstrap from estimates of successor states. Unlike dynamic programming or temporal-difference methods,
Monte Carlo methods wait until the end of an episode and use the actual sampled return.
This also means that the computational cost of estimating the value of a single state does not directly depend on
the total number of states in the state space. As a result, Monte Carlo methods can be particularly useful when
we are interested only in the value of one state or a small subset of states. In that case, we can generate many
sample episodes starting from the states of interest, average the observed returns from those episodes, and ignore
the remaining states.
4.2 Monte Carlo Estimation of Action Values
Ifamodelisnotavailable,itisoftenmoreusefultoestimateactionvaluesratherthanstatevalues. Whenamodel
is available, state values are sufficient to determine a policy: one can look ahead one step, evaluate the immediate
reward and the value of the resulting next state, and choose the action that gives the best expected outcome.
Without a model, however, state values alone are not enough, because the agent cannot predict the consequences
of each available action. Therefore, it must explicitly estimate the value of each state-action pair.
The policy evaluation problem for action values consists of estimating (q (s,a)), the expected return obtained by
π
starting in state (s), taking action (a), and then following policy (π) thereafter. A state-action pair ((s,a)) is said
to be visited in an episode if, at some point during that episode, the agent visits state (s) and selects action (a).
Themaincomplicationisthatmanystate-actionpairsmayneverbevisited. Forexample, if(π)isadeterministic
policy, then in each state the agent will always choose the same action. As a result, returns will only be observed
forthataction, whiletheotherpossibleactionsfromthesamestatewillnotreceiveanyupdates. Thisisaserious
issue because the purpose of learning action values is precisely to compare the actions available in each state and
choose among them. This is known as the problem of maintaining exploration.
For Monte Carlo policy evaluation of action values to work properly, we must ensure continual exploration. One
way to do this is to assume that each episode starts from a state-action pair, and that every state-action pair has
a nonzero probability of being selected as the starting point. Under this assumption, all state-action pairs will be
visitedinfinitelyofteninthelimitofinfinitelymanyepisodes. Thisiscalledtheassumptionof exploring starts.
4.3 Monte Carlo Control
Let’s consider a MC version of classical policy iteration. In this method, we perform alternating complete steps
of policy evaluaton and policy improvement, beginning with an arbitrary policy π and ending with the optimal
0
policy and optimal action-value function:
π →E q →I π →E q →I π →E ...→I π →E q ,
0 π0 1 π1 2 ∗ ∗
where →E denotes a complete policy evaluation and →I denotes a complete policy improvement.
Policy evaluation is done exactly as described in the preceding section. Many episodes are experienced, with the
approximate action-value function approaching the true function asymptotically. For the moment, let us assume
that we do indeed observe an infinite number of episodes and that, in addition, the episodes are generated with
exploring starts. Under these assumptions, the MC methods will compute each q exactly, for arbitrary π .
πk k
Policy improvement is done by making the policy greedy with respect to the current value function. In this case
we have an action-value function, and therefore no model is needed to construct the greedy policy. For any given
action-value function q, the corresponding greedy policy is the one that, for each s∈S, deterministically chooses
an action with maximal action-value:
π(s)=argmaxaq(s,a)
Policy improvement then can be done by constructing each π as the greedy policy with respect to q .
k+1 πk
26

The first strong assumption we made (exploring starts) is easy to remove. In both DP and MC cases there are
two ways to solve the problem. One is to hold firm to the idea of approximating q in each policy evaluation.
πk
Measurements and assumptions are made to obtain bounds on the magnitude and probability of error in the
estimates, and then sufficient steps are taken during each policy evaluation to assure that these bounds are
sufficiently small.
The second approach is to give up trying to complete policy evaluation before returning to policy improvement.
On each evaluation step we move the value function toward q , but we don’t expect to actually get close except
πk
over many steps. Below is the pseudocode of the so called Monte Carlo with Exploring Starts (MCES).
Figure 4.1: Monte Carlo with Exploring Starts (MCES)
4.4 Monte Carlo Control without Exploring Starts
How can we avoid the unlikely assumption of exploring starts? The only general way to ensure that all actions
are selected infinitely often is for the agent to continue to select them. There are two approaches:
• On-policy methods - attempt to evaluate or improve the policy that is used to make decisions
• Off-policy methods - evaluate or improve a policy different from the one used to generate the data
In on-policy control methods, the policy is generally soft, meaning that π(a|s)>0 for all s∈S and all a∈A(s),
but gradually shifted closer and closer to a deterministic optimal policy.
Theon-policycontrolmethodwepresentinthissectionusesϵ-greedypolicies,meaningthatmostofthetimethey
choose an action that has maximal estimated action value, but with probability ϵ they instead select an action at
random.
As in MCES, we use first-visit MC methods to estimate the action-value function for the current policy. Without
theassumptionofexploringstarts,however,wecannotsimplyimprovethepolicybymakingitgreedywithrespect
to the current value function, because that would prevent further exploration of nongreedy actions. Fortunately,
GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a
greedy policy. In our on-policy method we will move it only to an ϵ-greedy policy. For any ϵ-soft policy, π, any
ϵ-greedy policy with respect to q is guaranteed to be better than or equal to π.
π
27

|     |     |     |     |        |      | Monte Carlo | Control | with ϵ-greedy | policies |     |
| --- | --- | --- | --- | ------ | ---- | ----------- | ------- | ------------- | -------- | --- |
|     |     |     |     | Figure | 4.2: |             |         |               |          |     |
So, the policy ieration workds for ϵ-soft policies. Using the natural notion of greedy policy for ϵ-soft policies, one
is assured of improvement on every step, except when the best policy has been found among the ϵ-soft policies.
This analysis is independent of how the action-value functions are determined at each stage, but it does assume
that they are computed exactly. Now we only achieve the best policy among the ϵ-soft policies, but on the other
| hand, we | have       | eliminated | the        | assumption | of  | exploring  | starts. |          |     |     |
| -------- | ---------- | ---------- | ---------- | ---------- | --- | ---------- | ------- | -------- | --- | --- |
| 4.5      | Off-policy |            | Prediction |            | via | Importance |         | Sampling |     |     |
All learning control methods face a dilemma: they seek to learn action values conditional on subsequent optimal
behavior, but they need to behave non-optimally in order to explore all actions. How can they learn about the
optimalpolicywhilebehavingaccordingtoanexploratorypolicy? Theon-policyapproachintheprecedingsection
is actually a compromise, i.e., it learns action values not for the optimal policy but for a near-optimal policy that
still explores. A more straightforward approach is to use two policies, one that is learned about and that becomes
the optimal policy, and one that is more exploratory and is used to generate behavior. The first is the
target
policy, while the second is the behavior policy. In this case we say that learning is from data "off" the target
| policy, | and the | overall | process | is  | termed |     | learning. |     |     |     |
| ------- | ------- | ------- | ------- | --- | ------ | --- | --------- | --- | --- | --- |
off-policy
Off-policymethodsareoftenofgreatervarianceandareslowertoconverge. Ontheotherhand,off-policymethods
| are more | powerful | in  | general. |     |     |     |     |     |     |     |
| -------- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
We begin by considering the prediction problem, where both target and behavior policies are fixed. Suppose we
wish to estimate v or q , but all we have are episodes following another policy b, where b̸=π. In this case, π is
|     |     | π   | π   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
the target policy, is the behavior policy, and both policies are considered fixed and given.
b
We require that every action taken under is also taken, at least occasionally, under b. This is called the
π
assumption of coverage. It follows from coverage that b must be stochastic in states where it is not identical to π.
| The target | policy | π,  | on the | other | hand, may | be deterministic. |     |     |     |     |
| ---------- | ------ | --- | ------ | ----- | --------- | ----------------- | --- | --- | --- | --- |
Almost all off-policy methods utilize importance sampling, a general technique for estimating expected values
under one distribution given samples from another. We apply importance sampling to off-policy learning by
weightingreturnsaccordingtotherelativeprobabilityoftheirtrajectoriesoccurringunderthetargetandbehavior
policies, called the ratio. Given a starting state , the probability of the subsequent state-
|                   |     | importance-sampling |      |       |            |       |         | S t        |       |     |
| ----------------- | --- | ------------------- | ---- | ----- | ---------- | ----- | ------- | ---------- | ----- | --- |
| action trajectory |     | occurring           |      | under | any policy | π is: |         |            |       |     |
|                   |     |                     | Pr{A | ,S    | ,A ,...,S  | |S    | ,A      | ≈π}        |       |     |
|                   |     |                     |      | t t+1 | t+1        | T     | t t:T−1 |            |       |     |
|                   |     |                     |      | =π(A  | |S )p(S    | |S    | ,A )π(A | |S )...p(S | |S ,A | )   |
|                   |     |                     |      |       | t t        | t+1   | t t     | t+1 t+1    | T T−1 | T−1 |
T−1
(cid:89)
|     |     |     |     | =   | π(A |S | )p(S  | |S ,A | )   |     |     |
| --- | --- | --- | --- | --- | ------ | ----- | ----- | --- | --- | --- |
|     |     |     |     |     | k      | k k+1 | k     | k   |     |     |
k=t
| where | here | is the state-transition |     |     | probability | function. |     |     |     |     |
| ----- | ---- | ----------------------- | --- | --- | ----------- | --------- | --- | --- | --- | --- |
p
28

| The importance-sampling |     | ratio is: |           |            |            |          |     |
| ----------------------- | --- | --------- | --------- | ---------- | ---------- | -------- | --- |
|                         |     |           | (cid:81)T |            | T −1       |          |     |
|                         |     |           | − 1π(A |S | )p(S |S ,A | ) (cid:89) | π(A |S ) |     |
|                         |     | ρ =       | k= T k k  | k+1 k      | k =        | k k      |     |
|                         |     | t:T−1     | (cid:81)T |            |            |          |     |
|                         |     |           | − 1b(A |S | )p(S |S ,A | )          | b(A |S ) |     |
|                         |     |           | k= t k k  | k+1 k      | k k=t      | k k      |     |
Recall that we wish to estimate the expected returns (values) under the target policy, but all we have are returns
G due to the behavior policy. These returns have the wrong expectation E[G |S =s]=v (s) and so cannot be
| t   |     |     |     |     |     | t t | b   |
| --- | --- | --- | --- | --- | --- | --- | --- |
averaged to obtain v This is where importance sampling comes in. The ratio ρ transforms the returns to
|                |          | π      |       |          |     | t:T−1 |     |
| -------------- | -------- | ------ | ----- | -------- | --- | ----- | --- |
| have the right | expected | value: |       |          |     |       |     |
|                |          |        | E[ρ G | |S =s]=v | (s) |       |     |
|                |          |        | t:T−1 | t t      | π   |       |     |
We can define the set of all time steps in which state s is visited, denoted T(s). Also, let T(t) denote the first
time of termination following time t, and denote the return after up through T(t). Then are
|     |     |     | G t |     | t   |     | {G t } |
| --- | --- | --- | --- | --- | --- | --- | ------ |
t∈T(s)
the returns that pertain to state s, and (cid:8) ρ (cid:9) are the corresponding importance-sampling rations. To
t:T(t)−1
t∈T(s)
estimate v (s), we simply scale the returns by the ratios and average the results:
π
(cid:80)
|     |     |     |     | ρ               | G   |     |     |
| --- | --- | --- | --- | --------------- | --- | --- | --- |
|     |     |     |     | t∈T(s) t:T(t)−1 | t   |     |     |
V(s)=
|T(s)|
When importance sampling is done as a simple average in this way it is called ordinary importance sampling.
An important alternative is weighted importance sampling, which uses a weighted average, defined as:
(cid:80)
|     |     |     |       | ρ t:T(t)−1 | G t |     |     |
| --- | --- | --- | ----- | ---------- | --- | --- | --- |
|     |     |     | V(s)= | t∈T(s)     |     |     |     |
(cid:80)
ρ
t∈T(s) t:T(t)−1
| 4.6 Incremental |     | Implementation |     |     |     |     |     |
| --------------- | --- | -------------- | --- | --- | --- | --- | --- |
MCpredictionmethodscanbeimplementedincrementally,onanepisode-by-episodebasis,usingextensionsofthe
techniques described in Chapter 3. In MC methods we average returns. In all other aspects the same methods
of Chapter 3 can be used for on-policy MC methods. For off-policy MC methods, we need to separately consider
those that use ordinary importance sampling and those that use weighted importance sampling.
In ordinary importance sampling, the returns are scaled by the importance sampling ratio ρ , then simply
t:T(t)−1
averaged. This leaves the case of off-policy methods using weighted importance sampling. Here we have to form
a weighted average of the returns, and a slightly different incremental algorithm is required. We omit the full
| derivation here. |     |       |               |     |     |     |     |
| ---------------- | --- | ----- | ------------- | --- | --- | --- | --- |
| 4.7 Off-policy   |     | Monte | Carlo Control |     |     |     |     |
Recall that the distinguishing feature of on-policy methods is that they estimate the value of a policy while using
it for control. In off-policy methods these two functions are separated. The policy used to generate behavior,
called the behavior policy, may in fact be unrelated to the policy that is evaluated and improved, called the target
policy. An advantage of this separation is that the target policy may be deterministic (e.g. greedy), while the
| behavior policy | can continue | to sample | all possible actions. |     |     |     |     |
| --------------- | ------------ | --------- | --------------------- | --- | --- | --- | --- |
These techniques require that the behavior policy has a nonzero probability of selecting all actions that might be
selected by the target policy (coverage). To explore all possibilities, we require that the behavior policy be soft.
29

Figure 4.3: Off-policy Monte Carlo Control
A potential problem is that this method learns only from the tails of episodes, when all of the remaining actions
in the episode are greedy. If nongreedy actions are common, then learning will be slow, particularly for states
appearing in the early portions of long episodes.
30

| Chapter             | 5   |     |     |          |     |     |     |
| ------------------- | --- | --- | --- | -------- | --- | --- | --- |
| Temporal-Difference |     |     |     | Learning |     |     |     |
TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo
methods, TD methods can learn directly from raw experience without a model of the environment’s dynamics.
Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final
| outcome (they | bootstrap). |     |     |     |     |     |     |
| ------------- | ----------- | --- | --- | --- | --- | --- | --- |
Asusual, westartbyfocusingonthepolicyevaluationorprediction problem, theproblemofestimatingthevalue
functionv foragivenpolicyπ. Forthecontrol problem,DP,TDandMonteCarlomethodsallusesomevariation
π
| of generalized | policy iteration | (GPI). |     |     |     |     |     |
| -------------- | ---------------- | ------ | --- | --- | --- | --- | --- |
| 5.1 TD         | Prediction       |        |     |     |     |     |     |
Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience
following a policy π, both methods update their estimate V of v for the nonterminal states S occurring in that
π t
experience.
|     |     |     | V(S )←V(S | )+α[G | −V(S )] |     | (5.1) |
| --- | --- | --- | --------- | ----- | ------- | --- | ----- |
|     |     |     | t         | t t   | t       |     |       |
Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to V(S ), TD
t
methods need to wait only until the next time step. At time they immediately form a target and make a
t+1
useful update using the observed reward R and the estimate V(S ). The simplest TD method makes the
|     |     |     | t+1 |     | t+1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
update
(5.2)
|                |           | V(S         | t )←V(S t )+α[R | t+1 +γV(S    | t+1 −V(S | t ))]     |     |
| -------------- | --------- | ----------- | --------------- | ------------ | -------- | --------- | --- |
| This TD method | is called | one-step TD | (TD(0)).        |              |          |           |     |
|                |           | Figure      | 5.1: Pseudocode | for one-step | TD       | learning. |     |
Because TD(0) bases its update in part on an existing estimate, we say that it is a bootstrapping method, like DP.
31

|     |     |     |     |     | v (s)=E | [G |S | =s]    |         | (5.3) |
| --- | --- | --- | --- | --- | ------- | ----- | ------ | ------- | ----- |
|     |     |     |     |     | π       | π t   | t      |         |       |
|     |     |     |     |     | =E      | [R    | +γG    | |S =s]  |       |
|     |     |     |     |     |         | π t+1 | t+1    | t       |       |
|     |     |     |     |     | =E      | [R    | +γv (S | )|S =s] | (5.4) |
|     |     |     |     |     |         | π t+1 | π      | t+1 t   |       |
Monte Carlo methods use an estimate of (5.3) as a target, whereas DP methods use an estimate of (5.4) as a
target. The Monte Carlo target is an estimate because the expected value in (5.3) is not known; a sample return
is used in place of the real expected return. The DP target is an estimate not because of the expected values,
which are assumed to be completely provided by a model of the environment, but because v (S ) is not known
π t+1
and the current estimate, V(S ), is used instead. The TD target is an estimate for both reasons: it samples the
t+1
expected values in (5.4) and it uses the current estimate V instead of the true v . Thus, TD methods combine
π
| the sampling | Monte | Carlo | with | the bootstrapping |     | of DP. |     |     |     |
| ------------ | ----- | ----- | ---- | ----------------- | --- | ------ | --- | --- | --- |
We refer to TD and Monte Carlo updates as sample updates because they involve looking ahead to a sample
successor state (or state-action pair), using the value of the successor and the reward along the way to compute
a backed-up value, and then updating the value of the original state (or state-action pair) accordingly. Sample
updatesdifferfromtheexpected updatesofDPmethodsinthattheyarebasedonasinglesamplesuccessorrather
| than on | a complete | distribution |     | of all | possible | successors. |     |     |     |
| ------- | ---------- | ------------ | --- | ------ | -------- | ----------- | --- | --- | --- |
Finally, note that the quantity in brackets in update is a sort of error, measuring the difference between
TD(0)
the estimated value of S and the better estimate R +γV(S ). This quantity, called the TD error, arises in
|               |            |     | t   |     |            | t+1   |         | t+1 |       |
| ------------- | ---------- | --- | --- | --- | ---------- | ----- | ------- | --- | ----- |
| various forms | throughout |     | RL: |     |            |       |         |     |       |
|               |            |     |     |     | δ =R       | +γV(S | )−V(S   | )   | (5.5) |
|               |            |     |     |     | t          | t+1   | t+1     | t   |       |
| 5.2           | Advantages |     | of  | TD  | Prediction |       | Methods |     |       |
TD methods update their estimates based in part on other estimates. They learn a guess from a guess., i.e., they
bootstrap.
TD methods have an advantage over DP methods in that they do not require a model of the environment, of its
reward and next-state probability distributions. The next most obvious advantage of TD methods over Monte
Carlo methods is that they are naturally implemented in an online, fully incremental fashion. With Monte Carlo
methods one must wait until the end of an episode, because only then is the return known, whereas with TD
| methods | one need | wait | only | one time | step. |     |     |     |     |
| ------- | -------- | ---- | ---- | -------- | ----- | --- | --- | --- | --- |
Certainly it is convenient to learn one guess from the next, without waiting for an actual outcome, but can we
still guarantee convergence to the correct answer? Happily, the answer is yes.
| 5.3 | Optimality |     | of  | TD(0) |     |     |     |     |     |
| --- | ---------- | --- | --- | ----- | --- | --- | --- | --- | --- |
Suppose that only a finite amount of experience is available. In this case, a common approach with incremental
learning methods is to reuse the available experience repeatedly until the method converges to a stable solution.
Given an approximate value function (V), the increments specified by (5.1) or (5.2) are computed for every time
step (t) at which a nonterminal state is visited. However, the value function is not updated after each individual
increment. Instead, all increments are accumulated, and the value function is updated only once, using the sum
of these increments.
Afterthisupdate, theentirebatchofexperienceisprocessedagainusingthenewvaluefunction, producinganew
total increment. This procedure is repeated until the value function converges. This approach is called batch
updating, because updates are performed only after processing the complete of training data.
batch
Batch Monte Carlo methods converge to the estimates that minimize the mean squared error on the training set.
In contrast, batch TD(0) converges to the estimates that would be exactly correct for the maximum-likelihood
model of the Markov process. In general, the maximum-likelihood estimate of a parameter is the parameter value
| that assigns | the    | highest   | probability | to  | the observed | data. |     |     |     |
| ------------ | ------ | --------- | ----------- | --- | ------------ | ----- | --- | --- | --- |
| 5.4          | Sarsa: | On-policy |             | TD  | Control      |       |     |     |     |
We turn now to the use of TD prediction methods for the control problem. As with Monte Carlo methods, we
face the need to trade off exploration and exploitation, and again approaches fall into two main classes: on-policy
| and off-policy. |     | In this | section | we present | an on-policy |     | TD control | method. |     |
| --------------- | --- | ------- | ------- | ---------- | ------------ | --- | ---------- | ------- | --- |
32

The first step is to learn an action-value function rather than a state-value function. We must estimate
q π (S,a)
for the current behavior policy π for all states s and actions a. This can be done using essentially the same TD
| method described | above | for learning | v . |     |     |     |     |     |     |
| ---------------- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- |
π
Intheprevioussectionweconsideredtransitionsfromstatetostateandlearnedvaluesofstates. Nowweconsider
transitions from state-action pair to state-action pair, and learn the values of state-action pairs. Formally these
cases are identical: they are both Markov chains with a reward process. The theorems assuring the convergence
of state values under TD(0) also apply to the corresponding algorith, for action values:
|     |     | Q(S ,A | )←Q(S | ,A )+α[R | +γQ(S | ,A  | )−Q(S | ,A )]. | (5.6) |
| --- | --- | ------ | ----- | -------- | ----- | --- | ----- | ------ | ----- |
|     |     | t t    |       | t t      | t+1   | t+1 | t+1   | t t    |       |
This rule uses every element of the quintuple of events, ), that make up a transition from
|     |     |     |     |     | (S t ,A | t ,R t+1 ,S | t+1 ,A t+1 |     |     |
| --- | --- | --- | --- | --- | ------- | ----------- | ---------- | --- | --- |
one state-action pair to the next. This quintuple gives rise to the name Sarsa for the algorithm.
As in all on-policy methods, we continually estimate q for the behavior policy π, and at the same time change π
π
| toward greediness | with | respect to | q . |     |     |     |     |     |     |
| ----------------- | ---- | ---------- | --- | --- | --- | --- | --- | --- | --- |
π
|                 |     |            |     | Figure 5.2: | Pseudocode | for Sarsa. |     |           |     |
| --------------- | --- | ---------- | --- | ----------- | ---------- | ---------- | --- | --------- | --- |
| 5.5 Q-learning: |     | Off-policy |     | TD          | Control    |            |     |           |     |
|                 |     |            |     |             | (cid:104)  |            |     | (cid:105) |     |
(5.7)
|     |     | Q(S t ,A t )←Q(S |     | t ,A t )+α | R t+1 +γmaxQ(S |     | t+1 ,a)−Q(S | t ,A t ) |     |
| --- | --- | ---------------- | --- | ---------- | -------------- | --- | ----------- | -------- | --- |
a
This is an off-policy TD control algorithm known as Q-learning. Here, the learned action-value function, Q,
directly approximates , the optimal action-value function, independent of the policy being followed. The policy
q ∗
still has an effect in that it determines which state-action pairs are visited and updated. However, all that is
required for the correct convergence is that all pairs continue to be updated. This is a minimal requirement in the
sense that any method guaranteed to find optimal behavior in the general case must require it. has been shown
Q
to converge with probability 1 to q . The Q-learning algorithm is shown below:
∗
33

|              |       | Figure | 5.3: Pseudocode | for Q-learning. |     |     |
| ------------ | ----- | ------ | --------------- | --------------- | --- | --- |
| 5.6 Expected | Sarsa |        |                 |                 |     |     |
Consider the learning algorithm that is just like Q-learning except that instead of the maximum over the next
state-actionpairsitusestheexpectedvalue,takingintoaccounthowlikelyeachactionisunderthecurrentpolicy.
| That is, consider | the algorithm, | with the update | rule: |     |     |     |
| ----------------- | -------------- | --------------- | ----- | --- | --- | --- |
+γE (5.8)
|     | Q(S t ,A | t )←Q(S t ,A t )+α[R | t+1      | π [Q(S t ,A t | )|S t+1 ]−Q(S | t ,A t )] |
| --- | -------- | -------------------- | -------- | ------------- | ------------- | --------- |
|     |          |                      | (cid:34) |               |               | (cid:35)  |
(cid:88) (5.9)
|     |     | =Q(S t ,A t )+α | R t+1 +γ | (a|S t+1 )Q(S | t+1 ,a)−Q(S | t ,A t ) |
| --- | --- | --------------- | -------- | ------------- | ----------- | -------- |
a
but that otherwise follows the schema of Q-learning. Given the next state, S , this algorithm moves determin-
t+1
in the same direction as Sarsa moves in expectation, and accordingly it is called Sarsa.
| istically |     |     |     |     |     | Expected |
| --------- | --- | --- | --- | --- | --- | -------- |
Expected Sarsa is more complex computationally than Sarsa but, in return, it eliminates the variance due to the
| random selection | of A . |     |     |     |     |     |
| ---------------- | ------ | --- | --- | --- | --- | --- |
t+1
Figure 5.4: The backup diagrams for Q-learning (left) and Expected Sarsa (right).
34

Figure 5.5: Interim and asymptotic performance of TD control methods on the cliff-walking task as a function of α. All
algorithms used an ϵ-greedy policy with 0.1. Asymptotic performance is an average over 100,000 episodes whereas
ϵ =
| interim performance | is           | an average | over the | first 100 episodes. |          |     |     |
| ------------------- | ------------ | ---------- | -------- | ------------------- | -------- | --- | --- |
| 5.7                 | Maximization | Bias       | and      | Double              | Learning |     |     |
All the control algorithms that we have discussed so far involve a maximization in the construction of their target
policies. In these algorithms, a maximum over estimated values is used implicitly as an estimate of the maximum
value, which can lead to a significant positive bias. To see why, consider a single state s where there are many
actions a whose true values, q(s,a), are all zero but whose estimated values, Q(s,a), are uncertain and thus
distributed some above and some below zero. The maximum of the true values is zero, but the maximum of the
| estimates | is positive, | a positive bias. | We  | call this maximization | bias. |     |     |
| --------- | ------------ | ---------------- | --- | ---------------------- | ----- | --- | --- |
Aretherealgorithmsthatavoidthisbias? Considerabanditcaseinwhichwehavenoisyestimatesofthevalueof
eachofmanyactions, obtainedassimpleaveragesoftherewardsreceivedonalltheplayswitheachaction. There
will be a positive maximization bias if we use the maximum of the estimates as an estimate of the maximum of
the true values.
Suppose we divided the plays in two sets and used them to learn two independent estimates, call them Q (a)
1
and (a), each estimate of the true value q(a), for all A. We could then use one estimate to determine
| Q 2 |     |     |     |     | a ∈ |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
the maximizing action A∗ = argmaxaQ (a) and the other, Q in this case, to provide the estimate of its value,
|     |     |     |     | 1   | 2   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
Q (A∗)=Q (argmaxaQ (a)). This estimate will then be unbiased in the sense that E[Q (A∗)]=q(A∗).
| 2   | 2   | 1   |     |     |     |     | 2   |
| --- | --- | --- | --- | --- | --- | --- | --- |
Thisistheideaof double learning. Notethatalthoughwelearntwoestimates, onlyoneestimateisupdatedon
each play; double learning doubles the memory requirements, but does not increase the amount of computation
per step.
The idea of double learning extends naturally to algorithms for full MDPs. As an example, the double learning
algorithm analogous to Q-learning, called Double Q-learning, divides the time steps in two, perhaps by flipping a
| coin on | each step. If | the coin comes | up heads, | the update | is  |     |     |
| ------- | ------------- | -------------- | --------- | ---------- | --- | --- | --- |
Q (S ,A )←Q (S ,A )+α[R +γQ (S ,argmaxaQ (S ,a))−Q (S ,A )]. (5.10)
|     | 1 t | t 1 | t t | t+1 | 2 t+1 | 1 t+1 | 1 t t |
| --- | --- | --- | --- | --- | ----- | ----- | ----- |
If the coin comes up tails, then the same update is done with Q−1 and Q switched, so that Q is updated.
|     |     |     |     |     |     | 2   | 2   |
| --- | --- | --- | --- | --- | --- | --- | --- |
35

| Figure 5.6: | The Double | Learning algorithm. |
| ----------- | ---------- | ------------------- |
36

| Chapter | 6             |     |     |
| ------- | ------------- | --- | --- |
| n-step  | Bootstrapping |     |     |
Neither MC methods nor one-step TD methods are always the best. In this chapter we present
n-step TD
methodsthatgeneralizebothapproaches,sothatonecanshiftfromonetotheothersmoothlyasneededtomeet
the demands of a particular task. n-step methods span a spectrum with MC methods at one end and one-step
TD methods at the other. The best methods are often intermediate between the two extremes.
As usual, we first consider the prediction problem and then the control problem. That is, we first consider how
n-stepmethodscanhelpinpredictingreturnsasafunctionofstateforafixedpolicy(i.e.,inestimatingv ). Then
π
| we extend | the ideas to | action values and | control methods. |
| --------- | ------------ | ----------------- | ---------------- |
| 6.1       | n-step TD    | Prediction        |                  |
Considerestimatingv fromsampleepisodesgeneratedusingπ. MonteCarlomethodsperformanupdateforeach
π
state based on the entire sequence of observed rewards from that state until the end of the episode. The update
of one-step TD methods, on the other hand, is based on just the one next reward, bootstrapping from the value
of the state one step later as a proxy for the remaining rewards. One kind of intermediate method, then, would
| perform | an update based | on an intermediate | number of rewards. |
| ------- | --------------- | ------------------ | ------------------ |
Figure 6.1: Thebackupdiagramofn-stepmethods. Thesemethodsformaspectrumrangingfromone-stepTDmethods
| to Monte | Carlo methods. |     |     |
| -------- | -------------- | --- | --- |
37

Figure 6.1 shows the backup diagram of the spectrum of n-step updates for , with the one-step TD update on
v π
| the left | and the up-until-termination |     |     | Monte | Carlo | update | on  | the right. |     |     |     |
| -------- | ---------------------------- | --- | --- | ----- | ----- | ------ | --- | ---------- | --- | --- | --- |
Methods in which the TD extends over n steps are called n-step TD methods. We know that in Monte Carlo
updates the estimate of v (S ) is updated in the direction of the complete return:
π t
|     |     |     |     |     |     | +γ2R |     | +···+γT−t−1R |     |     | (6.1) |
| --- | --- | --- | --- | --- | --- | ---- | --- | ------------ | --- | --- | ----- |
|     |     |     | G   | =R  | +γR |      |     |              |     |     |       |
|     |     |     |     | t   | t+1 | t+2  | t+3 |              |     | T   |       |
where is the last time step of the episode. Let us call this quantity the of the update. Whereas in
| T   |     |     |     |     |     |     |     |     |     | target |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- |
MonteCarloupdatesthetargetisthereturn,inone-stepupdatesthetargetisthefirstrewardplusthediscounted
| estimated | value of the | next | state, | which | we call | the one-step |     | return: |     |     |       |
| --------- | ------------ | ---- | ------ | ----- | ------- | ------------ | --- | ------- | --- | --- | ----- |
|           |              |      |        |       | Gt:t+1  |              |     |         |     |     | (6.2) |
|           |              |      |        |       |         | =R           | +γv | (S      | )   |     |       |
|           |              |      |        |       | t       |              | t+1 | t t+1   |     |     |       |
where R here is the estimate at time of . Our point now is that this idea makes just as much sense
| V   | t : S → |     |     |     |     | t v π |     |     |     |     |     |
| --- | ------- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- |
after two steps as it does after one. The target for a two-step update is the two-step return:
|             |             |          |           | Gt:t+2    | =R     | +γR          |            | +γ2v    | (S )     |        | (6.3) |
| ----------- | ----------- | -------- | --------- | --------- | ------ | ------------ | ---------- | ------- | -------- | ------ | ----- |
|             |             |          |           |           | t      | t+1          | t+2        |         | t t+2    |        |       |
| Similarly,  | the target  | for an   | arbitrary | n-step    | update | is           | the n-step | return: |          |        |       |
|             |             |          | Gt:t+n    |           |        | +···+γn−1R   |            |         | +γnv     |        | (6.4) |
|             |             |          |           | =R        | +γR    |              |            |         |          | (S )   |       |
|             |             |          | t         |           | t+1    | t+2          |            |         | t+n      | t t+n  |       |
| The natural | state-value | learning |           | algorithm | for    | using n-step |            | returns | is thus: |        |       |
|             |             | V        | (S        | )=V       | (S     | )+α[G        |            | −V      | (S )],   | 0≤t<T, | (6.5) |
|             |             |          | t+n t     | t+n−1     |        | t            | t:t+n      | t+n−1   | t        |        |       |
while the values of all other states remain unchanged: V (S)=V (s), for all s̸=S . We call this algorithm
|     |     |     |     |     |     |     | t+n |     | t+n−1 | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
n-stepTD.Notethatnochangesatallaremadeduringthefirstn−1stepsofeachepisode. Tomakeupforthat,
an equal number of additional updates are made at the end of the episode, after termination and before starting
the next episode.
The n-step return uses the value function to correct for the missing rewards beyond . An important
V t+n−1 R t+n
property of n-step returns is that their expectation is guaranteed to be a better estimate of v than V is, in
π t+n−1
a worst-state sense. That is, the worst error of the expected n-step return is guaranteed to be less than or equal
| to γn times | the worst | error | under |     | :   |     |     |     |     |     |     |
| ----------- | --------- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
V t+n−1
|              |         | max|E  |       | [G        | |S =s]−v |          | (s)|≤γnmax|V |           |          | (s)−v (s)|, | (6.6) |
| ------------ | ------- | ------ | ----- | --------- | -------- | -------- | ------------ | --------- | -------- | ----------- | ----- |
|              |         |        | π     | t:t+n     | t        | π        |              |           | t+n−1    | π           |       |
|              |         |        | s     |           |          |          |              | s         |          |             |       |
| for all n≥1. | This is | called | the   |           |          |          |              |           | returns. |             |       |
|              |         |        | error | reduction |          | property |              | of n-step |          |             |       |
38

| 6.2 | n-step | Sarsa |     |     |     |     |     |     |     |     |     |     |     |
| --- | ------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Howcan n-stepmethodsbeusednotjustforprediction, butforcontrol? Themainideais tosimplyswitch states
for actions (state-action pairs) and then use an ϵ-greedy policy. This is the n-step version of Sarsa and is called
Sarsa. We redefine n-step returns (update targets) in terms of estimated action values:
n-step
|      | G     | =R  | +γR    | +···+γn−1R |         |           | +γnQ(t+n−1)(S |     | ,A      | ),n≥1,0≤t<T |     | −n, | (6.7) |
| ---- | ----- | --- | ------ | ---------- | ------- | --------- | ------------- | --- | ------- | ----------- | --- | --- | ----- |
|      | t:t+n |     | t+1    | t+2        |         |           | t+n           |     | t+n t+n |             |     |     |       |
| with | G =G  | if  | t+n≥T. | The        | natural | algorithm | is then:      |     |         |             |     |     |       |
|      | t:t+n | t   |        |            |         |           |               |     |         |             |     |     |       |
(6.8)
|     |     | Q   | t+n (S | t ,A t )=Q | t+n−1 | (S t ,A | t )+α[G t:t+n −Q | t+n−1 | (S t ,A t )], | 0≤t<T, |     |     |     |
| --- | --- | --- | ------ | ---------- | ----- | ------- | ---------------- | ----- | ------------- | ------ | --- | --- | --- |
while the values of all other states remain unchanged: for all such that or
|     |     |     |     |     |     |     | Q (s,a) | = Q | (s,a), |     | s,a | s   | ̸= S |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | ------ | --- | --- | --- | ---- |
|     |     |     |     |     |     |     | t+n     |     | t+n−1  |     |     |     | t    |
a̸=A .
t
Figure 6.2: The n-step Sarsa algorithm. The n-step return is used to update the action value function. The policy is
| ϵ-greedy | with respect |            | to the | action value | function. |     |     |     |     |     |     |     |     |
| -------- | ------------ | ---------- | ------ | ------------ | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.3      | n-step       | Off-policy |        |              | Learning  |     |     |     |     |     |     |     |     |
In n-step methods, returns are constructed over n steps, so we are interested in the relative probability of just
those n actions. To make a simple off-policy version of n-step TD, the update for time t (actually made at time
| t+n) | can simply | be  | weighted | by  |     | :   |     |     |     |     |     |     |     |
| ---- | ---------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ρ
t:t+n−1
|     |     |     | V (S | )=V | (S    | )+αρ | [G            | −V    | (S )], | 0≤t<T, |     |     | (6.9) |
| --- | --- | --- | ---- | --- | ----- | ---- | ------------- | ----- | ------ | ------ | --- | --- | ----- |
|     |     |     | t+n  | t   | t+n−1 | t    | t:t+n−1 t:t+n | t+n−1 | t      |        |     |     |       |
where ρ , called the importance sampling ratio, is the relative probability under the two policies of taking
t:t+n−1
| the | actions | from | to  | :   |     |     |     |     |     |     |     |     |     |
| --- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n   |         | A    | A   |     |     |     |     |     |     |     |     |     |     |
t t+n−1
min(h,T−1)
|     |     |     |     |     |     |     | (cid:89) π(A | k |S k ) |     |     |     |     |        |
| --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --- | --- | --- | --- | ------ |
|     |     |     |     |     |     | ρ   | =            |          |     |     |     |     | (6.10) |
|     |     |     |     |     |     | t:h | b(A          | |S )     |     |     |     |     |        |
k k
k=t
39

Then-stepoff-policylearningalgorithm. Theimportancesamplingratioρ isusedtoweighttheupdate.
Figure6.3:
t:t+n−1
| The policy | is ϵ-greedy with | respect to | the action value | function. |
| ---------- | ---------------- | ---------- | ---------------- | --------- |
40

| Chapter  | 7   |     |          |      |         |     |     |
| -------- | --- | --- | -------- | ---- | ------- | --- | --- |
| Planning |     | and | Learning | with | Tabular |     |     |
Methods
Inthischapterwedevelopaunifiedviewofreinforcementlearningmethodsthatrequireamodeloftheenvironment,
such as dynamic programming and heuristic search, and methods that can be used without a model, such as
Monte Carlo and temporal-difference methods. These are respectively called model-based and model-free
reinforcementlearningmethods. Model-basedmethodsrelyonplanningastheirprimarycomponent,whilemodel-
free methods primarily rely on learning. Although there are real differences between these two kinds of methods,
| there are | also great similarities. |     | In particular, |               |            |                    |     |
| --------- | ------------------------ | --- | -------------- | ------------- | ---------- | ------------------ | --- |
|           |                          |     | the heart      | of both kinds | of methods | is the computation | of  |
value functions. Moreover, allthemethodsarebasedonlookingaheadtofutureevents, computingabacked-up
value, and then using it as an update target for an approximate value function.
| 7.1 Models |     | and Planning |     |     |     |     |     |
| ---------- | --- | ------------ | --- | --- | --- | --- | --- |
By a model of the environment we mean anything that an agent can use to predict how the environment will
respond to its actions. Given a state and an action, a model produces a prediction of the resultant next state
and next reward. If the model is stochastic, then there are several possible next states and next rewards, each
with some probability of occurring. Some models produce a description of all possibilities and their probabilities;
these we call models. Other models produce just one of the possibilities, sampled according to the
distribution
| probabilities; | these we | call sample | models. |     |     |     |     |
| -------------- | -------- | ----------- | ------- | --- | --- | --- | --- |
The kind of model assumed in dynamic programming—estimates of the MDP’s dynamics, p(s′,r|s,a)—is a dis-
tribution model. The kind of model used in the blackjack example in Chapter 5 is a sample model. Distribution
models are stronger than sample models in that they can always be used to produce samples. However, in many
applications it is much easier to obtain sample models than distribution models.
Models can be used to mimic or simulate experience. Given a starting state and action, a sample model produces
a possible transition, and a distribution model generates all possible transitions weighted by their probabilities of
occurring. Givenastartingstateandapolicy, asamplemodelcouldproduceanentireepisode, andadistribution
model could generate all possible episodes and their probabilities. In either case, we say the model is used to
| simulate | the environment | and produce | simulated experience. |     |     |     |     |
| -------- | --------------- | ----------- | --------------------- | --- | --- | --- | --- |
Thewordplanningisusedinseveraldifferentwaysindifferentfields. Weusethetermtorefertoanycomputational
processthattakesamodelasinputandproducesorimprovesapolicyforinteractingwiththemodeledenvironment:
|     |     |     | model → planning | → policy |     |     |     |
| --- | --- | --- | ---------------- | -------- | --- | --- | --- |
State-space planning, which includes the approach we take in this book, is viewed primarily as a search through
the state space for an optimal policy or an optimal path to a goal. Actions cause transitions from state to state,
and value functions are computed over states. In what we call plan-space planning, planning is instead a search
through the space of plans. Operators transform one plan into another, and value functions, if any, are defined
| over the | space of plans. |     |     |     |     |     |     |
| -------- | --------------- | --- | --- | --- | --- | --- | --- |
The unified view we present in this chapter is that all state-space planning methods share a common structure,
a structure that is also present in the learning methods presented in this book. It takes the rest of the chapter
to develop this view, but there are two basic ideas: (1) all state-space planning methods involve computing value
functionsasakeyintermediatesteptowardimprovingthepolicy,and(2)theycomputevaluefunctionsbyupdates
| or backup | operations | applied to | simulated experience. |     |     |     |     |
| --------- | ---------- | ---------- | --------------------- | --- | --- | --- | --- |
Dynamicprogrammingmethodsclearlyfitthisstructure: theymakesweepsthroughthespaceofstates,generating
foreachstatethedistributionofpossibletransitions. Eachdistributionisthenusedtocomputeabacked-upvalue
41

(update target) and update the state’s estimated value. Viewing planning methods in this way emphasizes their
relationshiptothelearningmethodsthatwehavedescribedinthisbook. Theheartofbothlearningandplanning
methods is the estimation of value functions by backing-up update operations. The difference is that whereas
planning uses simulated experience generated by a model, learning methods use real experience generated by the
environment.
7.2 Dyna: Integrated Planning, Acting, and Learning
When planning is done online, while interacting with the environment, a number of interesting issues arise. New
information gained from the interaction may change the model and thereby interact with planning. It may be
desirabletocustomizetheplanningprocessinsomewaytothestatesordecisionscurrentlyunderconsideration,or
expectedinthenearfuture. Ifdecisionmakingandmodellearningarebothcomputation-intensiveprocesses,then
the available computational resources may need to be divided between them. To begin exploring these issues, in
thissectionwepresentDyna-Q,asimplearchitectureintegratingthemajorfunctionsneededinanonlineplanning
agent.
Within a planning agent, there are at least two roles for real experience: it can be used to improve the model (to
make it more accurately match the real environment) and it can be used to directly improve the value function
and policy using the kinds of reinforcement learning methods we have discussed in previous chapters. The former
we call model-learning, and the latter we call direct reinforcement learning (direct RL).
Both direct and indirect methods have advantages and disadvantages. Indirect methods often make fuller use of
a limited amount of experience and thus achieve a better policy with fewer environmental interactions. On the
otherhand,directmethodsaremuchsimplerandarenotaffectedbybiasesinthedesignofthemodel. Somehave
argued that indirect methods are always superior to direct ones, while others have argued that direct methods are
responsible for most human and animal learning.
Dyna-Q includes all of the processes shown in the diagram above—planning, acting, model-learning, and direct
RL—all occurring continually. The planning method is the random-sample one-step tabular Q-planning method.
ThedirectRLmethodisone-steptabularQ-learning. Themodel-learningmethodisalsotable-basedandassumes
theenvironmentisdeterministic. AftereachtransitionS ,A →R ,S ,themodelrecordsinitstableentryfor
t t t+1 t+1
S ,A thepredictionthatR ,S willdeterministicallyfollow. Thus,ifthemodelisqueriedwithastate–action
t t t+1 t+1
pair that has been experienced before, it simply returns the last-observed next state and next reward as its
prediction.
During planning, the Q-planning algorithm randomly samples only from state–action pairs that have previously
been experienced (in Step 1), so the model is never queried with a pair about which it has no information.
42

|     |     |     | Figure 7.1: | The general | Dyna Architecture. |
| --- | --- | --- | ----------- | ----------- | ------------------ |
Conceptually,planning,acting,model-learning,anddirectRLoccursimultaneouslyandinparallelinDynaagents.
Forconcretenessandimplementationonaserialcomputer,however,wefullyspecifytheorderinwhichtheyoccur
within a time step. In Dyna-Q, the acting, model-learning, and direct RL processes require little computation,
and we assume they consume just a fraction of the time. The remaining time in each step can be devoted to the
| planning | process, which | is inherently | computation-intensive. |     |     |
| -------- | -------------- | ------------- | ---------------------- | --- | --- |
| 7.3      | When the       | Model         | Is Wrong               |     |     |
In the maze example presented in the previous section, the changes in the model were relatively modest. The
model started out empty, and was then filled only with exactly correct information. In general, we cannot expect
to be so fortunate. Models may be incorrect because the environment is stochastic and only a limited number of
samples have been observed, or because the model was learned using function approximation that has generalized
imperfectly, or simply because the environment has changed and its new behavior has not yet been observed.
When the model is incorrect, the planning process is likely to compute a suboptimal policy.
In some cases, the suboptimal policy computed by planning quickly leads to the discovery and correction of the
modeling error. This tends to happen when the model is optimistic in the sense of predicting greater reward or
betterstatetransitionsthanareactuallypossible. Theplannedpolicyattemptstoexploittheseopportunitiesand
| in doing | so discovers that | they do | not exist. |     |     |
| -------- | ----------------- | ------- | ---------- | --- | --- |
Greater difficulties arise when the environment changes to become better than it was before, and yet the formerly
correct policy does not reveal the improvement. In these cases the modeling error may not be detected for a long
43

time, if ever.
The general problem here is another version of the conflict between exploration and exploitation. In a planning
context, exploration means trying actions that improve the model, whereas exploitation means behaving in the
optimal way given the current model.
We want the agent to explore to find changes in the environment, but not so much that performance is greatly
degraded. Asintheearlierexploration/exploitationconflict,thereprobablyisnosolutionthatisbothperfectand
practical, but simple heuristics are often effective.
7.4 Prioritized Sweeping
In the Dyna agents presented in the preceding sections, simulated transitions are started in state–action pairs
selected uniformly at random from all previously experienced pairs. But a uniform selection is usually not the
best;planningcanbemuchmoreefficientifsimulatedtransitionsandupdatesarefocusedonparticularstate–action
pairs.
This example suggests that search might be usefully focused by working backward from goal states. Of course, we
donotreallywanttouseanymethodsspecifictotheideaof"goalstate."Wewantmethodsthatworkforgeneral
reward functions. Goal states are just a special case, convenient for stimulating intuition. In general, we want
to work back not just from goal states but from any state whose value has changed. Suppose that the values are
initially correct given the model. Suppose now that the agent discovers a change in the environment and changes
its estimated value of one state, either up or down. Typically, this will imply that the values of many other states
should also be changed, but the only useful one-step updates are those of actions that lead directly into the one
state whose value has been changed.
This general idea might be termed backward focusing of planning computations.
It is natural to prioritize the updates according to a measure of their urgency, and perform them in order of
priority. This is the idea behind prioritized sweeping. A queue is maintained of every state–action pair whose
estimated value would change nontrivially if updated, prioritized by the size of the change. When the top pair in
the queue is updated, the effect on each of its predecessor pairs is computed. If the effect is greater than some
small threshold, then the pair is inserted in the queue with the new priority. In this way the effects of changes are
efficiently propagated backward until quiescence.
Prioritized sweeping is just one way of distributing computations to improve planning efficiency, and probably
not the best way. One of prioritized sweeping’s limitations is that it uses expected updates, which in stochastic
environments may waste lots of computation on low-probability transitions.
We have suggested in this chapter that all kinds of state-space planning can be viewed as sequences of value
updates, varying only in the type of update, expected or sample, large or small, and in the order in which the
updates are done.
44

7.5 Expected vs. Sample Updates
Much of this book has been about different kinds of value-function updates, and we have considered a great many
varieties. Focusing for the moment on one-step updates, they vary primarily along three binary dimensions. The
first two dimensions are whether they update state values or action values and whether they estimate the value
for the optimal policy or for an arbitrary given policy. These two dimensions give rise to four classes of updates
forapproximatingthefourvaluefunctions, q , v , q , andv . Theotherbinarydimensioniswhethertheupdates
∗ ∗ π π
are expected updates, considering all possible events that might happen, or sample updates, considering a
single sample of what might happen.
Whenweintroducedone-stepsampleupdatesinChapter6,wepresentedthemassubstitutesforexpectedupdates.
In the absence of a distribution model, expected updates are not possible, but sample updates can be done using
sampletransitionsfromtheenvironmentorasamplemodel. Implicitinthatpointofviewisthatexpectedupdates,
if possible, are preferable to sample updates. But are they? Expected updates certainly yield a better estimate
because they are uncorrupted by sampling error, but they also require more computation, and computation is
often the limiting resource in planning.
Figure 7.2: Backup diagrams for all the one-step updates considered in this chapter.
For concreteness, consider the expected and sample updates for approximating q , and the special case of discrete
∗
states and actions, a table-lookup representation of the approximate value function, Q, and a model in the form
45

of estimated dynamics, pˆ(s′,r|s,a). The expected update for a state–action pair, s,a, is:
|     |     |     |     | (cid:88)     | (cid:104)      | (cid:105) |       |
| --- | --- | --- | --- | ------------ | -------------- | --------- | ----- |
|     |     |     |     | pˆ(s′,r|s,a) | r+γmaxQ(s′,a′) |           | (7.1) |
Q(s,a)←
a′
s′,r
The corresponding sample update for s,a, given a sample next state and reward, S′ and (from the model), is
R
| the Q-learning-like | update: |                 |     |                       |     |           |       |
| ------------------- | ------- | --------------- | --- | --------------------- | --- | --------- | ----- |
|                     |         |                 |     | (cid:104)             |     | (cid:105) |       |
|                     |         | Q(s,a)←Q(s,a)+α |     | R+γmaxQ(S′,a′)−Q(s,a) |     |           | (7.2) |
a′
If there is enough time to complete an expected update, then the resulting estimate is generally better than that
of b sample updates because of the absence of sampling error. But if there is insufficient time to complete an
expected update, then sample updates are always preferable because they at least make some improvement in the
| value estimate | with fewer | than | updates. |     |     |     |     |
| -------------- | ---------- | ---- | -------- | --- | --- | --- | --- |
b
|                |     |             | Comparison | of efficiency | of expected | and sample updates. |     |
| -------------- | --- | ----------- | ---------- | ------------- | ----------- | ------------------- | --- |
|                |     | Figure 7.3: |            |               |             |                     |     |
| 7.6 Trajectory |     | Sampling    |            |               |             |                     |     |
Inthissectionwecomparetwowaysofdistributingupdates. Theclassicalapproach, fromdynamicprogramming,
is to perform sweeps through the entire state (or state–action) space, updating each state (or state–action pair)
once per sweep. This is problematic on large tasks because there may not be time to complete even one sweep. In
many tasks the vast majority of the states are irrelevant because they are visited only under very poor policies or
| with very low | probability. |     |     |     |     |     |     |
| ------------- | ------------ | --- | --- | --- | --- | --- | --- |
The second approach is to sample from the state or state–action space according to some distribution. One could
sample uniformly, as in the Dyna-Q agent, but this would suffer from some of the same problems as exhaustive
sweeps. More appealing is to distribute updates according to the on-policy distribution, that is, according
to the distribution observed when following the current policy. One advantage of this distribution is that it is
easily generated; one simply interacts with the model, following the current policy. We call this way of generating
| experience | and updates |     | sampling. |     |     |     |     |
| ---------- | ----------- | --- | --------- | --- | --- | --- | --- |
trajectory
Focusing on the on-policy distribution could be beneficial because it causes vast, uninteresting parts of the space
to be ignored, or it could be detrimental because it causes the same old parts of the space to be updated over and
over.
| 7.7 Real-time |     | Dynamic | Programming |     |     |     |     |
| ------------- | --- | ------- | ----------- | --- | --- | --- | --- |
Real-time dynamic programming, or RTDP, is an on-policy trajectory-sampling version of the value-iteration
algorithmofdynamicprogramming(DP).Becauseitiscloselyrelatedtoconventionalsweep-basedpolicyiteration,
RTDPillustratesinaparticularlyclearwaysomeoftheadvantagesthaton-policytrajectorysamplingcanprovide.
RTDP updates the values of states visited in actual or simulated trajectories by means of expected tabular value-
| iteration updates. |     |     |     |     |     |     |     |
| ------------------ | --- | --- | --- | --- | --- | --- | --- |
46

If trajectories can start only from a designated set of start states, and if you are interested in the prediction
problem for a given policy, then on-policy trajectory sampling allows the algorithm to completely skip states that
cannot be reached by the given policy from any of the start states: such states are irrelevant to the prediction
problem. For a control problem, where the goal is to find an optimal policy instead of evaluating a given policy,
there might well be states that cannot be reached by any optimal policy from any of the start states, and there
is no need to specify optimal actions for these irrelevant states. What is needed is an optimal partial policy,
meaning a policy that is optimal for the relevant states but can specify arbitrary actions, or even be undefined,
for the irrelevant states.
ThemostinterestingresultforRTDPisthatforcertaintypesofproblemssatisfyingreasonableconditions,RTDP
is guaranteed to find a policy that is optimal on the relevant states without visiting every state infinitely often, or
even without visiting some states at all. Indeed, in some problems, only a small fraction of the states need to be
visited. This can be a great advantage for problems with very large state sets, where even a single sweep may not
be feasible.
7.8 Planning at Decision Time
Planning can be used in at least two ways. The one we have considered so far in this chapter, typified by dynamic
programmingandDyna,istouseplanningtograduallyimproveapolicyorvaluefunctiononthebasisofsimulated
experience obtained from a model. We call planning used in this way background planning.
The other way to use planning is to begin and complete it after encountering each new state S , as a computation
t
whose output is the selection of a single action A ; on the next step planning begins anew with S to produce
t t+1
A , and so on. Unlike the first use of planning, here planning focuses on a particular state. We call this
t+1
decision-time planning.
7.9 Heuristic Search
Theclassicalstate-spaceplanningmethodsinartificialintelligencearedecision-timeplanningmethodscollectively
known as heuristic search. In heuristic search, for each state encountered, a large tree of possible continuations
is considered. The approximate value function is applied to the leaf nodes and then backed up toward the current
state at the root. The backing up within the search tree is just the same as in the expected updates with maxes
discussed throughout this book. The backing up stops at the state–action nodes for the current state. Once the
backed-up values of these nodes are computed, the best of them is chosen as the current action, and then all
backed-up values are discarded.
In conventional heuristic search no effort is made to save the backed-up values by changing the approximate value
function. In fact, the value function is generally designed by people and never changed as a result of search.
We should not overlook the most obvious way in which heuristic search focuses updates: on the current state.
Muchoftheeffectivenessofheuristicsearchisduetoitssearchtreebeingtightlyfocusedonthestatesandactions
that might immediately follow the current state.
47

Heuristicsearchcanbeimplementedasasequenceofone-stepupdates(blulines)backingupvaluesfromthe
Figure 7.4:
leaf nodes toward the root. The ordering shown here is for a selective depth-first search.
| 7.10 | Rollout | Algorithms |     |     |     |     |     |
| ---- | ------- | ---------- | --- | --- | --- | --- | --- |
Rollout algorithms are decision-time planning algorithms based on Monte Carlo control applied to simulated
trajectories that all begin at the current environment state. They estimate action values for a given policy by
averagingthereturnsofmanysimulatedtrajectoriesthatstartwitheachpossibleactionandthenfollowthegiven
policy. When the action-value estimates are considered to be accurate enough, the action (or one of the actions)
havingthehighestestimatedvalueisexecuted, afterwhichtheprocessiscarriedoutanewfromtheresultingnext
state.
Whatthendorolloutalgorithmsaccomplish? Thepolicyimprovementtheoremtellsusthatgivenanytwopolicies
and π′ that are identical except that π′(s) for some state s, if (s), then policy π′ is
| π   |     |     |     | =   | a ̸= π(s) |     | q (s,a) ≥ v |
| --- | --- | --- | --- | --- | --------- | --- | ----------- |
π π
as good as, or better, than π. This applies to rollout algorithms where s is the current state and π is the rollout
policy.
In other words, the aim of a rollout algorithm is to improve upon the rollout policy; not to find an optimal policy.
| Experience | has shown | that  | rollout algorithms | can    | be surprisingly | effective. |     |
| ---------- | --------- | ----- | ------------------ | ------ | --------------- | ---------- | --- |
| 7.11       | Monte     | Carlo | Tree               | Search |                 |            |     |
Monte Carlo Tree Search (MCTS) is a recent and strikingly successful example of decision-time planning. At its
base, MCTS is a rollout algorithm as described above, but enhanced by the addition of a means for accumulating
valueestimatesobtainedfromtheMonteCarlosimulationsinordertosuccessivelydirectsimulationstowardmore
highly-rewarding trajectories. MCTS is largely responsible for the improvement in computer Go from a weak
| amateur | level in | 2005 to a | grandmaster | level (6 | dan or more) | in 2015. |     |
| ------- | -------- | --------- | ----------- | -------- | ------------ | -------- | --- |
MCTS is executed after encountering each new state to select the agent’s action for that state; it is executed
again to select the action for the next state, and so on. The core idea of MCTS is to successively focus multiple
simulations starting at the current state by extending the initial portions of trajectories that have received high
| evaluations | from | earlier simulations. |     |     |     |     |     |
| ----------- | ---- | -------------------- | --- | --- | --- | --- | --- |
For the most part, the actions in the simulated trajectories are generated using a simple policy, usually called
a rollout policy as it is for simpler rollout algorithms. Monte Carlo value estimates are maintained only for the
subset of state–action pairs that are most likely to be reached in a few steps, which form a tree rooted at the
currentstate. MCTSincrementallyextendsthetreebyaddingnodesrepresentingstatesthatlookpromisingbased
| on the results | of  | the simulated | trajectories. |     |     |     |     |
| -------------- | --- | ------------- | ------------- | --- | --- | --- | --- |
48

Figure 7.5: MonteCarloTreeSearch. Whentheenvironmentchangestoanewstate,MCTSexecutesasmanyiterations
aspossiblebeforeanactionneedstobeselected,incrementallybuildingatreewhoserootnoderepresentsthecurrentstate.
Each iteration consists of the four operations Selection, Expansion, Simulation and Backup.
Each iteration of a basica version of MCTS consists of the following four steps:
1. Selection. Starting at the root node, a tree policy base on the action values attached to the edges of the tree
traverses the tree to select a leaf node.
2. Expansion. On some iterations, the tree is expanded from the selected leaf node by adding one or ore child
nodes reached from the selected node via unexplored actions.
3. Simulation. From the selected node or from one of its newly-added child nodes, simulation of a complete
episode is run with actions selected by the rolloud policy. The result is a MC trial with actions selected first
by the tree policy and beyond the tree by the rollout policy.
4. Backup. The return generated by the simulated episode is backed up to update or to initialize, the action
values attached to the edges of the tree traversed by the tree policy in this iteration of MCTS. No value are
saved for the states and actions visited by the rollout policy beyond the tree.
49

| Chapter     | 8   |     |          |     |         |
| ----------- | --- | --- | -------- | --- | ------- |
| Approximate |     |     | Solution |     | Methods |
The novelty in this chapter is that the approximate value function is represented not as a table but as a param-
eterized functional form with weight vector w ∈ R. We will write vˆ(s,w) ≈ v (s) for the approximate value of
π
| state s given | weight vector | w.  |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
Typically, the number of weights is much less than the number of states, consequently, when a single state is
updated,thechangegeneralizesfromthatstatetoaffectthevaluesofmanyotherstates. Suchgeneralizationmakes
the learning potentially more powerful byt also potentially more difficult to manage and understand. Extending
RL to function approximation also makes it applicable to partially observable problems, in which the full state is
| not available | to the agent. |     |     |     |     |
| ------------- | ------------- | --- | --- | --- | --- |
What function approximation can’t do, however, is augment the state representation with memories of past
observations.
| 8.1 Value-function |     |     | Approximation |     |     |
| ------------------ | --- | --- | ------------- | --- | --- |
All of the prediction methods covered in this book have been described as updates to an estimated value function
that shifts its value at particular states toward a "backed-up value", or target, for that state. Let us refer
update
to an individual update by the notation s → u, where s is the state updated and u is the update target that s’s
| estimated | value is shifted | toward. |     |     |     |
| --------- | ---------------- | ------- | --- | --- | --- |
It is natural to interpret each update as specifying an example of the desired input-output behavior of the value
function. In a sense, the update means that the estimated value for state should be more like the update
s→u s
target u. Up to now, the actual update has been trivial: the table entry for s’s estimated value has simply been
shifted a fraction of the way toward u, and the estimated values of all other states were left unchanged. Now we
permit arbitrarily complex and sophisticated methods to implement the update, and updating at s generalizer so
| that the | estimated values | of many | other states | are changed | as well. |
| -------- | ---------------- | ------- | ------------ | ----------- | -------- |
Function approximation methods expect to receive examples of the desired input-output behavior of the function
they are trying to approximate. We use these methods for value prediction simply by passing to them the
s→u
of each update as a training example. We then interpret the approximate function they produce as an estimated
valuefunction. InRL,however,itisimoirtantthatlearningbeabletooccuronline,whiletheagentinteractswith
its environment or with a model of its environment. To do this requires methods that are able to learn efficiently
from incrementally acquired data. In addition, RL generally requires function approximation methods able to
| handle nonstationary | target     | functions. |           |       |     |
| -------------------- | ---------- | ---------- | --------- | ----- | --- |
| 8.2 The              | Prediction |            | Objective | (V E) |     |
Up to now we have not specified an explicit objective for prediction. In the tabular case a continuous measure of
predictionqualitywasnotnecessarybecausethelearnedvaluefunctioncouldcometoequalthetruevaluefunction
exactly. Moreover,thelearnedvaluesateachstateweredecoupled—anupdateatonestateaffectednoother. But
with genuine approximation, an update at one state affects many others, and it is not possible to get the values
of all states exactly correct. By assumption we have far more states than weights, so making one state’s estimate
more accurate invariably means making others’ less accurate. We are obligated then to say which states we care
(cid:80)
most about. We must specify a state distribution µ(s) ≥ 0, µ(s) = 1, representing how much we care about
s
the error in each state s. By the error in a state s we mean the square of the difference between the approximate
value and the true value (s). Weighting this over the state space by µ, we obtain a natural objective
| vˆ(s,w) |     |     | v   |     |     |
| ------- | --- | --- | --- | --- | --- |
π
| function, | the mean square | value | error, denoted | VE: |     |
| --------- | --------------- | ----- | -------------- | --- | --- |
50

. (cid:88)
|     |     |     |     | VE(w)= |     | µ(s)[v |     | (s)−vˆ(s,w)]2 |     |     |     |     |
| --- | --- | --- | --- | ------ | --- | ------ | --- | ------------- | --- | --- | --- | --- |
π
s∈S
The square root of this measure, the root VE, gives a rough measure of how much the approximate values differ
fromthetruevaluesandisoftenusedinplots. Oftenµ(s)ischosentobethefractionoftimespentins. Underon-
policytrainingthisiscalledtheon-policydistribution; wefocusentirelyonthiscaseinthischapter. Incontinuing
| tasks, the | on-policy           | distribution |     | is the stationary |     | distribution  |     | under | π.      |     |     |     |
| ---------- | ------------------- | ------------ | --- | ----------------- | --- | ------------- | --- | ----- | ------- | --- | --- | --- |
| 8.3        | Stochastic-gradient |              |     |                   | and | Semi-gradient |     |       | Methods |     |     |     |
Wenowdevelopindetailoneclassoflearningmethodsforfunctionapproximationinvalueprediction,thosebased
onstochasticgradientdescent(SGD).SGDmethodsareamongthemostwidelyusedofallfunctionapproximation
| methods | and are | particularly | well | suited | to online | reinforcement |     |     | learning. |     |     |     |
| ------- | ------- | ------------ | ---- | ------ | --------- | ------------- | --- | --- | --------- | --- | --- | --- |
Ingradient-descentmethods, theweightvectorisacolumnvectorwithafixednumberofrealvaluedcomponents,
. )T, and the approximate value function is a differentiable function of for all S.
| w = (w 1 | ,w 2 ,...,w | d   |     |     |     |     |     | vˆ(s,w) |     |     | w   | s ∈ |
| -------- | ----------- | --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
We will be updating w at each of a series of discrete time steps, t=0,1,2,3,..., so we will need a notation w for
t
the weight vector at each step. For now, let us assume that, on each step, we observe a new example S →v (S )
t π t
consisting of a (possibly randomly selected) state and its true value under the policy. These states might
S
t
be successive states from an interaction with the environment, but for now we do not assume so. Even though
we are given the exact, correct values, for each , there is still a difficult problem because our function
|     |     |     |     |     | v π (S t ) |     | S t |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
approximator has limited resources and thus limited resolution. In particular, there is generally no w that gets all
the states, or even all the examples, exactly correct. In addition, we must generalize to all the other states that
| have not | appeared | in examples. |     |     |     |     |     |     |     |     |     |     |
| -------- | -------- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
We assume that states appear in examples with the same distribution, µ, over which we are trying to minimize
the VE as given by (9.1). A good strategy in this case is to try to minimize error on the observed examples.
Stochastic gradient-descent (SGD) methods do this by adjusting the weight vector after each example by a small
amount in the direction that would most reduce the error on that example:
1
|     |     |     |     |     | .    |           |     |            |     | )]2 |     |     |
| --- | --- | --- | --- | --- | ---- | --------- | --- | ---------- | --- | --- | --- | --- |
|     |     |     |     | w   | =w   | − α∇[v    | (S  | )−vˆ(S     | ,w  |     |     |     |
|     |     |     |     | t+1 |      | t 2       | π   | t          | t t |     |     |     |
|     |     |     |     | =w  | +α[v | (S )−vˆ(S |     | ,w )]∇vˆ(S | ,w  | )   |     |     |
|     |     |     |     |     | t    | π t       |     | t t        | t   | t   |     |     |
where α is a positive step-size parameter, and ∇f(w), for any scalar expression f(w) that is a function of a vector
(here w), denotes the column vector of partial derivatives of the expression with respect to the components of the
vector:
|     |     |     |     |     |     | (cid:18) ∂f(w) | ∂f(w) |     | ∂f(w) | (cid:19)T |     |     |
| --- | --- | --- | --- | --- | --- | -------------- | ----- | --- | ----- | --------- | --- | --- |
.
|     |     |     |     | ∇f(w)= |     |     | ,   | ,..., |     |     |     |     |
| --- | --- | --- | --- | ------ | --- | --- | --- | ----- | --- | --- | --- | --- |
|     |     |     |     |        |     | ∂w  | ∂w  |       | ∂w  |     |     |     |
|     |     |     |     |        |     | 1   |     | 2     | d   |     |     |     |
Thisderivativevectoristhegradientoff withrespecttow. SGDmethodsare"gradientdescent"methodsbecause
the overall step in w is proportional to the negative gradient of the example’s squared error (9.4). This is the
t
direction in which the error falls most rapidly. Gradient descent methods are called "stochastic" when the update
is done, as here, on only a single example, which might have been selected stochastically. Over many examples,
making small steps, the overall effect is to minimize an average performance measure such as the VE.
We turn now to the case in which the target output, here denoted ∈R, of the tth training example, ,
|     |     |     |     |     |     |     |     |     | U t |     |     | S t →U t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- |
is not the true value, v (S ), but some, possibly random, approximation to it. For example, U might be a
|     |     | π   | t   |     |     |     |     |     |     |     | t   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
noise-corrupted version of v (S ), or it might be one of the bootstrapping targets using vˆ mentioned in the
|     |     |     | π   | t   |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
previous section. In these cases we cannot perform the exact update (9.5) because is unknown, but we can
|     |     |     |     |     |     |     |     |     |     | v π (S t ) |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- |
approximateitbysubstitutingU inplaceofv (S ). ThisyieldsthefollowinggeneralSGDmethodforstate-value
|     |     |     |     | t   |     | π t |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
prediction:
.
|     |     |     |     | w   | =w  | +α[U | −vˆ(S | ,w )]∇vˆ(S | ,w  | )   |     |     |
| --- | --- | --- | --- | --- | --- | ---- | ----- | ---------- | --- | --- | --- | --- |
|     |     |     |     | t+1 |     | t    | t     | t t        | t   | t   |     |     |
If U is an unbiased estimate, that is, if E[U |S = s] = v (s), for each t, then w is guaranteed to converge to a
| t   |     |     |     |     | t   | t   | π   |     |     | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
local optimum under the usual stochastic approximation conditions (2.7) for decreasing α.
One does not obtain the same guarantees if a bootstrapping estimate of v (S ) is used as the target U in (9.7).
|     |     |     |     |     |     |     |     |     |     | π t |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Bootstrapping targets such as n-step returns or the DP target (cid:80) all
|     |     |     |     |     |     | G     |     |     |        | π(a|S )p(s′,r|S | ,a)[r+vˆ(s′,w | )]  |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | ------ | --------------- | ------------- | --- |
|     |     |     |     |     |     | t:t+n |     |     | a,s′,r | t               | t             | t   |
depend on the current value of the weight vector w , which implies that they will be biased and that they will not
t
produce a true gradient-descent method. One way to look at this is that the key step from (9.4) to (9.5) relies on
51

the target being independent of w . This step would not be valid if a bootstrapping estimate were used in place
t
of v (S ). Bootstrapping methods are not in fact instances of true gradient descent (Barnard, 1993). They take
π t
into account the effect of changing the weight vector w on the estimate, but ignore its effect on the target. They
t
include only a part of the gradient and, accordingly, we call them semi-gradient methods.
Although semi-gradient (bootstrapping) methods do not converge as robustly as gradient methods, they do con-
verge reliably in important cases such as the linear case discussed in the next section. Moreover, they offer
important advantages that make them often clearly preferred. One reason for this is that they typically enable
significantly faster learning, as we have seen in Chapters 6 and 7. Another is that they enable learning to be
continual and online, without waiting for the end of an episode. This enables them to be used on continuing
problems and provides computational advantages. A prototypical semi-gradient method is semi-gradient TD(0),
.
which uses U =R +vˆ(S ,w) as its target.
t t+1 t+1
8.3.1 Stochastic Approximation and Mean-Square Convergence
Manyreinforcementlearningalgorithmscanbeinterpretedasinstancesof stochastic approximation(SA),that
is, iterative methods for finding the root of an unknown function when only noisy observations are available. In
its classical Robbins–Monro form, stochastic approximation considers the problem of solving
f(x)=0,
when the learner does not observe f(x ) exactly, but only a noisy quantity
n
Y =f(x )+M ,
n n n+1
where M is a random noise term with zero conditional mean. The generic update is then
n+1
x =x −α Y ,
n+1 n n n
where α >0 is the step size.
n
A particularly important case is the affine one, in which
f(x)=Lx−b,
with L∈Rd×d and b∈Rd. If L is invertible, then the equation f(x)=0 has the unique solution
x∗ =L−1b.
In this case, the noisy observation can be written as
Y =Lx −b+M =L(x −x∗)+M ,
n n n+1 n n+1
and the stochastic approximation update becomes
x =x −α (cid:0) L(x −x∗)+M (cid:1) .
n+1 n n n n+1
To study convergence, it is convenient to introduce the error
e =x −x∗.
n n
Subtracting x∗ from both sides of the update yields the error recursion
e =(I−α L)e −α M .
n+1 n n n n+1
This expression makes clear the two forces acting on the iterate:
• the deterministic term (I−α L)e , which contracts the error toward zero;
n n
• the stochastic term −α M , which perturbs the dynamics.
n n+1
We now state standard assumptions ensuring convergence in mean square:
1. Positive definiteness: the matrix L is strictly positive definite, with minimum eigenvalue λ >0;
min
2. Zero-mean noise:
E[M |F ]=0;
n+1 n
3. Bounded conditional second moment:
E[∥M ∥2 |F ]≤σ2;
n+1 n
4. Robbins–Monro step sizes:
∞ ∞
(cid:88) (cid:88)
α =∞, α2 <∞.
n n
n=0 n=0
Under these assumptions, the stochastic approximation iterates converge to the true root in mean square.
52

|     | Under | the assumptions |     |     | above, |     |     |     |     |     |
| --- | ----- | --------------- | --- | --- | ------ | --- | --- | --- | --- | --- |
Theorem.
|     |     |     |     |     |     | E(cid:2) | −x∗∥2(cid:3) |     |     |     |
| --- | --- | --- | --- | --- | --- | -------- | ------------ | --- | --- | --- |
|     |     |     |     |     |     | lim      | ∥x           | =0. |     |     |
n
n→∞
|     | Starting |     | from |     |     |     |     |     |     |     |
| --- | -------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
Proof sketch.
|     |     |     |     |     | e   | =(I−α | L)e | −α  | M , |     |
| --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- |
|     |     |     |     |     | n+1 |       | n   | n n | n+1 |     |
we square both sides and take the conditional expectation with respect to . Since is -measurable and the
|           |                  |     |          |           |         |           |     |               | F n   | e n F n |
| --------- | ---------------- | --- | -------- | --------- | ------- | --------- | --- | ------------- | ----- | ------- |
| noise has | zero conditional |     | mean,    | the cross | term    | vanishes: |     |               |       |         |
|           |                  |     | E(cid:2) |           | (cid:3) |           |     | ∥2+α2E(cid:2) |       | (cid:3) |
|           |                  |     | ∥e       | ∥2        | |F      | =∥(I−α    | L)e |               | ∥M ∥2 | |F .    |
|           |                  |     |          | n+1       | n       |           | n n |               | n n+1 | n       |
Using the positive definiteness of L, for sufficiently large we obtain the bound
n
|     |     |     |     |     | ∥(I−α | L)e | ∥2 ≤(1−α | λ     | )∥e ∥2, |     |
| --- | --- | --- | --- | --- | ----- | --- | -------- | ----- | ------- | --- |
|     |     |     |     |     |       | n n |          | n min | n       |     |
hence
|     |     |     |     | E(cid:2) | ∥2  | (cid:3) |         |           | ∥2+σ2α2. |     |
| --- | --- | --- | --- | -------- | --- | ------- | ------- | --------- | -------- | --- |
|     |     |     |     | ∥e       | n+1 | |F n    | ≤(1−α n | λ min )∥e | n        |     |
n
| Taking | total expectation     |     | and defining |     |     |       |       |           |     |     |
| ------ | --------------------- | --- | ------------ | --- | --- | ----- | ----- | --------- | --- | --- |
|        |                       |     |              |     |     |       | =E[∥e | ∥2],      |     |     |
|        |                       |     |              |     |     | v     | n n   |           |     |     |
| we get | the scalar recurrence |     |              |     |     |       |       |           |     |     |
|        |                       |     |              |     | v   | ≤(1−α | λ     | )v +σ2α2. |     |     |
|        |                       |     |              |     | n+1 |       | n min | n         | n   |     |
Rearranging,
|         |              |     |     |     | v   | −v  | ≤−λ | α v +σ2α2. |     |     |
| ------- | ------------ | --- | --- | --- | --- | --- | --- | ---------- | --- | --- |
|         |              |     |     |     | n+1 | n   | min | n n        | n   |     |
| Summing | over n gives |     |     |     |     |     |     |            |     |     |
∞
(cid:88)
|     |     |     |     |     |     | λ   | α v | <∞. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     |     |     | min | n   | n   |     |     |
n=N0
Because (cid:80) while (cid:80) α2 ∞, the sequence must be driven arbitrarily close to zero, and the total
|          | α n = ∞            |     |         | <   |             |            | v n       |     |     |     |
| -------- | ------------------ | --- | ------- | --- | ----------- | ---------- | --------- | --- | --- | --- |
|          | n                  |     | n       | n   |             |            |           |     |     |     |
| possible | upward fluctuation |     | becomes |     | negligible. | Therefore, |           |     |     |     |
|          |                    |     |         |     |             |            | lim v =0, |     |     |     |
n
n→∞
that is,
|     |     |     |     |     |     | E[∥x | −x∗∥2]=0. |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --------- | --- | --- | --- |
|     |     |     |     |     |     | lim  | n         |     |     |     |
n→∞
This result is important in reinforcement learning because many update rules, especially in temporal-difference
learning and approximate methods, have the general form of a stochastic approximation recursion. Stochastic
approximation therefore provides the mathematical framework for analyzing the convergence of noisy iterative
learning updates.
| 8.4 | Linear | Methods |     |     |     |     |     |     |     |     |
| --- | ------ | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
One of the most important special cases of function approximation is that in which the approximate function,
vˆ(·,w), is a linear function of the weight vector, w. Corresponding to every state s, there is a real-valued vector
.
x(s) = (x (s),x (s),...,x (s))T, with the same number of components as w. Linear methods approximate the
|             | 1 2      |        | d     |         |         |     |           |     |     |     |
| ----------- | -------- | ------ | ----- | ------- | ------- | --- | --------- | --- | --- | --- |
| state-value | function | by the | inner | product | between |     | and x(s): |     |     |     |
w
d
|     |     |     |     |     |                 | .   |     | . (cid:88) |       |     |
| --- | --- | --- | --- | --- | --------------- | --- | --- | ---------- | ----- | --- |
|     |     |     |     |     | vˆ(s,w)=wTx(s)= |     |     | w          | x (s) |     |
i i
i=1
In this case the approximate value function is said to be linear in the weights, or simply linear.
The vector x(s) is called a feature vector representing state s. Each component x (s) of x(s) is the value of a
i
function R. We think of a feature as the entirety of one of these functions, and we call its value for a
|     | x i : S → |     |     |     |     |     |     |     |     |     |
| --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
state s a feature of s. For linear methods, features are basis functions because they form a linear basis for the set
of approximate functions. Constructing d-dimensional feature vectors to represent states is the same as selecting
a set of basis functions. Features may be defined in many different ways; we cover a few possibilities in the next
d
sections.
It is natural to use SGD updates with linear function approximation. The gradient of the approximate value
| function | with respect | to  | in this | case | is  |     |     |     |     |     |
| -------- | ------------ | --- | ------- | ---- | --- | --- | --- | --- | --- | --- |
w
53

∇vˆ(s,w)=x(s)
Thus, in the linear case the general SGD update (9.7) reduces to a particularly simple form:
.
|     |     |     |     |     | w   | t+1 =w | t +α[U t −vˆ(S | t ,w t )]x(S | t ) |
| --- | --- | --- | --- | --- | --- | ------ | -------------- | ------------ | --- |
Because it is so simple, the linear SGD case is one of the most favorable for mathematical analysis. Almost
all useful convergence results for learning systems of all kinds are for linear (or simpler) function approximation
methods.
Inparticular,inthelinearcasethereisonlyoneoptimum(or,indegeneratecases,onesetofequallygoodoptima),
and thus any method that is guaranteed to converge to or near a local optimum is automatically guaranteed to
converge to or near the global optimum. For example, the gradient Monte Carlo algorithm presented in the
previous section converges to the global optimum of the under linear function approximation if is reduced
VE α
| over time | according | to  | the usual | conditions. |     |     |     |     |     |
| --------- | --------- | --- | --------- | ----------- | --- | --- | --- | --- | --- |
The semi-gradient TD(0) algorithm presented in the previous section also converges under linear function approx-
imation, but this does not follow from general results on SGD; a separate theorem is necessary. The weight vector
converged to is also not the global optimum, but rather a point near the local optimum. It is useful to consider
this important case in more detail, specifically for the continuing case. The update at each time t is
.
|     |     |     |     |     |       |           | +wTx      | −wTx |        |
| --- | --- | --- | --- | --- | ----- | --------- | --------- | ---- | ------ |
|     |     |     |     |     | w t+1 | =w t +α(R | t+1       | t+1  | t )x t |
|     |     |     |     |     |       |           | t         |      | t      |
|     |     |     |     |     | =w    | +α(R      | x −x (x   | −x   | )Tw )  |
|     |     |     |     |     |       | t         | t+1 t t t | t+1  | t      |
where here we have used the notational shorthand x =x(S ). Once the system has reached steady state, for any
|         |              |     |             |     |        |        | t t     |     |     |
| ------- | ------------ | --- | ----------- | --- | ------ | ------ | ------- | --- | --- |
| given , | the expected |     | next weight |     | vector | can be | written |     |     |
w
t
|     |     |     |     |     |     | E[w | |w ]=w +α(b−Aw |     | )   |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- |
|     |     |     |     |     |     | t+1 | t t            |     | t   |
where
|     |     |     | b=E[R | .   | ]∈Rd    |     | and A=E[x . |            | )T]∈Rd×d |
| --- | --- | --- | ----- | --- | ------- | --- | ----------- | ---------- | -------- |
|     |     |     |       |     | t+1 x t |     |             | t (x t −γx | t+1      |
From (9.10) it is clear that, if the system converges, it must converge to the weight vector at which
w
TD
|     |     |     |     |     |     |     | b−Aw =0 |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
TD
⇒b=Aw
TD
.
|     |     |     |     |     |     |     | ⇒w =A−1b |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | --- |
TD
This quantity is called the TD fixed point. In fact linear semi-gradient TD(0) converges to this point. Some of
the theory proving its convergence, and the existence of the inverse above, is given in the box.
AttheTDfixedpoint,ithasalsobeenproven(inthecontinuingcase)thattheVE iswithinaboundedexpansion
| of the lowest | possible |     | error: |     |     |     |     |     |     |
| ------------- | -------- | --- | ------ | --- | --- | --- | --- | --- | --- |
1
|     |     |     |     |     |     | VE(w | )≤ minVE(w) |     |     |
| --- | --- | --- | --- | --- | --- | ---- | ----------- | --- | --- |
TD
|     |     |     |     |     |     |     | 1−γ w |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
That is, the asymptotic error of the TD method is no more than 1 times the smallest possible error, that
1−γ
attained in the limit by the Monte Carlo method. Because γ is often near one, this expansion factor can be quite
large, so there is substantial potential loss in asymptotic performance with the TD method. On the other hand,
recall that the TD methods are often of vastly reduced variance compared to Monte Carlo methods, and thus
faster, as we saw in Chapters 6 and 7. Which method will be best depends on the nature of the approximation
| and problem, | and | on how | long | learning | continues. |     |     |     |     |
| ------------ | --- | ------ | ---- | -------- | ---------- | --- | --- | --- | --- |
54

| 8.5 Feature | Construction |     | for | Linear | Methods |     |     |     |     |
| ----------- | ------------ | --- | --- | ------ | ------- | --- | --- | --- | --- |
Linear methods are interesting because of their convergence guarantees, but also because in practice they can be
veryefficientintermsofbothdataandcomputation. Whetherornotthisissodependscriticallyonhowthestates
are represented in terms of features, which we investigate in this large section. Choosing features appropriate to
the task is an important way of adding prior domain knowledge to reinforcement learning systems.
Intuitively, the features should correspond to the aspects of the state space along which generalization may be
appropriate. If we are valuing geometric objects, for example, we might want to have features for each possible
shape, color, size, or function. If we are valuing states of a mobile robot, then we might want to have features for
locations, degrees of remaining battery power, recent sonar readings, and so on.
A limitation of the linear form is that it cannot take into account any interactions between features, such as
the presence of feature being good only in the absence of feature j. For example, in the pole-balancing task
i
(Example 3.4), high angular velocity can be either good or bad depending on the angle. If the angle is high, then
high angular velocity means an imminent danger of falling—a bad state—whereas if the angle is low, then high
angular velocity means the pole is righting itself—a good state. A linear value function could not represent this
if its features coded separately for the angle and the angular velocity. It needs instead, or in addition, features
for combinations of these two underlying state dimensions. In the following subsections we consider a variety of
| general ways | of doing this. |     |     |     |     |     |     |     |     |
| ------------ | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
• You can represent a state with dimensions, so that )T, maybe with
| Polynomials. |     |     |     | n   |     |     | x(s) = (s | 1 ,s 2 ,...,s n |     |
| ------------ | --- | --- | --- | --- | --- | --- | --------- | --------------- | --- |
interactionsorappendingaconstantvalue(1)atthebeginningtorepresentaffinefunctionsintheoriginalstate
| numbers. | Thus, each | order-n polynomial-basis |     | feature | x can | be written | as  |     |     |
| -------- | ---------- | ------------------------ | --- | ------- | ----- | ---------- | --- | --- | --- |
i
n
(cid:89)
|     |     |     |     | x (s)= | sci,j |     |     |     | (8.1) |
| --- | --- | --- | --- | ------ | ----- | --- | --- | --- | ----- |
|     |     |     |     | i      | j     |     |     |     |       |
j=1
| where | is an integer | in the set  |     | for an integer | n≥0. |     |     |     |     |
| ----- | ------------- | ----------- | --- | -------------- | ---- | --- | --- | --- | --- |
| c     |               | {0,1,...,n} |     |                |      |     |     |     |     |
i,j
• The Fourier basis represents functions as a sum of sinusoids. For a state s, we can define
| Fourier          | Basis. |     |     |                 |     |     |     |     |       |
| ---------------- | ------ | --- | --- | --------------- | --- | --- | --- | --- | ----- |
| Fourier features | as     |     |     |                 |     |     |     |     |       |
|                  |        |     |     | x (s)=cos(ωTs+ϕ |     | )   |     |     | (8.2) |
|                  |        |     |     | i               | i   | i   |     |     |       |
where ω is a frequency vector and ϕ is a phase shift. This allows for capturing periodic patterns in the state
| i   |     |     | i   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
space.
• Radial Basis Functions (RBFs). RBFs are a popular choice for function approximation, defined as
|     |     |     |             | (cid:18) | 1      |     | (cid:19) |     |       |
| --- | --- | --- | ----------- | -------- | ------ | --- | -------- | --- | ----- |
|     |     |     |             |          |        | ∥2  |          |     | (8.3) |
|     |     |     | x i (s)=exp |          | − ∥s−µ | i   |          |     |       |
2σ2
whereµ isthecenteroftheRBFandσ controlsthewidth. RBFsareparticularlyusefulforcapturinglocalized
i
| features | in the state space. |     |     |     |     |     |     |     |     |
| -------- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
• ANN. Artificial Neural Networks (ANNs) can be used to learn complex feature representations from raw
state inputs. They consist of multiple layers of interconnected nodes, where each node applies a non-linear
transformation to its inputs. The learned features can capture intricate patterns and interactions in the state
space.
55

| Chapter |     |          | 9   |     |         |     |
| ------- | --- | -------- | --- | --- | ------- | --- |
| Policy  |     | Gradient |     |     | Methods |     |
So far in this book almost all the methods have been methods; they learned the values of actions and
action-value
then selected actions based on their estimated action values. In this chapter we consider methods that instead
learn a parameterized policy that can select actions without consulting a value function. A value function may
still be used to the policy parameter, but it is not required for action selection. We will use the notation
learn
| θ ∈Rd′ | for the | policy’s | parameter | vector. |     |     |
| ------ | ------- | -------- | --------- | ------- | --- | --- |
In this chapter we consider methods for learning the policy parameter based on the gradient of some scalar
performancemeasureJ(θ)withrespecttothepolicyparameter. Thesemethodsseektomaximize theperformance,
| so their | updates | approximate |     | gradient | ascent in J: |             |
| -------- | ------- | ----------- | --- | -------- | ------------ | ----------- |
|          |         |             |     |          | θ =θ         | +α∇J(θ ˆ ), |
|          |         |             |     |          | t+1          | t t         |
All methods that follow this general schema are called methods.
policy gradient
| 9.1 | Policy |     | Approximation |     | and its | Advantages |
| --- | ------ | --- | ------------- | --- | ------- | ---------- |
In policy gradient methods, the policy can be parameterized in any way, as long as is differentiable with
π(a|s,θ)
∈Rd′.
respect to its parameters, i.e., as long as ∇π(a|s,θ) exists and is finite for all s∈S,a∈A(s),θ
If the action space is discrete and not too large, then a natural and common kind of parameterization is to form
parameterized numerical preferences h(s,a,θ)∈R for each state-action pair. The actions with the highest prefer-
ences in each state are given the highest probabilities of being selected, for example, according to an exponential
| soft-max | distribution: |     |     |     |     |     |
| -------- | ------------- | --- | --- | --- | --- | --- |
eh(s,a,θ)
π(a|s,θ)=
(cid:80)
eh(s,a′,θ)
a′
The action preferences themselves can be parameterized arbitrarily. As an example, they might be computed by
a deep artificial neural network (ANN), where θ is the vector of all the connection weights of the network. Or the
| preferences | could | be  | simply | linear in features: |     |     |
| ----------- | ----- | --- | ------ | ------------------- | --- | --- |
h(s,a,θ)=θ⊤ϕ(s,a)
| 9.2 | Policy |     | Gradient | Theorem |     |     |
| --- | ------ | --- | -------- | ------- | --- | --- |
We can simplify the notation without losing any meaningful generality by assuming that every episode starts in
some particular (non-random) state s . Then, in the episodic case we define performance as:
0
|     |     |     |     |     | J(θ)=v | πθ (s 0 ) |
| --- | --- | --- | --- | --- | ------ | --------- |
where is the true value function for , the policy determined by θ. From here on in our discussion we will
| v      |                |     |     |             | π              |     |
| ------ | -------------- | --- | --- | ----------- | -------------- | --- |
|        | πθ             |     |     |             | θ              |     |
| assume | no discounting |     | (γ  | =1) for the | episodic case. |     |
With function approximation it may seem challenging to change the policy parameter in a way that ensures
improvement. The problem is that performance depends on both the action selections and the distribution of
states in which those selections are made, and that both of these are affected by the policy parameter. Given a
state, the effect of the policy parameter on the actions, and thus on the reward, can be computed in a relatively
56

straightforwardwayfromknowledgeoftheparameterization. Buttheeffectofthepolicyonthestatedistribution
is a function of the environment and is typically unknown. How can we estimate the performance gradient with
respect to the policy parameter when the gradient depends on the unknown effect of policy changes on the state
distribution?
Here comes the theorem, which provides an analytical expression for the gradient of performance
policy gradient
with respect to the policy parameter. The policy gradient theorem for the episodic case establishes that
|     |     |     |        | (cid:88) | (cid:88)                 |     |     |
| --- | --- | --- | ------ | -------- | ------------------------ | --- | --- |
|     |     |     | ∇J(θ)∝ |          | µ(s) q π (s,a)∇π(a|s,θ), |     |     |
|     |     |     |        | s        | a                        |     |     |
The distribution here (seen in Chapters 8 and 9) is the on-policy distribution under π.
µ
| 9.3 REINFORCE: |     | Monte |     | Carlo | Policy | Gradient |     |
| -------------- | --- | ----- | --- | ----- | ------ | -------- | --- |
Recallourstrategyforstochasticgradientascent,whichrequiresawaytoobtainsamplessuchthattheexpectation
of the sample gradient is proportional to the actual gradient of the performance measure as a function of the
parameter. Thesamplegradientneedonlybeproportionaltothegradientbecauseanyconstantofproportionality
| can be absorbed | into the | step size | α, which | is otherwise | arbitrary. |     |     |
| --------------- | -------- | --------- | -------- | ------------ | ---------- | --- | --- |
Notice that the right-hand side of the policy gradient theorem is a sum over states weighted by how often the
statesoccurunderthetargetpolicyπ; ifπ isfollowed, thenstateswillbeencounteredintheseproportions. Thus:
|     |     |     |        | (cid:88) | (cid:88)              |     |     |
| --- | --- | --- | ------ | -------- | --------------------- | --- | --- |
|     |     |     | ∇J(θ)∝ |          | µ(s) q (s,a)∇π(a|s,θ) |     |     |
π
|     |     |     |     | s   | a        |          |     |
| --- | --- | --- | --- | --- | -------- | -------- | --- |
|     |     |     |     |     | (cid:34) | (cid:35) |     |
(cid:88)
|     |     |     |     | =E  | q (S ,a)∇π(a|S | ,θ) |     |
| --- | --- | --- | --- | --- | -------------- | --- | --- |
|     |     |     |     | π   | π t            | t   |     |
a
We could stop here and instantiate our stochastic gradient-ascent algorithm as
(cid:88)
|     |     |     | θ   | =θ +α | qˆ(S ,a,w)∇π(a|S | ,θ) |     |
| --- | --- | --- | --- | ----- | ---------------- | --- | --- |
|     |     |     | t+1 | t     | t                | t   |     |
a
but our current interest is the classical REINFORCE algorithm whose update at time t involves just A , the one
t
| action actually | taken at | time t. |     |     |     |     |     |
| --------------- | -------- | ------- | --- | --- | --- | --- | --- |
We continue our derivation of REINFORCE by introducing A in the same way as we introduced S , i.e., by
t t
replacing a sum over the random variable’s possible values by an expectation under π, and then sampling the
expectation.
|     |         | (cid:34) |          |         |             | (cid:35) |     |
| --- | ------- | -------- | -------- | ------- | ----------- | -------- | --- |
|     | ∇J(θ)∝E |          | (cid:88) |         | ∇π(a|S t    | ,θ)      |     |
|     |         |          | π(a|S    | ,θ)q (S | ,a)         |          |     |
|     |         | π        |          | t π     | t π(a|S ,θ) |          |     |
t
a
|     |     | (cid:20) |         |       | (cid:21)   |     |     |
| --- | --- | -------- | ------- | ----- | ---------- | --- | --- |
|     |     |          |         | ∇π(A  | t |S t ,θ) |     |     |
|     |     | =E       | q (S ,A | )     |            |     |     |
|     |     | π        | π t     | t π(A | |S ,θ)     |     |     |
t t
|     |     | (cid:20) | ∇π(A | |S ,θ) | (cid:21) |                      |                |
| --- | --- | -------- | ---- | ------ | -------- | -------------------- | -------------- |
|     |     | =E       |      | t t    | (because | E                    |                |
|     |     | π        | G t  |        |          | π [G t |S t ,A t ]=q | π (S t ,A t )) |
π(A |S ,θ)
t t
where is the return as usual. The final expression in brackets is exactly what is needed, a quantity that can
G t
be sampled on each time step whose expectation is proportional to the gradient. Using this sample to instantiate
our generic stochastic gradient-ascent algorithm yields the REINFORCE update:
|     |     |     |     |      | ∇π(A | |S ,θ) |     |
| --- | --- | --- | --- | ---- | ---- | ------ | --- |
|     |     |     |     | θ =θ | +αG  | t t    |     |
|     |     |     |     | t+1  | t t  |        |     |
|     |     |     |     |      | π(A  | |S ,θ) |     |
t t
The vector is the direction in parameter space that most increases the probability of repeating the action on
A
t
future visits to state S . The update increases the parameter vector in this direction proportional to the return,
t
and inversely proportional to the action probability. The former makes sense because it causes the parameter to
move most in the directions that favour actions that yield the highest return. The latter makes sense because
otherwise actions that are selected frequently are at an advantage and might win out even if they do not yield the
highest return.
57

|     |     |     | Figure | 9.1: Monte-Carlo |     | Policy-Gradient |     | Control | (episodic) | for π |
| --- | --- | --- | ------ | ---------------- | --- | --------------- | --- | ------- | ---------- | ----- |
∗
As a stochastic gradient method, REINFORCE has good theoretical convergence properties. By construction,
the expected update over an episode is in the same direction as the performance gradient. This assures an
improvementinexpectedperformanceforsufficientlysmallα,andconvergencetoalocaloptimumunderstandard
stochastic approximation conditions for decreasing α. However, as Monte Carlo method REINFORCE may be of
| high variance | and       | thus produce |      | slow learning. |     |     |     |     |     |     |
| ------------- | --------- | ------------ | ---- | -------------- | --- | --- | --- | --- | --- | --- |
| 9.3.1         | REINFORCE |              | with | Baseline       |     |     |     |     |     |     |
The REINFORCE update derived above is unbiased, but it can suffer from high variance because the return
G t
depends on all future rewards in the trajectory. A standard way to reduce this variance is to subtract a baseline
| from the      | return.  |                    |     |           |     |           |     |             |            |     |
| ------------- | -------- | ------------------ | --- | --------- | --- | --------- | --- | ----------- | ---------- | --- |
| The update    | becomes: |                    |     |           |     |           |     |             |            |     |
|               |          |                    |     |           |     | (cid:0)   |     | (cid:1)∇π(A | t |S t ,θ) |     |
|               |          |                    |     | θ         | =θ  | +α G −b(S | )   |             | .          |     |
|               |          |                    |     | t+1       | t   | t         | t   | π(A         | |S ,θ)     |     |
|               |          |                    |     |           |     |           |     |             | t t        |     |
| Equivalently, | using    | the log-derivative |     | identity, |     |           |     |             |            |     |
∇ π (a|s)
|     |     |     |     |     | ∇   | logπ (a|s)= |     | θ θ     | ,   |     |
| --- | --- | --- | --- | --- | --- | ----------- | --- | ------- | --- | --- |
|     |     |     |     |     | θ   | θ           |     | π (a|s) |     |     |
θ
we can write:
|     |     |     |     |       |         | (cid:0)  | (cid:1) |        |                |     |
| --- | --- | --- | --- | ----- | ------- | -------- | ------- | ------ | -------------- | --- |
|     |     |     |     | θ t+1 | =θ t +α | G t −b(S | t ) ∇   | θ logπ | θ (A t |S t ). |     |
Thebaselineb(S )canbeanyfunctionofthestate,aslongasitdoesnotdependonthespecificactionA selected
t t
at time t. The reason is that subtracting such a baseline does not change the expected policy gradient:
|     |     |     |      |         |             |     | (cid:88) |         | ∇ π (a|s) |       |
| --- | --- | --- | ---- | ------- | ----------- | --- | -------- | ------- | --------- | ----- |
|     |     |     | E    | [∇ logπ | (a|s)b(s)]= |     |          | π (a|s) | θ θ       | b(s). |
|     |     |     | a∼πθ | θ       | θ           |     |          | θ       |           |       |
π θ (a|s)
a∈A
Therefore,
(cid:88)
|     |     |     |     | E    | [∇ logπ | (a|s)b(s)]=b(s) |     |     | ∇ π (a|s). |     |
| --- | --- | --- | --- | ---- | ------- | --------------- | --- | --- | ---------- | --- |
|     |     |     |     | a∼πθ | θ       | θ               |     |     | θ θ        |     |
a∈A
| Since the | probabilities | of  | all actions | must | sum | to one, |     |     |     |     |
| --------- | ------------- | --- | ----------- | ---- | --- | ------- | --- | --- | --- | --- |
(cid:88)
π θ (a|s)=1,
a∈A
we have
|     |     |     |     | (cid:88) |           |     | (cid:88) |         |      |     |
| --- | --- | --- | --- | -------- | --------- | --- | -------- | ------- | ---- | --- |
|     |     |     |     | ∇        | π (a|s)=∇ |     | π        | (a|s)=∇ | 1=0. |     |
|     |     |     |     |          | θ θ       | θ   | θ        |         | θ    |     |
|     |     |     |     | a∈A      |           |     | a∈A      |         |      |     |
Thus,
|     |     |     |     |     | E    | [∇ logπ | (a|s)b(s)]=0. |     |     |     |
| --- | --- | --- | --- | --- | ---- | ------- | ------------- | --- | --- | --- |
|     |     |     |     |     | a∼πθ | θ       | θ             |     |     |     |
This shows that the baseline does not introduce bias into the gradient estimate. It only changes the variance. A
common and useful choice is to use the state-value function as the baseline:
|     |     |     |     |     |     | b(S )=v | (S  | ).  |     |     |
| --- | --- | --- | --- | --- | --- | ------- | --- | --- | --- | --- |
|     |     |     |     |     |     | t       | π   | t   |     |     |
In practice, is usually unknown and is approximated by a parameterized value function ,w).
v π (S t ) vˆ(S t
58

9.3.2 Advantage Function
Using the state-value function as a baseline leads naturally to the advantage function. The advantage of taking
action a in state s under policy π is defined as:
Aπ(s,a)=q (s,a)−v (s).
π π
The value q (s,a) measures the expected return after taking action a in state s and then following policy π. The
π
value v (s) measures the expected return from state s when following the policy normally.
π
Therefore, the advantage function measures how much better or worse action a is compared to the average action
selected by the policy in state s.
Using the advantage function, the policy gradient can be written as:
∇J(θ)∝E [Aπ(S ,A )∇ logπ (A |S )].
π t t θ θ t t
This form is useful because it updates the policy according to whether the selected action was better or worse
than expected:
• if Aπ(S ,A )>0, the probability of action A is increased;
t t t
• if Aπ(S ,A )<0, the probability of action A is decreased;
t t t
• if Aπ(S ,A )=0, the action performed as expected and the update is small or zero.
t t
9.3.3 Actor-Critic Methods
REINFORCE is a Monte Carlo method. This means that it must wait until the end of an episode to compute the
return G . Although this gives an unbiased estimate, it can produce high variance and slow learning.
t
Actor-critic methods combine policy-gradient methods with temporal-difference learning. The method has two
components:
• the actor, which is the parameterized policy π (a|s);
θ
• the critic, which estimates the value function, usually vˆ(s,w).
The actor decides which action to take, while the critic evaluates how good the action was. Instead of using the
full Monte Carlo return G , the critic uses a one-step TD error:
t
δ =R +γvˆ(S ,w)−vˆ(S ,w).
t t+1 t+1 t
This TD error can be interpreted as an estimate of the advantage function:
δ ≈Aπ(S ,A ).
t t t
Indeed, if the value-function approximation is accurate, then:
E[δ |S ,A ]≈q (S ,A )−v (S )=Aπ(S ,A ).
t t t π t t π t t t
The critic is updated using a semi-gradient TD update:
w =w +α δ ∇ vˆ(S ,w).
t+1 t w t w t
The actor is updated in the policy-gradient direction, using the TD error as the advantage estimate:
θ =θ +α δ ∇ logπ (A |S ).
t+1 t θ t θ θ t t
Therefore, the actor-critic algorithm proceeds as follows:
1. observe the current state S ;
t
2. sample an action A ∼π (·|S );
t θ t
3. observe the reward R and the next state S ;
t+1 t+1
4. compute the TD error
δ =R +γvˆ(S ,w)−vˆ(S ,w);
t t+1 t+1 t
5. update the critic:
w =w +α δ ∇ vˆ(S ,w);
t+1 t w t w t
6. update the actor:
θ =θ +α δ ∇ logπ (A |S ).
t+1 t θ t θ θ t t
The advantage of actor-critic methods is that they can learn online, step by step, without waiting for the end of
the episode. Compared with REINFORCE, they usually reduce variance because they replace the full return G
t
with a bootstrapped estimate. However, because the critic is only an approximation of the true value function,
actor-critic methods can introduce bias.
59

| Chapter  |            | 10  |                  |            |          |          |         |       |     |
| -------- | ---------- | --- | ---------------- | ---------- | -------- | -------- | ------- | ----- | --- |
| Advanced |            |     | Policy           |            | Gradient |          |         | Notes |     |
| 10.1     | Derivation |     | of               | the Policy |          | Gradient | Theorem |       |     |
| 10.1.1   | Setup      | and | Parameterization |            |          |          |         |       |     |
We consider the same Markov Decision Process (MDP) notation as before. However, instead of optimizing over
the space of all possible policies, we assume the policy is parameterized by a continuous vector ∈Rd.
θ
Let denote the probability of taking action in state under parameters θ. The goal is to maximize the
| π θ (a|s) |            |         |     |     |                    | a   | s   |     |        |
| --------- | ---------- | ------- | --- | --- | ------------------ | --- | --- | --- | ------ |
| expected  | discounted | return: |     |     |                    |     |     |     |        |
|           |            |         |     |     | J(θ)=rT(I−γPπθ)−1ρ |     |     |     | (10.1) |
where:
| • Pπθ is | the |     | left-stochastic | transition |     | matrix with | entries |     |     |
| -------- | --- | --- | --------------- | ---------- | --- | ----------- | ------- | --- | --- |
|S|×|S|
(cid:88)
|     |     |     |     |     | Pπθ  | = p(s′ | |s,a)π | (a|s). |     |
| --- | --- | --- | --- | --- | ---- | ------ | ------ | ------ | --- |
|     |     |     |     |     | s′,s |        | θ      |        |     |
a
| • is the       |       | column       | vector  | of state               | rewards. |               |     |     |     |
| -------------- | ----- | ------------ | ------- | ---------------------- | -------- | ------------- | --- | --- | --- |
| r              | |S|×1 |              |         |                        |          |               |     |     |     |
| • ρ is the     | |S|×1 | column       | vector  | of the initial         | state    | distribution. |     |     |     |
| • γ ∈[0,1)     | is    | the discount | factor. |                        |          |               |     |     |     |
| As established |       | previously,  | we      | define the state-value |          | row vector    |     |     |     |
Vπθ =rT(I−γPπθ)−1
| and the | discounted | state | visitation | column | vector |     |     |     |     |
| ------- | ---------- | ----- | ---------- | ------ | ------ | --- | --- | --- | --- |
µπθ =(I−γPπθ)−1ρ.
Theexpectedreturnoftakingactionainstates,andthereafter
| Definition | 1   | (State-Action |     | Value Function). |     |     |     |     |     |
| ---------- | --- | ------------- | --- | ---------------- | --- | --- | --- | --- | --- |
| following  | π , | is defined    | as: |                  |     |     |     |     |     |
θ
(cid:88)
|     |     |     |     | Qπθ(s,a)=r |     | +γ  | p(s′ |s,a)V | π θ. | (10.2) |
| --- | --- | --- | --- | ---------- | --- | --- | ----------- | ---- | ------ |
|     |     |     |     |            |     | s   |             | s ′  |        |
s′∈S
| 10.1.2 | Matrix | Calculus |     | Identity |     |     |     |     |     |
| ------ | ------ | -------- | --- | -------- | --- | --- | --- | --- | --- |
To compute the gradient ∇ J(θ), we need to differentiate the inverse of a matrix that depends on θ. We rely on
θ
| a fundamental |     | identity | from | matrix calculus. |     |     |     |     |     |
| ------------- | --- | -------- | ---- | ---------------- | --- | --- | --- | --- | --- |
Lemma 1 (Derivative of an Inverse Matrix). ForaninvertiblematrixA(θ)parameterizedbyθ, thegradient
| of its inverse |     | is: |     |                    |     |     |              |     |        |
| -------------- | --- | --- | --- | ------------------ | --- | --- | ------------ | --- | ------ |
|                |     |     |     | ∇ A−1(θ)=−A−1(θ)(∇ |     |     | A(θ))A−1(θ). |     | (10.3) |
|                |     |     |     | θ                  |     |     | θ            |     |        |
Inourcontext,A(θ)=I−γPπθ . BecausetheidentitymatrixI doesnotdependonθ,wehave∇ A(θ)=−γ∇ Pπθ .
θ θ
| Substituting | this | into | the lemma | yields:      |                |     |     |                 |        |
| ------------ | ---- | ---- | --------- | ------------ | -------------- | --- | --- | --------------- | ------ |
|              |      |      |           | ∇ (I−γPπθ)−1 | =γ(I−γPπθ)−1(∇ |     |     | Pπθ)(I−γPπθ)−1. | (10.4) |
|              |      |      |           | θ            |                |     | θ   |                 |        |
60

| 10.1.3 | Derivation |     | of  | the | Policy | Gradient |     | Theorem |     |     |     |     |     |
| ------ | ---------- | --- | --- | --- | ------ | -------- | --- | ------- | --- | --- | --- | --- | --- |
Theorem 1 (Policy Gradient Theorem). For any differentiable policy π , the gradient of the objective J(θ)
θ
is:
|     |     |     |     |     |         |     | (cid:88) | (cid:88) |                |     |     |     |        |
| --- | --- | --- | --- | --- | ------- | --- | -------- | -------- | -------------- | --- | --- | --- | ------ |
|     |     |     |     |     | ∇ J(θ)= |     | µπ θ     | ∇ π      | (a|s)Qπθ(s,a). |     |     |     | (10.5) |
|     |     |     |     |     | θ       |     | s        | θ        | θ              |     |     |     |        |
|     |     |     |     |     |         |     | s∈S      | a∈A      |                |     |     |     |        |
Proof. We begin by applying the gradient operator to the objective function:
|              |     |        |          |           |                             | (cid:2)       |     | (cid:3) | (cid:2)               | (I−γPπθ)−1(cid:3) |         |     |     |
| ------------ | --- | ------ | -------- | --------- | --------------------------- | ------------- | --- | ------- | --------------------- | ----------------- | ------- | --- | --- |
|              |     |        | ∇        | J(θ)=∇    |                             | rT(I−γPπθ)−1ρ |     |         | =rT                   | ∇                 | ρ.      |     |     |
|              |     |        |          | θ         | θ                           |               |     |         |                       | θ                 |         |     |     |
| Substituting | the | matrix | identity | from      | Equation                    |               | 4:  |         |                       |                   |         |     |     |
|              |     |        |          |           |                             | (cid:2)       |     |         | Pπθ)(I−γPπθ)−1(cid:3) |                   |         |     |     |
|              |     |        |          | ∇ J(θ)=rT |                             | γ(I−γPπθ)−1(∇ |     |         |                       |                   | ρ       |     |     |
|              |     |        |          | θ         |                             |               |     |         | θ                     |                   |         |     |     |
|              |     |        |          |           | (cid:2) rT(I−γPπθ)−1(cid:3) |               |     | Pπθ)    | (cid:2) (I−γPπθ)−1ρ   |                   | (cid:3) |     |     |
|              |     |        |          | =γ        |                             |               |     | (∇ θ    |                       |                   | .       |     |     |
Recognizing our definitions for Vπθ and µπθ , this dramatically simplifies to a vector-matrix-vector product:
(10.6)
|     |     |     |     |     |     | ∇ J(θ)=γVπθ(∇ |     |     | Pπθ)µπθ. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------- | --- | --- | -------- | --- | --- | --- | --- |
|     |     |     |     |     |     | θ             |     |     | θ        |     |     |     |     |
Now, we expand Equation 6 into its scalar sums. The element at row s′ and column of the matrix Pπθ is
|     |     |     |     |     |     |     |     |     |     |     |     | s   | ∇ θ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:80)
| given by | p(s′ | |s,a)∇ | π   | (a|s).     |          |             |          |            |          |        |          |     |     |
| -------- | ---- | ------ | --- | ---------- | -------- | ----------- | -------- | ---------- | -------- | ------ | -------- | --- | --- |
|          | a    |        | θ θ |            |          |             |          |            |          |        |          |     |     |
|          |      |        |     |            |          | (cid:88) µπ | (cid:88) | π (cid:88) | p(s′     |        |          |     |     |
|          |      |        |     | ∇ θ J(θ)=γ |          |             | θ V      | θ          | |s,a)∇   | θ π θ  | (a|s)    |     |     |
|          |      |        |     |            |          | s           |          | s ′        |          |        |          |     |     |
|          |      |        |     |            |          | s∈S         | s′∈S     | a∈A        |          |        |          |     |     |
|          |      |        |     |            |          |             |          |            | (cid:34) |        | (cid:35) |     |     |
|          |      |        |     |            | (cid:88) | µπ          | (cid:88) |            | (cid:88) | p(s′   | π        |     |     |
|          |      |        |     |            | =        | θ           | ∇ θ      | π θ (a|s)  | γ        | |s,a)V | θ .      |     |     |
|          |      |        |     |            |          | s           |          |            |          |        | s ′      |     |     |
s′∈S
|     |     |     |     |     | s∈S |     | a∈A |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Notice that the term in the brackets is exactly Qπθ(s,a)−r . Substituting this back into the sum:
s
|     |     |     |         | (cid:88) |     | (cid:88) |                     |     |     |          |           |          |     |
| --- | --- | --- | ------- | -------- | --- | -------- | ------------------- | --- | --- | -------- | --------- | -------- | --- |
|     |     |     | ∇ J(θ)= |          | µπ  | θ ∇      | π (a|s)(Qπθ(s,a)−r  |     |     | )        |           |          |     |
|     |     |     | θ       |          | s   |          | θ θ                 |     |     | s        |           |          |     |
|     |     |     |         | s∈S      |     | a∈A      |                     |     |     |          |           |          |     |
|     |     |     |         |          |     | (cid:32) |                     |     |     |          |           | (cid:33) |     |
|     |     |     |         | (cid:88) |     | (cid:88) |                     |     |     | (cid:88) |           |          |     |
|     |     |     |         | =        | µπ  | θ        | ∇ π (a|s)Qπθ(s,a)−r |     |     |          | ∇ π (a|s) | .        |     |
|     |     |     |         |          | s   |          | θ θ                 |     |     | s        | θ θ       |          |     |
|     |     |     |         | s∈S      |     | a∈A      |                     |     |     | a∈A      |           |          |     |
Finally, recall that probabilities must sum to 1, meaning (cid:80) (a|s)=1. The derivative of a constant is zero, so
π
| (cid:80) |     |     |     |     |     |     |     | a   | θ   |     |     |     |     |
| -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∇ π (a|s)=∇ (1)=0. Consequently, the term involving the state reward r completely vanishes:
| a θ θ  |     | θ              |     |     |         |     |          |          |                |     | s   |     |        |
| ------ | --- | -------------- | --- | --- | ------- | --- | -------- | -------- | -------------- | --- | --- | --- | ------ |
|        |     |                |     |     |         |     | (cid:88) | (cid:88) |                |     |     |     | (10.7) |
|        |     |                |     |     | ∇ J(θ)= |     | µπ θ     | ∇ π      | (a|s)Qπθ(s,a). |     |     |     |        |
|        |     |                |     |     | θ       |     | s        | θ        | θ              |     |     |     |        |
|        |     |                |     |     |         |     | s∈S      | a∈A      |                |     |     |     |        |
| 10.1.4 | The | Log-Derivative |     |     | Trick   |     |          |          |                |     |     |     |        |
To make this theorem practical for estimation via sampling, we utilize the log-derivative trick:
|     |     |     |     |           |     |       | ∇ θ π   | θ (a|s) |           |     |             |     |        |
| --- | --- | --- | --- | --------- | --- | ----- | ------- | ------- | --------- | --- | ----------- | --- | ------ |
|     |     |     | ∇   | π (a|s)=π |     | (a|s) |         |         | =π (a|s)∇ |     | logπ (a|s). |     | (10.8) |
|     |     |     |     | θ θ       |     | θ     | π (a|s) |         | θ         | θ   | θ           |     |        |
θ
SubstitutingthisintothePolicyGradientTheoremyieldsanexpectationovertrajectoriesgeneratedbythepolicy:
|          |            |         |        |         |            | (cid:88) | (cid:88) |                 |      |               |     |     |        |
| -------- | ---------- | ------- | ------ | ------- | ---------- | -------- | -------- | --------------- | ---- | ------------- | --- | --- | ------ |
|          |            |         |        | ∇ J(θ)= |            | µπ       | π        | (a|s)∇          | logπ | (a|s)Qπθ(s,a) |     |     | (10.9) |
|          |            |         |        | θ       |            | s        | θ θ      |                 | θ    | θ             |     |     |        |
|          |            |         |        |         |            | s∈S      | a∈A      |                 |      |               |     |     |        |
| which is | commonly   | written |        | as      |            |          |          |                 |      |               |     |     |        |
|          |            |         |        |         | E          |          |          | (a|s)Qπθ(s,a)]. |      |               |     |     |        |
|          |            |         |        |         | s∼µπθ,a∼πθ |          | [∇       | logπ            |      |               |     |     |        |
|          |            |         |        |         |            |          | θ        | θ               |      |               |     |     |        |
| 10.2     | Global     |         | Optima |         | in Markov  |          | Decision |                 |      | Processes     |     |     |        |
| 10.2.1   | Background |         |        | and     | Notation   |          |          |                 |      |               |     |     |        |
WeconsideraMarkovDecisionProcess(MDP)definedbyafinitestatespaceS,afiniteactionspaceA,atransition
dynamicsmodelp(s′ |s,a),andageneralrewardfunctionr(s,a),whichrepresentstheexpectedimmediatereward
| received | when taking |     | action | in state | s.  |     |     |     |     |     |     |     |     |
| -------- | ----------- | --- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
a
61

Astochasticpolicyπ(a|s)defines theprobability oftaking actiona instate s. Givenapolicy π, wecan compute
| the expected | immediate |     | reward | for each | state. |     |     |     |     |     |     |     |     |
| ------------ | --------- | --- | ------ | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
Definition 1 (Expected Reward Vector). Let rπ ∈ R|S|×1 be the column vector of expected immediate
| rewards | under policy | π.  | The entry | for | state | s is: |     |     |     |     |     |     |     |
| ------- | ------------ | --- | --------- | --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|     |     |     |     |     |     | rπ = | π(a|s)r(s,a). |     |     |     |     |     | (10.10) |
| --- | --- | --- | --- | --- | --- | ---- | ------------- | --- | --- | --- | --- | --- | ------- |
s
a∈A
Definition 2 (Left-Stochastic Transition Matrix). Let Pπ ∈ R|S|×|S| be the transition probability matrix
induced by π. Using the column-to-row (left-stochastic) convention, the entry in row s′ and column s is:
|     |     |     |     |     | Pπ  | (cid:88) | p(s′ |              |     |     |     |     | (10.11) |
| --- | --- | --- | --- | --- | --- | -------- | ---- | ------------ | --- | --- | --- | --- | ------- |
|     |     |     |     |     |     | =        |      | |s,a)π(a|s). |     |     |     |     |         |
s′,s
a∈A
Let ρ ∈ R|S|×1 be the initial state probability distribution vector. The objective function is the expected sum of
| discounted | rewards, | written         | in  | matrix-vector |                      | form: |     |     |     |     |     |     |         |
| ---------- | -------- | --------------- | --- | ------------- | -------------------- | ----- | --- | --- | --- | --- | --- | --- | ------- |
|            |          |                 |     |               | J(π)=(rπ)T(I−γPπ)−1ρ |       |     |     |     |     |     |     | (10.12) |
| where γ    | ∈[0,1)   | is the discount |     | factor.       |                      |       |     |     |     |     |     |     |         |
Definition 3 (State Value Row-Vector). We define Vπ ∈ R1×|S| as the row vector of expected discounted
returns:
|            |               |       | Vπ =(rπ)T(I−γPπ)−1 |                |                 |     | =⇒  |     | Vπ           | =(rπ)T | +γVπPπ.   |            | (10.13)   |
| ---------- | ------------- | ----- | ------------------ | -------------- | --------------- | --- | --- | --- | ------------ | ------ | --------- | ---------- | --------- |
|            |               |       |                    |                |                 |     |     |     | We define    |        | R|S|×1 as | the column | vector of |
| Definition | 4 (Discounted |       | Visitation         |                | Column-Vector). |     |     |     |              | µπ     | ∈         |            |           |
| expected   | discounted    | state | visitations:       |                |                 |     |     |     |              |        |           |            |           |
|            |               |       |                    | µπ =(I−γPπ)−1ρ |                 |     | =⇒  |     | µπ =ρ+γPπµπ. |        |           |            | (10.14)   |
Definition 5 (State-Action Value Function). Theexpectedreturnoftakingactionainstates,andthereafter
| following | π, is: |     |     |     |     |     |     |     |     |     |     |     |     |
| --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|     |     |     |     | Qπ(s,a)=r(s,a)+γ |     |     |     |     | p(s′ |s,a)Vπ. |     |     |     | (10.15) |
| --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | ------------- | --- | --- | --- | ------- |
s′
s′∈S
| 10.2.2 | The | Performance |     | Difference |     |     | Lemma |     |     |     |     |     |     |
| ------ | --- | ----------- | --- | ---------- | --- | --- | ----- | --- | --- | --- | --- | --- | --- |
Lemma 1 (Performance Difference). For any two policies π and π′, the exact difference in their expected
returns is:
|     |     |     |             |     |     | (cid:88) | (cid:88)                 |     |     |     |     |     |         |
| --- | --- | --- | ----------- | --- | --- | -------- | ------------------------ | --- | --- | --- | --- | --- | ------- |
|     |     |     | J(π′)−J(π)= |     |     | µπ′      | (π′(a|s)−π(a|s))Qπ(s,a). |     |     |     |     |     | (10.16) |
s
|     |     |     |     |     |     | s∈S | a∈A |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Proof. Using the identity Vπ′(I−γPπ′)=(rπ′)T, we evaluate the difference between Vπ′ and Vπ:
|     |     |     |     |         |       | Vπ′ −γVπ′ | Pπ′    | =(rπ′ | )T  |         |     |     |     |
| --- | --- | --- | --- | ------- | ----- | --------- | ------ | ----- | --- | ------- | --- | --- | --- |
|     |     |     |     | Vπ′ −Vπ | =(rπ′ | )T        | −(rπ)T | +γVπ′ | Pπ′ | −γVπPπ. |     |     |     |
We add and subtract γVπPπ′ on the right side to factor out the value functions:
|                  |     |             | Vπ′ −Vπ     | =(rπ′       | )T             | −(rπ)T      | +γ(Vπ′   | −Vπ)Pπ′   |          | +γVπ(Pπ′          | −Pπ)      |     |         |
| ---------------- | --- | ----------- | ----------- | ----------- | -------------- | ----------- | -------- | --------- | -------- | ----------------- | --------- | --- | ------- |
|                  |     |             | (Vπ′        | −Vπ)(I−γPπ′ |                | )=(rπ′      |          | )T −(rπ)T | +γVπ(Pπ′ |                   | −Pπ)      |     |         |
|                  |     |             | Vπ′         |             | (cid:104) (rπ′ |             | +γVπ(Pπ′ |           |          | (cid:105) (I−γPπ′ |           |     |         |
|                  |     |             | −Vπ         | =           | )T             | −(rπ)T      |          |           | −Pπ)     |                   | )−1.      |     |         |
| Post-multiplying |     | by ρ yields | the         | scalar      | objective      | difference: |          |           |          |                   |           |     |         |
|                  |     |             |             |             |                | (cid:104)   |          |           |          |                   | (cid:105) |     |         |
|                  |     |             | J(π′)−J(π)= |             |                | (rπ′ )T     | −(rπ)T   | +γVπ(Pπ′  |          | −Pπ)              | µπ′       |     | (10.17) |
.
| We now | expand | the term | inside | the brackets |          | for a    | specific | state    | s:      |          |     |     |     |
| ------ | ------ | -------- | ------ | ------------ | -------- | -------- | -------- | -------- | ------- | -------- | --- | --- | --- |
|        |        |          |        |              | (cid:16) | (cid:17) |          | (cid:16) |         | (cid:17) |     |     |     |
|        |        |          | Terms= |              | rπ′ −rπ  |          | (cid:88) | Vπ       | Pπ′ −Pπ |          |     |     |     |
+γ
|     |     |     |     |     | s   | s   |     | s′  | s′,s | s′,s |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | --- | --- | --- |
s′∈S
(cid:88)
|     |     |     |     | =   | (π′(a|s)−π(a|s))r(s,a) |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
a∈A
|     |     |     |     |     | (cid:88) | (cid:88) |                      |     |     |     |        |     |     |
| --- | --- | --- | --- | --- | -------- | -------- | -------------------- | --- | --- | --- | ------ | --- | --- |
|     |     |     |     |     | +γ       | Vπ       | (π′(a|s)−π(a|s))p(s′ |     |     |     | |s,a). |     |     |
s′
|     |     |     |     |     | s′∈S | a∈A |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
62

| Factoring | out the | policy | difference |          | (π′(a|s)−π(a|s)): |     |          |          |              |          |     |
| --------- | ------- | ------ | ---------- | -------- | ----------------- | --- | -------- | -------- | ------------ | -------- | --- |
|           |         |        |            |          |                   |     | (cid:34) |          |              | (cid:35) |     |
|           |         |        | Terms=     | (cid:88) | (π′(a|s)−π(a|s))  |     |          | (cid:88) | p(s′ |s,a)Vπ |          |     |
r(s,a)+γ
s′
|     |     |     |     | a∈A |     |     |     | s′∈S |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- |
(cid:88)
|     |     |     |     | =   | (π′(a|s)−π(a|s))Qπ(s,a). |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | ------------------------ | --- | --- | --- | --- | --- | --- |
a∈A
| Substituting | this | back into | Equation    |     | 8 gives | the final    | result:                  |     |     |     |         |
| ------------ | ---- | --------- | ----------- | --- | ------- | ------------ | ------------------------ | --- | --- | --- | ------- |
|              |      |           |             |     |         | (cid:88) µπ′ | (cid:88)                 |     |     |     | (10.18) |
|              |      |           | J(π′)−J(π)= |     |         |              | (π′(a|s)−π(a|s))Qπ(s,a). |     |     |     |         |
s
|         |       |            |         |     |          | s∈S  | a∈A         |          |     |     |     |
| ------- | ----- | ---------- | ------- | --- | -------- | ---- | ----------- | -------- | --- | --- | --- |
| 10.2.3  | Local | and        | Global  |     | Optima   |      |             |          |     |     |     |
|         | If    | is a local | maximum |     | of J(π), | then | is a global | maximum. |     |     |     |
| Theorem | 1.    | π          |         |     |          |      | π           |          |     |     |     |
Assume is a local maximum. Consider a linear mixture towards any other policy π′:
| Proof. |     | π   |     |     |              |     |              |     |     |     |     |
| ------ | --- | --- | --- | --- | ------------ | --- | ------------ | --- | --- | --- | --- |
|        |     |     |     |     | π =π+α(π′−π) |     | for α∈[0,1]. |     |     |     |     |
α
Applying the generalized Performance Difference Lemma and taking the limit as α→0+ (where µπα →µπ):
|     |     |     |     | J(π )−J(π) |     | (cid:88) | (cid:88)                   |     |     |     |         |
| --- | --- | --- | --- | ---------- | --- | -------- | -------------------------- | --- | --- | --- | ------- |
|     |     |     | lim | α          | =   | µπ       | (π′(a|s)−π(a|s))Qπ(s,a)≤0. |     |     |     | (10.19) |
s
|     |     |     | α→0+ |     | α   |     |     |     |     |     |     |
| --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |      |     |     | s∈S | a∈A |     |     |     |     |
Because the choice of π′(·| s) can be made completely independently for each state s, and assuming µπ >0, the
s
| inner sum | must | be non-positive |     | for             | every state: |     |               |     |         |     |         |
| --------- | ---- | --------------- | --- | --------------- | ------------ | --- | ------------- | --- | ------- | --- | ------- |
|           |      |                 |     | (cid:88)        |              |     | (cid:88)      |     |         |     |         |
|           |      |                 |     | π′(a|s)Qπ(s,a)≤ |              |     | π(a|s)Qπ(s,a) |     | ∀π′,∀s. |     | (10.20) |
|           |      |                 |     | a∈A             |              |     | a∈A           |     |         |     |         |
Thisindicatesthatthelocalmaximumpolicyπ satisfiestheBellmanOptimalityEquation: itplacesallprobability
| mass on | actions | that maximize |     | Qπ(s,a). |     |     |     |     |     |     |     |
| ------- | ------- | ------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
Now, let π∗ be an absolute global optimal policy. We apply the Performance Difference Lemma between π∗ and
| our local | maximum | π:  |             |     |          |          |                           |     |     |          |         |
| --------- | ------- | --- | ----------- | --- | -------- | -------- | ------------------------- | --- | --- | -------- | ------- |
|           |         |     |             |     |          | (cid:34) |                           |     |     | (cid:35) |         |
|           |         |     |             |     | (cid:88) | (cid:88) |                           |     |     |          |         |
|           |         |     | J(π∗)−J(π)= |     |          | µπ∗      | π∗(a|s)Qπ(s,a)−maxQπ(s,a) |     |     | .        | (10.21) |
s
a
|     |     |     |     |     | s∈S | a∈A |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Because the term in the brackets is mathematically forced to be for every state, and µπ∗ 0, the sum is
|     |     |     |     |     |     |     |     | ≤ 0 |     | ≥   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
s
non-positive:
J(π∗)−J(π)≤0.
| Since π∗ | is the | global maximum, |     | J(π∗)≥J(π). |     | Thus, |     |     |     |     |     |
| -------- | ------ | --------------- | --- | ----------- | --- | ----- | --- | --- | --- | --- | --- |
J(π)=J(π∗).
63

| Chapter     | 11      |               |     |     |          |
| ----------- | ------- | ------------- | --- | --- | -------- |
| Multi-Agent |         | Reinforcement |     |     | Learning |
| 11.1        | General | Introduction  |     |     |          |
Multi-agent reinforcement learning studies decision-making problems in which more than one agent interacts in
the same environment. In this setting, the outcome of one agent’s action depends not only on the state of the
environment, but also on the actions selected by the other agents. As a result, the environment is no longer
stationary from the perspective of a single learner: while one agent is improving its policy, the behavior of the
| others may  | change as | well.              |             |     |     |
| ----------- | --------- | ------------------ | ----------- | --- | --- |
| Multi-agent | problems  | can have different | structures: |     |     |
• cooperative, when agents share the same objective and try to maximize a common return;
• competitive, when one agent’s gain corresponds to another agent’s loss, as in zero-sum games;
• mixed, when agents may have partially aligned and partially conflicting interests.
The main difficulty is that each agent must learn while adapting to the presence of other learners. This creates
several challenges, including non-stationarity, coordination, credit assignment, and the possible need to model the
| behavior | or intentions | of other agents. |     |     |     |
| -------- | ------------- | ---------------- | --- | --- | --- |
In cooperative settings, agents must discover policies that work well together, often without direct access to
each other’s internal state. In competitive settings, agents must instead learn robust strategies that perform well
againststrongopponents. Forthisreason,multi-agentreinforcementlearningoftendrawsfrombothreinforcement
| learning | and game theory. |     |     |     |     |
| -------- | ---------------- | --- | --- | --- | --- |
Typicalapplicationsincludeboardgames,roboticteams,autonomousdriving,distributedcontrol,networkrouting,
and economic or strategic environments in which multiple decision-makers interact over time.
| 11.1.1 | Formulation |     |     |     |     |
| ------ | ----------- | --- | --- | --- | --- |
In the multi-agent case, a policy is no longer associated with a single decision-maker, but with each agent in the
system. If there are n agents, then the evolution of the environment depends on a joint action
(a1,a2,...,an),
where ai is the action selected by agent i. Similarly, rewards may be shared by all agents or may be different for
| each one, | depending | on whether the | task is cooperative | or competitive. |     |
| --------- | --------- | -------------- | ------------------- | --------------- | --- |
Thismeansthatthequalityofoneagent’spolicycannotbeevaluatedindependentlyfromthepoliciesoftheothers.
A strategy that is good against one set of opponents may be poor against another, and in cooperative tasks an
individually reasonable action may still be harmful if it is not coordinated with the team.
| 11.1.2 | Common | Approaches |     |     |     |
| ------ | ------ | ---------- | --- | --- | --- |
Onesimpleapproachistoleteachagentlearnindependently, treatingtheothersaspartoftheenvironment. This
is often easy to implement, but it ignores the fact that the environment is changing because the other agents are
learning too.
A common alternative is centralized training with decentralized execution. During training, agents may
use additional information such as the observations or actions of the other agents, which helps stabilize learning.
At execution time, however, each agent must act using only the information available locally. This framework is
widely used because it combines richer learning signals with realistic deployment constraints.
64

| 11.1.3 | Relation | with | Game | Theory |     |     |     |     |
| ------ | -------- | ---- | ---- | ------ | --- | --- | --- | --- |
Multi-agent reinforcement learning is closely connected to game theory because agents are not only optimizing
returns, but are also reacting to each other strategically. In competitive settings, an important goal is to learn
policies that are hard to exploit. In cooperative settings, the focus is often on coordination and on reaching joint
| behaviors | that are | beneficial | for the | whole group. |     |     |     |     |
| --------- | -------- | ---------- | ------- | ------------ | --- | --- | --- | --- |
For this reason, ideas such as equilibrium, best response, and opponent modeling often appear naturally in multi-
agentRL.Reinforcementlearningprovidesthemechanismforlearningfromexperience,whilegametheoryprovides
| tools to | reason about | strategic | interaction. |     |     |     |     |     |
| -------- | ------------ | --------- | ------------ | --- | --- | --- | --- | --- |
Exercise:
For each game, identify the main RL components of the problem and discuss which challenges arise from the
| presence | of multiple | agents. |     |     |       |     |     |     |
| -------- | ----------- | ------- | --- | --- | ----- | --- | --- | --- |
|          |             |         |     |     | Chess |     |     | GO  |
Boardconfiguration
|     |     |     |     | Castlingrights |            |     | Boardconfiguration |            |
| --- | --- | --- | --- | -------------- | ---------- | --- | ------------------ | ---------- |
|     |     |     |     |                | Sidetomove |     |                    | Sidetomove |
StateSpace
|     |     |     |     | Enpassantavailability |           |     | Ko-relatedinformation |               |
| --- | --- | --- | --- | --------------------- | --------- | --- | --------------------- | ------------- |
|     |     |     |     |                       | Drawrules |     |                       | Terminalstate |
Terminalstate
|     |         |     | Alllegalmovesinthecurrentstate |     |            |     | Alllegalstoneplacements |        |
| --- | ------- | --- | ------------------------------ | --- | ---------- | --- | ----------------------- | ------ |
|     | Actions |     |                                |     | Resign     |     |                         | Pass   |
|     |         |     |                                |     | Offeradraw |     |                         | Resign |
TransitionProbabilities Deterministic,{0,1} Deterministic,{0,1}
Rewards Finaloutcome,orshapedintermediaterewards Finaloutcome,orshapedintermediaterewards
| PerfectlyObservable |     |     |     |     | Yes        |     |     | Yes        |
| ------------------- | --- | --- | --- | --- | ---------- | --- | --- | ---------- |
|                     |     |     |     |     | Treesearch |     |     | Treesearch |
|                     |     |     |     |     | Self-play  |     |     | Self-play  |
Techniquesthatcouldbeused
|     |     |     |     | Functionapproximation |     |     | Functionapproximation |     |
| --- | --- | --- | --- | --------------------- | --- | --- | --------------------- | --- |
Pokemon
|     |       |       |     |       |     | Turn                  | number         |          |
| --- | ----- | ----- | --- | ----- | --- | --------------------- | -------------- | -------- |
|     |       |       |     |       |     | Remaining             | HP             |          |
|     |       |       |     |       |     | Status conditions     |                |          |
|     | State | Space |     |       |     | Active                | Pokemon        |          |
|     |       |       |     |       |     | Available team        | members        |          |
|     |       |       |     |       |     | Remaining             | PP             |          |
|     |       |       |     |       |     | Items and             | field effects  |          |
|     |       |       |     | Moves | of  | the pokemon available | with remaining | energies |
Resign
Actions
|     |     |     |     |     |     | Use an | Item    |     |
| --- | --- | --- | --- | --- | --- | ------ | ------- | --- |
|     |     |     |     |     |     | Change | Pokemon |     |
Transition Probabilities Not deterministic, probability of missing, some moves are stochastic
Rewards Win/Lose, Damage Inflicted, How effective is the move, Damage Taken
|     | Perfectly | Observable |     |     |     | No, it is partially | observable |     |
| --- | --------- | ---------- | --- | --- | --- | ------------------- | ---------- | --- |
65