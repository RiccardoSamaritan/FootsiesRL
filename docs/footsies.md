# FootsiesRL

---

## 1. The problem

The goal is to train an agent to play **Footsies**, a two-player fighting game, using a single algorithm (**PPO in self-play**) and to study **how the reward function changes what the agent learns**, measuring not only how much it wins but also how robust it is.

The guiding question is:

> **Is an agent that wins against simple opponents also hard to exploit?**

The FootsiesGym paper shows that the two can diverge: a method can beat the others head-to-head and still be the most exploitable. The project reproduces this analysis on a small scale by keeping the algorithm fixed and changing the reward, and connects it to three themes: **modeling** (POMDP, multi-agent), the **reward** as a specification of the objective, and the **bias–variance** trade-off of policy gradient estimators.

From an RL point of view the problem has four difficulties:

- **two agents** learning together: the outcome of an action also depends on what the other one chooses, and while one improves its policy the other changes its own. For this reason the environment, seen from a single agent, **is not stationary**;
- **partial observability**: the agent does not receive the full state but an observation that is not enough to act optimally; here the opponent's intentions are not visible and actions take effect with a delay;
- **sparse and delayed reward**: you only win or lose at the end of the match, which can come after thousands of steps;
- **mixed strategies**: in a game of mutual prediction a deterministic policy is exploitable by construction.

---

## 2. The game

### 2.1 What Footsies is

Footsies is a 2D fighting game: it is a **two-player, zero-sum, imperfect-information** game. It is built around the **neutral game**: the phase in which neither player has a clear advantage and both move, adjust spacing and timing, and try to create an opening to attack.

Fighting games have two strategic regimes. The **neutral game** is **non-transitive**: options beat each other in a cycle, as in rock-paper-scissors (if A beats B and B beats C, C can still beat A), so there is no best move in absolute terms and it pays to mix. **Combos** (long sequences of chained attacks after finding an opening) are instead an execution contest that is **largely transitive**: whoever executes better gains more against anyone, and players can be ranked along a single skill axis. Footsies removes combos and keeps only the first regime. This is why it is the right testbed for our question: in a non-transitive game no option dominates and **whoever commits to a pure strategy is easily exploited**, so "wins more against X" does not imply "is stronger".

In the project the game is used through the **FootsiesGym** library.

### 2.2 The rules

The key mechanics are:

1. **No health bar**: the round is lost when hit by a **special attack**.
2. **Guard**: opponent attacks can be **blocked up to three times** (guard bar at the top of the screen). After three blocks every attack causes a **guard break**, which leaves the player open to a follow-up attack. The guard **does not regenerate**: during a match it can only decrease and is refilled only on reset. The players' starting position is also always the same.
3. **Dash**: you dash forward or backward by pressing the direction **twice in quick succession**.
4. **Special attacks**: performed by **holding attack for 60 frames and then releasing it**, either standing still or with a forward/back direction.

There are four attack moves, with these **frame data** (60 frames = 1 second):

| Attack | Startup | Active | Recovery | How it is performed |
|---|---|---|---|---|
| `N_ATTACK` | 4 | 2 | 16 | normal attack |
| `B_ATTACK` | 3 | 3 | 15 | attack with direction |
| `N_SPECIAL` | 11 | 4 | 29 | charge >= 60 frames and release **without** direction; or press attack again while a normal attack is in progress (combination) |
| `B_SPECIAL` | 2 | 6 | 47 | charge >= 60 frames and release **with** forward or back |

- **Startup**: frames before the attack becomes active, i.e. the window in which it can be **blocked on reaction**.
- **Recovery**: frames afterwards, during which the attacker is vulnerable. This is the window for the **whiff punish**, i.e. punishing an attack that missed.

Both specials, if they land on a vulnerable opponent, **win the round**.

### 2.3 The agent's actions

At each step each player chooses one of **6 discrete actions**:

| Action | Meaning |
|---|---|
| `NONE` | do nothing |
| `BACK` | move back |
| `FORWARD` | move forward |
| `ATTACK` | attack |
| `BACK+ATTACK` | attack with back direction |
| `FORWARD+ATTACK` | attack with forward direction |

Actions are **player-relative** (forward/back relative to where the player is facing). `game_action_to_env_action` in `src/opponents.py` converts the game's absolute actions into these using the player's side, and this is what allows the **same network** to play both as P1 and as P2.

Dash and specials have no dedicated action: they emerge from **sequences** of actions. `B_SPECIAL` requires **15 consecutive steps** with an attack action (60 frames with `frame_skip` 4) followed by release on `FORWARD` or `BACK`. Under a uniform policy this sequence occurs with probability of about **0.001%**, and on top of that it only works if the opponent is vulnerable. This is why the library offers `use_special_charge_action`, which brings the actions to **9** by adding `BACK-CHARGE`, `NEUTRAL-CHARGE` and `FORWARD-CHARGE`. The first press starts the charge; the second releases it and performs the corresponding special.

### 2.4 Time: `frame_skip` and `action_delay`

- The game runs at **60 Hz**; one environment step corresponds to **4 frames** (`frame_skip = 4`).
- The action chosen at step $t$ is executed at step $t + \texttt{action\_delay}/\texttt{frame\_skip}$: with `action_delay = 8` that is **2 steps later**. The delay must be a multiple of `frame_skip`. The environment keeps a queue of pending actions, filled with `NONE` on reset, and puts the player's previous action in their observation.

The delay determines **which attacks can be handled on reaction and which must be predicted**:

- with **delay 0** the agent reacts almost immediately (only the delay due to `frame_skip` remains, up to 4 frames). The paper observes that in this case the game **degenerates into reactive, brittle strategies**: in its appendix, with PPO and 5 seeds, the delay-0 agent beats random more often than the delay-12 one, but collapses against no-op, while the delay-12 agent beats it almost always. "Degenerate" is the paper's judgment, based on this result and on visual inspection of the matches; we have not verified it yet;
- as the delay increases, moves become less reactable (see below): beyond about 12 frames no attack can be **blocked** on reaction and defense becomes predicting the opponent's intentions, but whiffed attacks can still be **punished** on reaction (normals up to about 12–16 frames, specials always). The paper also states that at very high delays the game tends towards randomness, but shows no data for values above 12.

The paper measures reactability as follows: after the delay, a stimulus can fall anywhere within the `frame_skip` window, so the reaction time ranges from $D$ to $D+4$ frames; this is compared with startup (to block) or recovery (to punish). Applying this rule to the frame data with $D = 8$ (reaction time 8–12 frames) gives the following picture: an attack is *always* reactable if even the worst case ($D+4$) fits in the window, *sometimes* if only the best case ($D$) fits, *never* otherwise. The calculation matches the $D = 8$ column of Figure 8 in the paper.

| Attack | Blockable on reaction (startup) | Punishable on whiff (recovery) |
|---|---|---|
| `N_ATTACK` | never (4 < 8) | always (16 >= 12) |
| `B_ATTACK` | never (3 < 8) | always (15 >= 12) |
| `N_SPECIAL` | sometimes (8 <= 11 < 12) | always (29) |
| `B_SPECIAL` | never (2 < 8) | always (47) |

With the default configuration, therefore, **attacks must be predicted, but mistakes can always be punished on reaction**; both specials have recoveries so long that they are punishable at any reasonable delay. This leads to a hypothesis of ours, not yet verified, for the observed passivity (§7): whiffing an attack is costly, waiting is not.

### 2.5 How a match ends

- **Win/loss**: a player is hit by a special (`is_dead`) -> episode *terminated*, $\pm w$.
- **Timeout**: after `max_t = 4000` steps → episode *truncated*, both receive 0. That is 4000 × 4 / 60 ≈ **267 seconds** of play. The timeout counts as a **draw**.

---

## 3. The environment

### 3.1 Configuration

The configuration is shared between training and evaluation:

| Parameter | Value | Notes |
|---|---|---|
| `frame_skip` | 4 | frames per step |
| `action_delay` | 8 | frames of action delay |
| `max_t` | 4000 | steps before timeout |
| `win_reward_scaling_coeff` | 10.0 | **must be set explicitly**: the library default is 1.0, the paper uses 10 |
| `guard_break_reward` | 0.0 | dense reward off (on only in R1) |
| `use_reward_budget` | False | on only in R1 |
| `headless` | True | no graphics |

The reward scale is not a detail: the paper's entropy coefficient (0.025) and value loss coefficient (1.0) are tuned for $w=10$; with $w=1$ the entropy bonus would weigh relatively 10 times more.

### 3.2 Observation

Each player's observation is a vector of **85 dimensions** (88 with the charge actions), built **from that player's point of view**. It contains the features needed to fully describe the state of **both** players, plus a small **privileged** block about oneself that does not appear in the opponent's block, like two people sitting at separate controllers:

| Block | Copies | Features (dimension) |
|---|---|---|
| **Common** | 1 | distance between the players $\lvert x_{p1} - x_{p2}\rvert$, normalized (1) |
| **Shared** | 2 (self and opponent) | position $x$ (1), velocity $x$ (1), `is_dead` (1), `vital_health` (1), **guard one-hot over {0,1,2,3} (4)**, current action one-hot over 17 scripts (17), frame of the current action, its duration and remaining frames (3), action end (1), always cancelable (1), hits landed by the action (1), hit stun frames and whether in hit stun (2), sprite shake and its maximum (2), facing right (1), current frame advantage (1) -> **37** |
| **Privileged** | 1 (self only) | next forward would dash (1), next back would dash (1), special charge progress in [0,1] (1), **previous action** one-hot over $\lvert\mathcal A\rvert$ (6), holding the charge (1) -> **10** |

### 3.3 The vectorized wrapper FootsiesVecEnv

`src/env.py` wraps FootsiesGym in vectorized mode:

- $N$ matches run **in a single Unity server** (`num_envs = N`, in training $N=48$);
- each match has two players, so the learner sees **$2N$ "slots"**: slots $[0, N)$ are the P1s of matches $0..N-1$, slots $[N, 2N)$ the P2s;
- **all slots use the same network**, each from its own perspective (**self-play**);
- `step()` returns observations, rewards, `terminated` and `truncated` of shape $(2N, \dots)$, collects per-episode statistics (length, truncated or not, outcome for P1, raw return) and keeps the step counter `t` of each match;
- it has a **`reward_fn` hook** that lets `src/rewards.py` replace the training reward without touching the algorithm; episode statistics always use the raw reward;
- it waits for the server to accept connections before the first `reset()`.

Two properties of the library in vectorized mode constrain the algorithm:

1. finished matches are **reset automatically** and the returned observation is already the **reset** one: the final observation is lost;
2. with `num_envs = 1` the library switches mode and output shape, so the wrapper requires $N \ge 2$.

### 3.4 Fixed opponents

| Opponent | Logic | What it measures |
|---|---|---|
| `random` | uniform action among the 6 | minimum competence threshold |
| `noop` | always `NONE` | whether the agent can **initiate** the fight |
| `back` | always `BACK` | whether the agent can **chase** and hit someone running away |
| `footsies_bot` | rule-based AI from the original game (BattleAI) rewritten in FootsiesGym, with some randomness | level against a benchmark also used by the paper |

`FootsiesBot` is not a network: it reads the **fight state** from the `info` (distance, damage, guard break, block, attacks, facing), `return_fight_state_in_infos=True`.

---

## 4. Problem formalization

### 4.1 The game as a multi-agent problem

Multi-agent Reinforcement Learning distinguishes three structures:

- **cooperative**: agents share the objective and maximize a common return;
- **competitive**: one agent's gain is the other's loss, as in zero-sum games;
- **mixed**: interests partly aligned and partly in conflict.

Footsies is a **competitive zero-sum** problem. With $n$ agents the evolution of the environment depends on the **joint action** $(a^1, \dots, a^n)$; here $n = 2$:

$$
s_{t+1} \sim p(\,\cdot \mid s_t, a^1_t, a^2_t), \qquad a^i_t \in \mathcal{A} = \{0,\dots,5\}.
$$

- **State** $s_t \in \mathcal{S}$: complete configuration of the game for both players, including the parts one player cannot see of the other: their privileged block (dash ready, charge, charge held) and their queue of actions already chosen but not yet executed.
- **Transitions**: applied by the Unity simulator.
- **Zero-sum reward** (competitive game):

$$
r^1_t = -\,r^2_t .
$$

With the terminal reward (R0):

$$
r^1_t =
\begin{cases}
+w & \text{if player 2 is defeated at step } t\\
-w & \text{if player 1 is defeated at step } t\\
0 & \text{otherwise (including at timeout)}
\end{cases}
\qquad w = 10 .
$$

Since the game is competitive, the quality of a policy **cannot be evaluated independently** of the opponent's: a strategy that is good against one opponent can be terrible against another.

### 4.2 A single agent's point of view: a POMDP

The problem can be tackled in two different ways:

- **independent learning**: each agent learns by treating the others as part of the environment. It is simple, but ignores that the environment changes because the others are also learning;
- **centralized training with decentralized execution**: during training extra information is used (observations or actions of the others), while at execution time each agent uses only its own.

The first approach is used, with a single network shared by both sides: neither the policy nor the critic sees opponent information beyond what is in their own observation. Fixing the opponent's policy $\pi^{-i}$ and treating it as part of the environment, what remains for agent $i$ is a **POMDP**.

**POMDP.** A *Partially Observable Markov Decision Process* generalizes the MDP to the case in which the agent does not directly observe the state $s_t$, but receives an observation $y_t$ that carries partial, and sometimes noisy, information about it. It is formalized with an **observation function** $f(y \mid s, a, s')$, the probability of observing $y$ after the transition to $s'$, so that the joint model is

$$
p(s', y \mid s, a) = p(s' \mid s, a)\, f(y \mid s, a, s').
$$

If the observation coincides with the state we are back to an MDP. In our case:

- **hidden state**: $s_t$;
- **observation**: $y^i_t = o^i(s_t)$, the 85-feature vector from $i$'s point of view. The observation function is deterministic but **not invertible**: different states give the same observation;
- **effective dynamics**: $p_{\pi^{-i}}(s' \mid s, a^i) = \sum_{a^{-i}} \pi^{-i}(a^{-i} \mid y^{-i})\, p(s' \mid s, a^i, a^{-i})$, i.e. the game dynamics averaged over the opponent's choices.

Why the observation is not enough (i.e. why it is not an MDP):

1. **the opponent's intentions**: their action already chosen but still queued because of `action_delay` (2 steps) is part of the state but not of $i$'s observation. With the default delay attacks cannot be blocked on reaction (§2.4), so they must be **predicted**;
2. the **opponent's privileged block** is not visible: in particular their special charge progress, which is precisely the threat that loses the round;
3. the opponent is itself a stochastic policy.

All the rest of both players' physical state (positions, guards, actions in progress, frame advantage) is instead in the observation: the imperfect information is concentrated on **intentions** and **charge**, not on position.

**What would be needed to act optimally.** In a POMDP the current observation is usually not enough, and the agent must retain information from the past. The simplest solution is a **perfect memory**, which records the whole history of actions and observations,

$$
m_0 = \varnothing, \qquad m_t = a_0\, y_1\, a_1\, y_2 \cdots a_{t-1}\, y_t ,
$$

and the policy becomes $\pi(a \mid m)$. The pair (hidden state, memory) evolves in a Markovian way, and the memory is an **information state**: it contains everything needed to update knowledge about the hidden state. Since the history grows over time, one uses the **belief**, i.e. the probability distribution over hidden states given the history:

$$
b_t(s) = \mathbb{P}(S_t = s \mid H_t),
$$

updated with Bayes' rule after action $a$ and observation $y$:

$$
b'(s') = \frac{\sum_s p(s', y \mid s, a)\, b(s)}{\sum_s p(y \mid s, a)\, b(s)} .
$$

The numerator is the probability of reaching $s'$ and observing $y$; the denominator normalizes. The belief is a sufficient statistic of the history: the POMDP becomes an MDP over beliefs (*belief-MDP*), with its own Bellman equation. However, the belief space is continuous even when states are finite, so exact dynamic programming is generally intractable and approximations are used.

In the case of Footsies neither memory nor belief is handled: the policy depends only on the current observation,

$$
\pi_\theta(a \mid y_t),
$$

and the only "memory" is what the environment puts in the observation (the previous queued action). There is therefore this limitation: approximation makes RL applicable to partially observable problems, but **it cannot add the memory of past observations to the state**.

### 4.3 Episodes, return, termination vs truncation

**Episodes and return.** A task is **episodic** when the interaction naturally splits into sequences, the episodes, each ended by a **terminal state** followed by a reset. A Footsies match is an episode. The agent wants to maximize the expected **discounted return**

$$
G_t = \sum_{k=t+1}^{T} \gamma^{\,k-t-1} R_k = R_{t+1} + \gamma\, G_{t+1}, \qquad \gamma = 0.995 ,
$$

where the second form is the recursive relation between successive returns. To unify finite and infinite episodes, the end of the episode is treated as entering an **absorbing state**, which transitions only to itself with zero reward.

**The discount factor.** $\gamma$ is a **survival probability**: at each step the agent "survives" with probability $\gamma$ and with probability $1-\gamma$ ends up in an absorbing state; the expected survival time is $T_{\text{eff}} = 1/(1-\gamma)$. With $\gamma = 0.995$:

- **effective horizon**: $1/(1-\gamma) = 200$ steps;
- **credit assignment**: $0.995^{920} \approx 0.01$, so an action taken more than about 920 steps before the end weighs less than 1% on the return. With episodes of up to 4000 steps and only a final reward, this is the core of the difficulty.

**Value functions.** These are needed later for the critic and the advantage. The **value of a state** under policy $\pi$ is the expected return starting from $s$ and following $\pi$; the **value of an action** is the expected return starting from $s$, taking $a$ and then following $\pi$:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s], \qquad q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a].
$$

From the recursion of the return follows the **Bellman equation** $v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a)\,[\,r + \gamma v_\pi(s')\,]$, which links the value of a state to that of its successors.

**Termination and truncation.** The two ways an episode can end must be treated differently:

| Event | Meaning | Value of the next state |
|---|---|---|
| `terminated` (defeat) | true terminal state (absorbing state with zero reward) | $V(s_{t+1}) = 0$ |
| `truncated` (timeout at 4000) | the game is **not** over, the budget has run out | bootstrap: $V(s_{t+1})$ estimated by the critic |

Treating the timeout as terminal would introduce a bias, because the state is not really terminal: its value is not zero. Since the final observation is lost with auto-reset (§3.3), the code approximates $V(s_{t+1}) \approx V(s_t)$: this is a small bias, much smaller than setting the value to zero.

### 4.4 Objective: self-play, non-stationarity, exploitability

With a single network for both sides the training objective is

$$
J(\theta) = \mathbb{E}_{\tau \sim (\pi_\theta,\, \pi_\theta)}\!\left[ G_0 \right],
$$

where the trajectory is generated by the same policy on both sides, each from its own point of view. The consequences:

- the "environment" seen from one side contains the policy of the other, which **changes** at every update -> the problem is **non-stationary**;
- in a symmetric zero-sum game the expected outcome of self-play is roughly zero: $J(\theta)$ does **not** measure how strong one is. External metrics are needed (§6).

In competitive settings an important goal is to learn policies that are **hard to exploit**. A **best response** to $\pi$ is the policy that achieves the best result against it:

$$
\mathrm{BR}(\pi) \in \arg\max_{\pi'} \; \mathbb{E}_{\tau \sim (\pi',\, \pi)}\big[\text{outcome for } \pi'\big],
$$

and the **exploitability** of $\pi$ is the value a best response achieves against it. In the project the best response is **approximated** by training a PPO against the frozen policy; being approximate, it gives a **lower bound** on the true exploitability.

### 4.5 The four reward functions

#### R0 — terminal (baseline)

$$
r_t =
\begin{cases}
\pm 10 & \text{at the step in which a player is defeated}\\
0 & \text{at all other steps and at timeout}
\end{cases}
$$

It is the environment's default reward and the one used in the paper.

**Why it is hard to learn: sparse signal, hence high variance.** In a match won at step $T$, the return from a step $t$ is

$$
G_t = 10\,\gamma^{\,T-t-1},
$$

and it equals $-10\,\gamma^{\,T-t-1}$ in a loss and $0$ in a draw. Two problems follow.

- **Variance.** For the same state, $G_t$ can be positive, negative or zero depending on **all** subsequent actions of both players, up to thousands of steps later. $G_t$ depends on all future rewards of the trajectory. With a single nonzero reward per match, almost every step has zero reward and all the information lies in the final outcome.
- **Credit assignment.** With $\gamma = 0.995$, an action more than about 920 steps from the end receives less than 1% of the reward (§4.3): actions far from the end are rewarded or punished almost exclusively through the critic's estimate.

#### R1 — dense guard reward with budget (library option)

- **Dense reward.** Every time a player's guard drops by one point, the attacker receives $+x = +2$ and the blocker $-2$. The sum stays zero.
- **Budget.** The shaping received is **subtracted from the win reward**: the guard has 3 points, so at most $3x = 6$ can be collected before the end, and the constraint $3x < w$ guarantees that the final reward stays positive. The total (undiscounted) return of a win remains $w = 10$.