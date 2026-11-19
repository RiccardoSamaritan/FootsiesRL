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

Dash and specials have no dedicated action: they emerge from **sequences** of actions. `B_SPECIAL` requires **15 consecutive steps** with an attack action (60 frames with `frame_skip` 4) followed by release on `FORWARD` or `BACK`. Under a uniform policy this sequence occurs with probability of about **0.001%**, and on top of that it only works if the opponent is vulnerable. This is why the library offers `use_special_charge_action`, which brings the actions to **9** by adding `BACK-CHARGE`, `NEUTRAL-CHARGE` and `FORWARD-CHARGE`. The first press starts the charge; the second releases it and performs the corresponding special. In training it is enabled with `--special-charge` (observation of 88 dimensions, §3.2); it is off by default and used only as an ablation (§7).

### 2.4 Time: `frame_skip` and `action_delay`

- The game runs at **60 Hz**; one environment step corresponds to **4 frames** (`frame_skip = 4`).
- The action chosen at step $t$ is executed at step $t + \texttt{action\_delay}/\texttt{frame\_skip}$: with `action_delay = 8` that is **2 steps later**. The delay must be a multiple of `frame_skip`. The environment keeps a queue of pending actions, filled with `NONE` on reset, and puts the player's previous action in their observation.

The delay determines **which attacks can be handled on reaction and which must be predicted**:

- with **delay 0** the agent reacts almost immediately (only the delay due to `frame_skip` remains, up to 4 frames). The paper observes that in this case the game **degenerates into reactive, brittle strategies**: in its appendix, with PPO and 5 seeds, the delay-0 agent beats random more often than the delay-12 one, but collapses against no-op, while the delay-12 agent beats it almost always. "Degenerate" is the paper's judgment, based on this result and on visual inspection of the matches. Our one-seed ablation (§8.5) agrees on no-op (delay 0 never beats it, delay 12 always does) but not on random (delay 0 wins 64%, delay 12 70%), and cannot isolate the effect of the delay;
- as the delay increases, moves become less reactable (see below): beyond about 12 frames no attack can be **blocked** on reaction and defense becomes predicting the opponent's intentions, but whiffed attacks can still be **punished** on reaction (normals up to about 12–16 frames, specials always). The paper also states that at very high delays the game tends towards randomness, but shows no data for values above 12.

The paper measures reactability as follows: after the delay, a stimulus can fall anywhere within the `frame_skip` window, so the reaction time ranges from $D$ to $D+4$ frames; this is compared with startup (to block) or recovery (to punish). Applying this rule to the frame data with $D = 8$ (reaction time 8–12 frames) gives the following picture: an attack is *always* reactable if even the worst case ($D+4$) fits in the window, *sometimes* if only the best case ($D$) fits, *never* otherwise. The calculation matches the $D = 8$ column of Figure 8 in the paper.

| Attack | Blockable on reaction (startup) | Punishable on whiff (recovery) |
|---|---|---|
| `N_ATTACK` | never (4 < 8) | always (16 >= 12) |
| `B_ATTACK` | never (3 < 8) | always (15 >= 12) |
| `N_SPECIAL` | sometimes (8 <= 11 < 12) | always (29) |
| `B_SPECIAL` | never (2 < 8) | always (47) |

With the default configuration, therefore, **attacks must be predicted, but mistakes can always be punished on reaction**; both specials have recoveries so long that they are punishable at any reasonable delay. This leads to a hypothesis of ours, not verified, for the observed passivity (§9.1): whiffing an attack is costly, waiting is not.

### 2.5 How a match ends

- **Win/loss**: a player is hit by a special (`is_dead`) -> episode *terminated*, $\pm w$.
- **Timeout**: after `max_t = 4000` steps -> episode *truncated*, both receive 0. That is 4000 × 4 / 60 ≈ **267 seconds** of play. The timeout counts as a **draw**.

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

**Game settings of a run.** Two options can be changed from the command line for the ablations: `--action-delay` (default 8) and `--special-charge` (default off). They are saved in `config.json` (key `"game"`) and in every checkpoint (`env_config`), so that `evaluate.py`, `head_to_head.py` and `best_response.py` rebuild the environment with the **same rules** used in training. Agents trained with different settings cannot play each other: the evaluation stops with an error. Older checkpoints have no `env_config` and use the default settings.

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

and the **exploitability** of $\pi$ is the value a best response achieves against it. In the project the best response is **approximated** by training a PPO against the frozen policy (§6.3); being approximate, it gives a **lower bound** on the true exploitability.

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
- **`r1_floor`.** It is not a new reward: it is R1 trained with `--adv-std-floor`, a fix for the stalls described in §5.4. The same fix is also applied to R0 (`r0_floor`, §7).

#### R2 — potential-based shaping

The idea is to provide the same kind of information as R1 (the guard) **without changing the objective**. We define an antisymmetric potential on the guard difference,

$$
\Phi(s) = \frac{6}{3}\,\big(g_{\text{self}}(s) - g_{\text{opp}}(s)\big) = 2\,\big(g_{\text{self}}(s) - g_{\text{opp}}(s)\big) \in [-6, 6],
$$

where $g \in \{0,1,2,3\}$ is the guard read from the observation. The factor $\frac{6}{3}$ follows R1's scale: the shaping budget is 6 and there are 3 guard points, so each point is worth 2, as in R1. The **potential difference** is added to the reward:

$$
\tilde r_t = r_t + \underbrace{\gamma\,\Phi(s_{t+1}) - \Phi(s_t)}_{F_t},
$$

with $\Phi = 0$ in the terminal state. Both players' guard is in the observation (§3.2), so $\Phi$ is computed exactly. The only approximation is at truncation, where the final observation is lost and $\Phi(s_{t+1}) \approx \Phi(s_t)$ is used. It is wrong only if a guard point is lost exactly on the last step, so the error is negligible.

$\Phi$ is zero at the start (full guards) and changes sign when the players are swapped, so the shaping **remains zero-sum**. Since the guard does not regenerate (§2.2), $\Phi$ changes only when a player loses a guard point: a guard advantage is permanent until the end of the match.

**Example.** The opponent blocks one of my attacks and their guard goes from 3 to 2.

| Moment | $\Phi$ on my side | Shaping $F_t$ for me |
|---|---|---|
| the block | from 0 to +2 | $\gamma \cdot 2 - 0 \approx +1.99$ |
| each following step, as long as nothing changes | stays +2 | $(\gamma - 1)\cdot 2 = -0.01$ |
| the end of the match (termination) | from +2 to 0 | $0 - 2 = -2$ |

The bonus for the block is therefore **an advance payment**: it arrives immediately, when the event happens, and is paid back over time and at the end. The opponent receives exactly the opposite.

**What changes compared to R0.** The optimal policy does not, but the **learning path** does. The signal arrives close to the event that causes it, which can reduce variance. In exchange, the critic must learn a different function, $v_\pi - \Phi$, and with finite samples and self-play PPO may behave differently than with R0.

#### R3 — time-scaled win

The terminal reward is multiplied by a factor that decreases linearly from 1 to 0.5 over the horizon:

$$
\tilde r_t = r_t \cdot f(t), \qquad f(t) = 1 - 0.5\,\min\!\big(1,\; t / t_{\max}\big), \qquad t_{\max} = 4000,
$$

only if the episode terminates at step $t$; draws remain at 0.

**What changes compared to R0.**

- **Incentive for speed.** Discounting already makes winning earlier preferable, but only **relative to the current state**: what matters is how many steps remain until the end from now. R3 adds pressure on the **absolute time** of the match: the same win is worth less if the match is already long.
- **Symmetry of incentives.** For the losing side, $-10 \cdot f(t)$ is less negative if the loss comes late. R3 therefore also rewards whoever **delays** the loss, not only whoever speeds up the win.
- **A reward that depends on something the agent cannot see.** The factor depends on the step counter $t$, which is not in the observation (§3.2). The same observed state is therefore worth different things depending on how long the match has lasted: the critic has to guess the elapsed time, which adds partial observability and hence variance or bias in the value estimate.
- **Risk.** The agent may prefer quick, risky matches.

#### Comparison

| Variant | When the signal arrives | Does it change the optimal policy? | Expected effect on variance |
|---|---|---|---|
| R0 | only at the end of the match | no (it is the true objective) | high |
| R1 | at each guard point removed, plus the rest at the end | yes: it rewards a sub-goal | lower |
| R2 | at each guard point removed, then "paid back" | no (advantage unchanged) | lower, if the potential is informative |
| R3 | only at the end of the match, scaled by time | yes: it rewards quick wins | similar to R0, plus the effect of unobserved time |

---

## 5. How it was tackled: PPO

### 5.1 Why a policy gradient method

- The policy must be **stochastic** (§4.2), and policy gradient methods parametrize it directly. For each state–action pair a numerical **preference** $h(s, a, \theta)$ is computed, which can be produced by a neural network, and the probabilities are obtained with a **softmax**:

$$
\pi(a \mid s, \theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a'} e^{h(s,a',\theta)}} .
$$

  The actor outputs six preferences (the *logits*) and the policy is the corresponding categorical distribution. Every action always has positive probability, and the policy can stay as mixed as needed.
- Value-based methods such as Q-learning and DQN instead produce **greedy**, i.e. deterministic, policies, whereas here a mixed policy is needed. They also suffer from *maximization bias*.

### 5.2 PPO

**The policy gradient theorem.** The quality of a policy is measured by the value of the initial state, $J(\theta) = v_{\pi_\theta}(s_0)$. Computing its gradient looks hard, because changing $\theta$ changes both the actions chosen and the states visited, and the second effect depends on the environment, which we do not know. The theorem shows that there is no need to differentiate the state distribution:

$$
\nabla J(\theta) \propto \sum_s \mu(s) \sum_a q_\pi(s,a)\, \nabla \pi(a \mid s, \theta),
$$

where $\mu(s)$ indicates how often the policy visits state $s$. Using $\nabla \pi = \pi\, \nabla \log \pi$, the gradient becomes an average over the policy's trajectories, so it can be estimated by playing:

$$
\nabla J(\theta) \propto \mathbb{E}_{\pi}\big[\, q_\pi(S_t, A_t)\, \nabla_\theta \log \pi_\theta(A_t \mid S_t) \,\big].
$$

As in the paper, we use $\gamma = 0.995$.

**REINFORCE.** The value $q_\pi$ is unknown, but on average it equals the observed return $G_t$, which can be used in its place:

$$
\theta \leftarrow \theta + \alpha\, G_t\, \nabla \log \pi_\theta(A_t \mid S_t).
$$

In practice: if the return is high, the chosen action becomes more likely. The estimate is correct on average, but **very noisy**, because $G_t$ depends on everything that happens up to 4000 steps later (§4.5, R0). For this reason we do not use it.

PPO is built by improving REINFORCE in three steps:

1. **Baseline**. A value $b(S_t)$ that depends only on the state is subtracted from the return. The mean of the gradient does not change (the added term has zero mean, because probabilities always sum to 1), but the noise decreases. Usually $b(s) = v_\pi(s)$ is chosen, estimated by a network $\hat v(s, w)$.
2. **Advantage**. With this baseline the advantage $A_\pi(s,a) = q_\pi(s,a) - v_\pi(s)$ appears, i.e. how much better action $a$ is than the policy's average in $s$:

$$
\nabla J \propto \mathbb{E}_\pi\big[A_\pi(S_t,A_t)\,\nabla\log\pi_\theta(A_t\mid S_t)\big].
$$

   Actions better than average become more likely, worse ones less likely.
3. **Actor-critic**. The **actor** is the policy $\pi_\theta$, the **critic** is the value estimate $\hat v(s, w)$. Instead of the full return, the one-step TD error is used,

$$
\delta_t = R_{t+1} + \gamma \hat v(S_{t+1}, w) - \hat v(S_t, w),
$$

   which approximates the advantage if the critic is accurate. The critic is updated with $w \leftarrow w + \alpha_w\, \delta_t\, \nabla_w \hat v(S_t, w)$, the actor with $\theta \leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \log \pi_\theta(A_t \mid S_t)$. The noise is lower than in REINFORCE, but the critic's errors introduce a **bias**.

The critic is pushed towards a target that is computed with the critic itself. The target therefore depends on $w$, but this dependence is ignored and only part of the gradient is followed. In our code too, the target is computed without gradient.

### 5.3 GAE: a middle ground between TD and Monte Carlo

GAE (*Generalized Advantage Estimation*) combines three ideas already seen: the advantage, the actor-critic TD error and $n$-step returns.

**The problem.** The gradient requires the advantage $A_\pi(s,a)$, which we do not know and must estimate. The two estimates sit at opposite extremes:

| Estimate | Formula | Bias | Variance |
|---|---|---|---|
| Monte Carlo with baseline | $G_t - V(s_t)$ | none | high: sums all rewards until the end of the match |
| One-step TD error | $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | yes, if the critic is wrong | low: uses a single reward |

**$n$-step returns.** Between the two extremes one can use $n$ real rewards and then rely on the critic for the rest:

$$
G_{t:t+n} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}).
$$

The corresponding advantage, $A_t^{(n)} = G_{t:t+n} - V(s_t)$, is the TD error for $n = 1$ and the Monte Carlo estimate when $n$ reaches the end of the episode. It can also be written as a sum of successive TD errors (the intermediate values cancel out):

$$
A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}.
$$

Increasing $n$ reduces the bias, because the critic's error weighs at most $\gamma^n$ times (*error reduction property*); but each additional real step adds noise. This is why the best methods often lie in between.

**GAE.** Instead of choosing a single $n$, GAE averages all the $A_t^{(n)}$, with weights that decay geometrically according to $\lambda$. The result has a simple form:

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n \ge 1} \lambda^{n-1} A_t^{(n)} = \sum_{k \ge 0} (\gamma\lambda)^k\, \delta_{t+k}.
$$

(Each $\delta_{t+k}$ appears in all the $A^{(n)}$ with $n > k$, and their weights sum to $\lambda^k$.)

- $\lambda = 0$: only $\delta_t$ remains, i.e. one-step actor-critic (low variance, critic bias);
- $\lambda = 1$: we get $G_t - V(s_t)$, i.e. REINFORCE with baseline (no bias, high variance);
- intermediate values: a trade-off controlled by a single number (in the project $\lambda = 0.95$).

**Example.** The last three steps before a win, with $\gamma = 1$. The critic estimates $V = 2, 4, 6$, the rewards are $0, 0, +10$, then the terminal state ($V = 0$). The TD errors are $\delta = 2, 2, 4$.

| Step | $\lambda = 0$ (TD) | $\lambda = 0.5$ | $\lambda = 1$ (Monte Carlo) |
|---|---|---|---|
| 1 | 2 | $2 + 0.5 \cdot 4 = 4$ | $10 - 2 = 8$ |
| 2 | 2 | $2 + 0.5 \cdot 4 = 4$ | $10 - 4 = 6$ |
| 3 | 4 | 4 | $10 - 6 = 4$ |

With $\lambda = 0$ the first steps "see" the win only through the critic; with $\lambda = 1$ they see it directly, but in a real match $G_t$ would carry all the noise of the subsequent actions.

**The project's numbers.** With $\gamma = 0.995$ and $\lambda = 0.95$, $\gamma\lambda \approx 0.945$: GAE effectively looks about 18 steps ahead (about 73 frames, just over one second of play). The rollout is 64 steps long and beyond that $V$ of the last observation is used; the lost weight is about $0.03$, so the cut matters little. The important consequence is that a win up to 4000 steps away reaches the early actions **only through the critic**: GAE reduces variance, but long-term credit assignment depends on how well $V$ has learned.

**In the code** the computation runs backwards over the rollout, handling episode ends as in §4.3 (`compute_gae` in `src/ppo.py`):

$$
\hat V_{t+1} =
\begin{cases}
0 & \text{terminated}\\
V(s_t) & \text{truncated}\\
V(s_{t+1}) & \text{otherwise}
\end{cases}
\qquad
\delta_t = r_t + \gamma \hat V_{t+1} - V(s_t),
\qquad
A_t = \delta_t + \gamma\lambda\,(1-d_t)\,A_{t+1},
$$

where $d_t = 1$ if the episode ends at $t$: there the sum stops, because the next step belongs to a new episode. The critic's target is $R_t = A_t + V(s_t)$.

### 5.4 The PPO loss

Clipping is the central idea of PPO (*Proximal Policy Optimization*, Schulman et al., 2017). It is built from the policy gradient and importance sampling.

**The problem: reusing data.** Each update collects 6144 samples (§5.6). Using them for a single gradient step would be wasteful, so PPO reuses them for 8 epochs × 8 minibatches = **64 gradient steps**. But the policy gradient is valid only for data generated by the **current** policy, and after the first step the policy has already changed: the data come from $\pi_{\theta_{\text{old}}}$, while we want to improve $\pi_\theta$.

This is an **off-policy** situation: the policy that generates the data (*behavior policy*) differs from the one we want to improve (*target policy*). It is handled with **importance sampling**, which estimates an expectation under one distribution using samples from another, weighting each sample by the ratio of the probabilities. Every action the new policy could take must have nonzero probability under the old one (*coverage*). Here this is guaranteed, because the policy is a softmax (§5.1).

**The importance sampling ratio.**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid y_t)}{\pi_{\theta_{\text{old}}}(a_t \mid y_t)}
$$

Weighting the advantage by $\rho_t$ gives the surrogate objective

$$
L^{\mathrm{IS}}(\theta) = \mathbb{E}_t\big[\rho_t(\theta)\,\hat A_t\big].
$$

This objective corrects the choice of actions and, at $\theta = \theta_{\text{old}}$, has exactly the policy gradient (by the log-derivative trick).

**Why it is not enough.** The ratio corrects the **actions**, but the **states** remain those visited by the old policy, so the approximation holds only as long as $\pi_\theta$ stays close to $\pi_{\theta_{\text{old}}}$. Moreover, off-policy methods tend to have high variance and converge slowly. Maximizing $L^{\mathrm{IS}}$ without limits makes the ratios grow, and a single update can ruin the policy.

**The solution: clipping the ratio.**

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat A_t,\; \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\,\hat A_t\big)\Big], \qquad \epsilon = 0.3 .
$$

| Advantage | The gradient wants to | Clipping |
|---|---|---|
| $\hat A_t > 0$ (action better than expected) | increase $\rho_t$ | above $1.3$ it stops rewarding: zero gradient |
| $\hat A_t < 0$ (action worse than expected) | decrease $\rho_t$ | below $0.7$ it stops rewarding: zero gradient |

The $\min$ makes the objective **pessimistic**: the clip only kicks in when the policy has already moved more than 30% in the right direction. If it has moved in the wrong direction (e.g. $\hat A_t > 0$ but $\rho_t < 0.7$), there is no clipping and the gradient can correct it.

In short, within an update no sample pushes the probability of its own action to change by more than 30%. The limit is **soft**, though: it removes the incentive but does not forbid it, and with 64 steps the policy can still drift away. The price is a **bias**, because a surrogate is optimized instead of the true objective. In exchange we get stable updates and data used 8 times instead of once.

**The full loss.** The critic and the entropy bonus are added to the policy term:

$$
\mathcal{L}(\theta) =
\underbrace{-\,L^{\mathrm{CLIP}}(\theta)}_{\text{policy}}
+ c_v\,\underbrace{\tfrac12\,\mathbb{E}_t\big[(V_\theta(y_t) - R_t)^2\big]}_{\text{critic}}
- c_e\,\underbrace{\mathbb{E}_t\big[\mathcal{H}(\pi_\theta(\cdot \mid y_t))\big]}_{\text{entropy}}.
$$

- **Policy**: it appears with a minus sign because the loss is minimized.
- **Critic**: squared error towards the GAE target $R_t$, i.e. the semi-gradient update of §5.2 written as a loss.
- **Entropy**: a **deliberate bias** towards exploration, which rewards policies that spread probability over more actions.

The coefficient values are in §5.5.

**Advantage normalization and stalls.** In each minibatch the advantages are normalized, $\hat A \leftarrow (\hat A - \text{mean}) / \text{std}$, so that the size of the update does not depend on the reward scale. This has a side effect when matches stop being decided (a **stall**): the reward is almost always 0, the advantages are tiny (std ~$10^{-3}$) and dividing by their std blows **noise** up to scale 1. Meanwhile the entropy bonus pushes the policy towards uniform, and the learning rate going to 0 prevents escaping the last stall.

The fix (`--adv-std-floor`) divides by $\max(\text{std}, \text{floor})$ instead: small advantages stay small. With floor = 0 (default) nothing changes, so the baselines are untouched. It was first tried on one seed (`r1_floor_s1`: stalled matches from 30% to 4%) and then trained with floor 0.1 on 5 seeds for both R0 (`r0_floor_s1..5`) and R1 (`r1_floor_s1..5`) with `slurm/floor_array.sbatch`, compared seed by seed with `r0_s*` and `r1_s*`. The result is in §8.6: the final stalls go from 2 runs out of 8 to 0 out of 9 and the spread between seeds drops, while the mean strength does not change measurably. Limit: it removes the noise amplification, not the passivity against the fixed opponents (§8.2, §9.1). The same option is available for the best-response learner (§6.3).

### 5.5 Network and hyperparameters

Two **separate** MLPs (no shared features), 2×256 ReLU (the same architecture used for all algorithms in the paper), orthogonal initialization (gain $\sqrt2$, 0.01 on the policy output, 1.0 on the value output), Adam with $\varepsilon = 10^{-5}$. Hyperparameters from Table 3 of the paper:

| Hyperparameter | Value |
|---|---|
| $\gamma$ | 0.995 |
| GAE $\lambda$ | 0.95 |
| clip $\epsilon$ | 0.3 |
| value loss coeff. | 1.0 |
| entropy coeff. | 0.025 (fixed); 0.1 -> 0 linearly in PPO (Sched.) |
| epochs × minibatches | 8 × 8 |
| learning rate | $10^{-4}$, linearly decaying to 0 |
| max gradient norm | 0.5 |
| parallel matches × rollout | 48 × 64 |
| value clipping | off (returns on a ±10 scale) |

As in the paper there are **two PPO baselines**, both with the terminal reward R0: PPO with fixed entropy (0.025) and **PPO (Sched.)**, where the coefficient decays linearly from 0.1 to 0 (`--ent-decay`). The reward variants R1–R3 all use fixed entropy, so the reward is the only thing that changes.

### 5.6 The training loop

For each update:

1. **rollout**: 64 steps over 48 matches -> $48 \times 64 = 3072$ game steps, i.e. **6144 samples** (two players per match), all generated by the same network;
2. **GAE** over the whole rollout, bootstrapping on the value of the last observation;
3. **PPO**: 8 epochs × 8 minibatches of 768 samples;
4. **logging** of one row per update in `runs/<run>/log.csv` (see below);
5. **checkpoint** every 1M steps; checkpoints also store the state of the Adam optimizer, so that an interrupted run can be resumed (see below).

**What is logged.** Besides times, learning rate and entropy coefficient:

- **episodes**: mean length `ep_len` and share of timeouts `timeout_rate`;
- **self-play outcomes**: in self-play the mean return is ~0 by symmetry, so it says nothing. We log instead the wins per side (`p1_win_rate`, `p2_win_rate`), the share of decided matches `decided_rate` (1 − draws), the mean $\lvert$raw return$\rvert$ `abs_return` (about 10 for decided matches, 0 for draws) and the mean $\lvert$training reward$\rvert$ per step `train_abs_reward`. These columns were added after the first runs: older logs (`r0_s1`, `r1_s1`, `r2_s1`) do not have them, and their self-play figure is simply empty;
- **bias/variance diagnostics**: explained variance of the critic `explained_var` (how much of the return $V$ explains: low means a biased critic), std of the advantages `adv_std` and of the returns `return_std` (variance), `approx_kl` (how far an update moved the policy).

**Budget and seeds.** Every run has **20M steps** (6510 updates, last checkpoint at 19,998,720 steps), because of the 2-hour per-job limit on the Orfeo cluster. The five variants of the comparison (R0, R1, R2, R3, R0 Sched.) have **5 seeds** each (1–5, the same for all, so the comparison is paired), run with `slurm/seeds_array.sbatch`. The other arrays:

| Sbatch | Runs | Status |
|---|---|---|
| `slurm/floor_array.sbatch` | 9: `r0_floor_s1..5`, `r1_floor_s2..5` (plus the earlier `r1_floor_s1`) | done |
| `slurm/ablations_array.sbatch` | 6: λ 0 and 1, γ 0.99, charge, delay 0 and 12 (all seed 1) | done |
| `slurm/br_array.sbatch` | 5 best responses (seed 1 of R0, Sched., R1, R2, R3), plus 5 with `SEED=2` | done |
| `slurm/diagnostics_array.sbatch` | behaviour diagnostics of 9 checkpoints (§6.4) | done |
| `slurm/h2h_array.sbatch` | 8 shards per pool (`final`, `time`) | `final` done, `time` not played |

Orfeo allows at most 20 jobs in the queue and 5 running, so each array goes in waves of 5 jobs of about 1.5 h. One best response failed twice on an I/O error of the shared filesystem (log not written); the code now ignores that write error.

**Resuming a run.** A run stopped by the time limit can be continued with `--resume` (in `train.py` and `best_response.py`; `RESUME=1` in `br_array.sbatch`), with the same arguments as the original run. The latest `ckpt_*.pt` is loaded, `log.csv` is cut after that checkpoint and training continues from the next update. Checkpoints written before this option existed have no optimizer state: Adam then restarts with empty moment estimates, a small deviation from an uninterrupted run, which the script prints.

**Compute used.** S = one 20M-step PPO run.

| Group | Runs | Cost |
|---|---|---|
| Reward comparison: R0, R0 Sched., R1, R2, R3, 5 seeds each | 25 | 25 S |
| Stall fix: R0 floor and R1 floor, 5 seeds each | 10 | 10 S |
| Ablations ($\lambda$ 0 and 1, $\gamma$ 0.99, charge, delay 0 and 12), 1 seed | 6 | 6 S |
| Best responses: 5 targets, 2 learner seeds | 10 | 10 S |
| **Total completed** | **51** | **51 S (1.02 billion steps)** |

Evaluations, head-to-head and diagnostics take a few minutes per model.

Speeds measured on the full PPO: ~6700 steps/s on the Mac, ~4050 on an Orfeo THIN node with 4 threads; time is split between network updates (56–63%) and the simulator (25–37%). Real 20M-step runs on THIN took 85–100 minutes.

---

## 6. Evaluation

- the agent plays $n$ matches against each opponent, **half as P1 and half as P2**;
- actions are **sampled** from the policy, never argmax (§4.2);
- the outcome is the **sign of the terminal reward** (+1 win, 0 draw = timeout, −1 loss);
- for checkpoints, random, noop and back the vectorized environment is used, with the same number of matches per slot: stopping at the first $n$ finished matches **would overestimate quick wins**;
- FootsiesBot plays one match at a time (§3.4);
- the checkpoints of every run are evaluated at about 5M, 10M, 15M steps and at the end, with $n = 100$ matches per opponent (`results/eval_<run>.csv`); FootsiesBot is evaluated on the final checkpoint only;
- the win rate has a **95% Wilson interval**, well-behaved even at 0% and 100%:

$$
\tilde p = \frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}, \qquad
h = \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1.96 .
$$

With $n = 100$ the half-width $h$ is at most about 0.10 (at $\hat p = 0.5$) and about 0.04 at 0% or 100%. Draws count as non-wins, so their number is always reported as well.

### 6.1 Head-to-head

`src/head_to_head.py` makes checkpoints play each other: each pair plays $n = 100$ matches (default), half per side, both sampling from their policy. The **score** of A against B is

$$
\text{score}_{A} = \frac{W + 0.5\,D}{n},
$$

so 0.5 means even and a draw (timeout, both passive) counts half. The output is a matrix (score of the row agent against the column agent) and, when several seeds of the same variant are present, the matrix averaged per variant: there the diagonal compares different seeds of the same variant, i.e. the spread between training runs. A ranking (mean score against all the others, `results/h2h_<pool>_ranking.csv`) is also saved, but self-play is **not transitive** (A can beat B, B beat C and C beat A), so the matrix is what matters.

**Pools and sharding.** `slurm/h2h_array.sbatch` plays a pool of agents split into **8 shards** (one job each, `--shard K/8`): all pairs are numbered in a fixed order and shard $k$ plays those with number mod 8 = $k$. Each shard writes `results/h2h_<pool>.s<k>of8.csv` and skips the pairs already in any file of the pool, so a shard that hits the time limit can be resubmitted as it is. When all shards are done, `head_to_head.py --label <pool> --plot-only` merges them and draws the figure (no game server needed). Two pools are defined:

- **`final`**: the final checkpoint of every comparison run (R0, R0 Sched., R1, R2, R3 × seeds 1–5) plus `r1_floor_s1` = 26 agents, 325 pairs;
- **`time`**: seed 1 of each variant at about 5M, 10M, 15M steps and final = 24 agents, 276 pairs. It would show how each variant progresses and who beats whom **at the same stage of training**; it has not been played.

### 6.2 Several seeds

`plots/plot_results.py` groups runs by variant (the name without `_s<seed>`). A variant with several seeds is drawn as the **mean over seeds with a min–max band**; a single run is drawn as it is, with its Wilson interval. `results/summary_final.csv` has, for each variant and opponent, the mean win rate of the final checkpoints and its min–max over seeds.

The two kinds of uncertainty are different: the **Wilson interval** measures the noise of the evaluation (how many matches were played), the **band between seeds** measures how much the result changes from one training run to another. With 5 seeds the comparisons remain indicative. The only test made is for the stall fix (§8.6): a bootstrap over seeds of the difference of the means, whose 95% interval is wide (with 5 seeds per group the smallest detectable difference is about 0.3 in win rate).

**Figures in the style of the paper.** `plots/plot_paper_figures.py` draws `paper_heuristics.png`, in the format of the paper's Figure 3: the win rate against random and against noop at the checkpoints (5M, 10M, 15M, 20M), as the mean over the 5 seeds of each variant (R0, R0 Sched., R1, R2, R3, R0 floor, R1 floor) with a 95% confidence band from Student's $t$ with 4 degrees of freedom. Unlike the min–max band of `plot_results.py`, this band is an interval on the **mean**, not the range of the seeds. The same script draws the best-response figure of §6.3.

### 6.3 Approximate best response

`src/best_response.py` estimates the exploitability of §4.4: how strong can a dedicated opponent become against one fixed agent?

- the **target** is a frozen checkpoint (by default the final checkpoint of seed 1 of each comparison variant); it keeps sampling from its policy, as in evaluation;
- a **new network** (the learner) is trained **from scratch** with our PPO against it. There is no self-play: the learner plays one side of every match (P1 in even matches, P2 in odd ones, so sides are balanced) and only its samples go into the batch, which is therefore half the size of the training batch;
- the learner's reward is always the **terminal** one (±10, 0 on timeout), whatever reward the target was trained with, so that the exploitabilities of different variants are comparable;
- the environment uses the **target's game settings** (action delay, charge actions);
- hyperparameters are those of the fixed-entropy runs (entropy 0.025), with the same 20M-step budget; by default the learner has no advantage floor;
- with `--adv-std-floor` (`FLOOR=0.1` in `slurm/br_array.sbatch`) the learner is trained with the stall fix of §5.4 and the run is called `br_<target>_b<seed>_floor`. This changes the **learner**, not the target: the targets are the same seed-1 checkpoints. It tests whether the learners that only drew (§8.4) were held back by their own stalls. It is a different experiment from the "best responses against the floor runs" listed as not done in §7;
- at the end the learner plays **500 matches** against the target, sides alternated; the result (wins, draws, losses, BR win rate with Wilson interval) is appended to `results/br_<target run>.csv`. The training curves go to `runs_br/<name>/` (excluded from git, like `runs/`). Their win, draw and loss rates are computed over the matches that **finish** in each update: a timeout ends one match every 4000 steps, a quick win many, so draws are underrepresented and the curves overestimate the final win rate; updates in which no match finishes have no value. Only the final evaluation, with the same number of matches per slot, measures exploitability (§8.4).

The **BR win rate** is the measure of exploitability: close to 0.5 or lower means the learner found no systematic weakness in its budget; close to 1 means the target is easy to exploit. It is a **lower bound**, because PPO only finds the weaknesses it manages to discover in 20M steps (as in the paper, Section 4, and Timbers et al. 2020). Draws matter here too: a learner that only draws against a passive target has not exploited it.

`slurm/br_array.sbatch` runs the 5 targets in parallel; `SEED=2 sbatch slurm/br_array.sbatch` adds a second learner seed per target, to see how much the estimate depends on the learner's training. A learner stopped by the time limit during training can be continued with `--resume` (§5.6); one that hits the limit after training can be evaluated alone with `--eval-only`. `plots/plot_report.py` draws the learners' win rate during training (`report_br_curves.png`). `plots/plot_paper_figures.py` draws `paper_br_outcomes.png`, in the format of the paper's Figure 4: the win, draw and loss rates of the **target** during the learners' training (per 1M-step bin, averaged over learner seeds with equal weight), with a bar for the final 500-match evaluation; `--br-suffix _floor` draws the same figure for the floored learners. The curves have the same bias as above; the bar is the measurement.

### 6.4 Behaviour diagnostics

The win rate cannot tell a **stall** (no match is decided, whatever the agents do) from **reactive** play (the agent waits for the opponent). `src/diagnostics.py` measures behaviour directly. For each checkpoint it plays a fixed number of steps on a vectorized environment, in self-play and against noop, back and random, and records:

- the **action distribution** of the agent: the share of each action, the **attack share** (`ATTACK`, `BACK+ATTACK`, `FORWARD+ATTACK`) and, when the charge actions are enabled, the **special share** (the three charge actions);
- for every finished match: outcome, length, whether anybody attacked, the time of the first attack and who made it.

Results are appended to `results/diag_<run>.csv`, one row per matchup. `slurm/diagnostics_array.sbatch` runs 9 checkpoints (seed 1 of R0, R0 Sched., R1, R2, R3, R0 floor and R1 floor, plus `r0_charge_s1` and `r0_lam0_s1`), and `plots/plot_diagnostics.py` draws `diag_stall.png` and `diag_actions.png`.

The script also records the distance between the players from observation features 2 and 43, but in our extract these features are not informative, so the distance is not used in the results.

---

## 7. Experiments

There are 3 comparisons: the 2 paper baselines, the reward functions and the bias–variance trade-off; then the stall fix and the environment ablations.

| Priority | Experiment | What changes | Runs | Question | Status |
|---|---|---|---|---|---|
| Core | Paper baselines | fixed entropy (0.025) vs Sched. (0.1 -> 0), both on R0 | `r0`, `r0_sched`, 5 seeds | Does the entropy schedule change stalls and robustness, as in the paper? | done |
| Core | Reward | R0, R1, R2, R3 (§4.5) | 5 seeds each | Which reward gives the strongest and least exploitable agent? | done |
| Core | Exploitability | best response against seed 1 of each variant (§6.3) | 5 targets × 2 learner seeds | Does head-to-head strength match low exploitability? | done (§8.4) |
| Core | Bias and variance | GAE $\lambda$ 0 and 1 vs 0.95; $\gamma$ 0.99 vs 0.995, on R0 | 1 seed each | How do speed and stability change with more bias (TD) or more variance (Monte Carlo)? | done (§8.5) |
| Secondary | Stall fix | `--adv-std-floor 0.1` on R0 and R1 | `r0_floor`, `r1_floor`, 5 seeds each | Does the fix reduce stalls on more than one seed, and does it change strength? | done (§8.6) |
| Secondary | Best response with a floored learner | learner trained with `--adv-std-floor 0.1`; targets `r0_s1`, `r0_sched_s1`, `r1_s1` | [TODO: learner seeds] | Does the learner escape the draws more often, and is `r1_s1` exploitable after all? | running (§8.4) |
| Secondary | Charge actions | `use_special_charge_action` on (9 actions, 88-dim observation), on R0 | 1 seed | Does the agent discover and use the specials? | done (§8.5, §8.7) |
| Secondary | Action delay | `action_delay` 0 and 12 vs 8, on R0 | 1 seed each | With delay 0 does the game become reactive and brittle, as in the paper? | done (§8.5) |
| Optional | Checkpoint pool (PFSP) | self-play against past snapshots | — | Does the pool reduce exploitability? | not run |
| Optional | Timeout penalty | symmetric penalty, outside zero-sum | — | Does it discourage passivity, and what does the evaluation lose? | excluded (§9.1) |

The bias–variance runs and the two game ablations are all in `slurm/ablations_array.sbatch` (6 runs: `r0_lam1_s1`, `r0_lam0_s1`, `r0_gamma099_s1`, `r0_charge_s1`, `r0_delay0_s1`, `r0_delay12_s1`); the reference is `r0_s1`. The stall fix is in `slurm/floor_array.sbatch`, the best responses in `slurm/br_array.sbatch` (with `FLOOR=0.1` for the floored learners), the diagnostics in `slurm/diagnostics_array.sbatch`. All runs use the same network, so differences can be attributed to the variable that changed.

---

## 8. Results

The runs of the original plan are complete. The five comparison variants and the two stall-fix variants have **5 seeds each**, the six ablations one seed each, and the best responses **2 learner seeds** for each of the 5 targets. The final checkpoints have been evaluated against the fixed opponents, the `final` head-to-head pool is complete and the behaviour diagnostics have been run on seed 1 of each variant. The `time` head-to-head pool has not been played.

### 8.1 Against the fixed opponents

Final checkpoints, 100 matches per opponent and seed, `results/summary_final.csv`; mean over seeds, min–max in brackets. **Final stall**: in the last 10% of training less than 1% of the self-play matches are decided. It can only be measured on runs whose log has the `decided_rate` column, so not on `r0_s1`, `r1_s1` and `r2_s1`.

| Variant | Seeds | vs random | vs FootsiesBot | vs noop (wins) | vs back (wins) | Final stall |
|---|---|---|---|---|---|---|
| R0 | 5 | 74% (41–88) | 68% (53–90) | 7% (0–34) | 0% | 1 of 4 |
| R0 Sched. | 5 | 82% (74–94) | 65% (13–87) | 0% | 0% | 3 of 5 |
| R1 | 5 | 90% (84–94) | 74% (41–85) | 2% (0–12) | 0% | 1 of 4 |
| R2 | 5 | 75% (18–90) | 63% (5–86) | 12% (0–50) | 0.4% (0–2) | 0 of 4 |
| R3 | 5 | 52% (5–77) | 35% (18–65) | 0% | 0% | 5 of 5 |
| R0 floor | 5 | 88% (82–92) | 75% (61–84) | 10% (0–34) | 0% | 0 of 5 |
| R1 floor | 5 | 91% (87–97) | 77% (72–84) | 2% (0–4) | 0% | 0 of 4 |

Every match against noop or back that is not won is a draw (no losses), so the draw share is 1 minus the win rate in those two columns.

The behaviour diagnostics (§8.7) cover two of the three runs that cannot be measured from the logs: the final models of `r1_s1` and `r2_s1` decide 0% of their self-play matches, while `r0_s1` decides 99%. The measure is not the same (final model instead of the last 10% of training), so the counts in the table are left unchanged; but the stalls of R1 and R2 are probably underestimated.

- **Among the comparison variants, R1 is the most consistent**: the best mean against random and FootsiesBot, and the narrowest range across seeds against random (84–94%).
- **R3 is the weakest.** It has the lowest mean against both random and FootsiesBot, one seed that almost never beats random (5%), and it is the only variant with many **draws** against FootsiesBot (15% on average, against 1–5% for the others). All 5 of its runs end in a stall. This fits the risk described in §4.5: the reward depends on a time the agent cannot see.
- **R2 is not the worst.** With seed 1 alone (18% against random, 5% against FootsiesBot) it looked like the worst variant, but that seed was an outlier: the other seeds reach up to 90% and 86%. The earlier hypothesis that the shaping breaks the optimal policy is therefore not supported; what remains is a **large spread between seeds**.
- **The spread between seeds is often larger than the difference between variants** (figure `report_seeds.png`): against FootsiesBot the ranges of R0, Sched., R1 and R2 overlap almost completely (Sched. goes from 13% to 87%; the standard deviation across seeds is 0.15–0.33). With 5 seeds the ordering of these four is not reliable; only R3 stands out.
- **The stall fix gives the most consistent agents**: R0 floor and R1 floor have the narrowest ranges against both random and FootsiesBot and no stalled run. The comparison with the base variants is in §8.6.

### 8.2 Reactivity and loss of initiative

Against an opponent that stands still or runs away the final agents almost never win: every non-won match is a timeout. The paper observes the same thing: the best policies become reactive and do not start the fight. Two facts refine this picture:

- it is **not absolute**: some final seeds do take the initiative against noop (R0 and R0 floor up to 34%, R2 up to 50%);
- it **emerges during training**: several runs beat the passive opponents at the early checkpoints and lose this ability later. For example, `r3_s5` beats noop and back in 100% of the matches at 5M steps and in 0% at the end; `r3_s4` beats noop 100% at 10M; `r1_s2` beats noop 100% at 5M; `r0_sched_s4` 90% at 5M. In all these cases the final checkpoints draw 100% of the matches. Figure `report_noop.png` shows the same drop as a mean over seeds: at 5M steps the variants beat noop in 60–100% of the matches, at 20M in 0–12%. Figure `paper_heuristics.png` (§6.2) shows it in the format of the paper, with a 95% interval on the mean instead of the min–max band.

So the agents **can** attack an opponent that stands still, but self-play pushes them away from it: against an opponent that is itself reactive, starting the fight is punished. This matches the hypothesis of §2.4 (whiffing an attack is costly, waiting is not), but it remains a hypothesis (§9.1). It also does not mean the agents cannot play: against FootsiesBot, which attacks, most seeds win most matches. The stall fix does not change this picture (R0 floor 10% against noop, R1 floor 2%): passivity against fixed opponents and stalls in self-play are different phenomena (§8.7).

### 8.3 Head-to-head

`final` pool: 26 agents, 325 pairs, 100 matches per pair, `results/h2h_final.s*of8.csv`, figure `h2h_final.png`:

| Variant | Mean score of its agents | Best seed | Worst seed |
|---|---|---|---|
| R0 | ≈ 0.53 | 0.58 (`r0_s2`) | 0.50 (`r0_s1`) |
| R0 Sched. | ≈ 0.51 | 0.65 (`r0_sched_s3`) | 0.36 (`r0_sched_s1`) |
| R2 | ≈ 0.51 | 0.58 (`r2_s2`) | 0.43 (`r2_s1`) |
| R1 | ≈ 0.49 | 0.55 (`r1_s4`) | 0.44 (`r1_s3`) |
| R3 | ≈ 0.45 | 0.50 (`r3_s5`) | 0.40 (`r3_s4`) |
| R1_floor | 0.57 (1 seed) | — | — |

The per-variant value is the mean of the ranking scores of its 5 agents (`results/h2h_final_ranking.csv`); it includes the matches between seeds of the same variant, so it is only a summary of the per-variant matrix on the right of `h2h_final.png`, which is the figure to cite. Of the floor runs only `r1_floor_s1` is in the pool, because the others were trained after it was played.

- **All variants are close to 0.5** and R3 is again last, consistent with the fixed opponents.
- **The best and the worst agent are both PPO (Sched.)** (0.65 and 0.36): the entropy schedule gives the largest spread between seeds.
- **Many pairs end in 100% draws.** Some agents are passive against almost everyone: `r0_s5`, for example, draws all 100 matches against several agents of every variant, so its score of 0.50 says nothing about its strength (§8.4). When both agents are reactive the match is a timeout; when at least one is less passive, matches are decided.
- **Head-to-head ranking and fixed opponents do not agree.** R1 is the best variant against random and FootsiesBot but slightly below 0.5 head-to-head, while R0 is the other way round. A strong result against simple opponents does not imply a strong result against other learned agents. Whether it implies **low exploitability** is measured by the best responses in §8.4.

The draws against the fixed opponents and in head-to-head are therefore partly a property of the game and of self-play, not only of the reward: the comparison has to keep the two effects apart.

### 8.4 Exploitability

Approximate best responses (§6.3): one learner per target and learner seed, trained for 20M steps, then 500 matches against the target with sides alternated (`results/br_<target>.csv`; learner runs `br_<target>_b1` and `br_<target>_b2`). The targets are the final checkpoints of seed 1 of each variant. The head-to-head columns come from `results/h2h_final_ranking.csv` (26 agents). The **BR score** uses the same definition as head-to-head (§6.1), $(W + 0.5\,D)/500$, so that a learner that only draws gets 0.5.

| Target | H2H score | H2H rank (of 26) | Learner seed | BR wins / draws / losses | BR win rate (95% CI) | BR score |
|---|---|---|---|---|---|---|
| `r0_s1` | 0.50 | 15 | 1 | 241 / 222 / 37 | 0.48 (0.44–0.53) | 0.70 |
| | | | 2 | 0 / 500 / 0 | 0.00 (0.00–0.01) | 0.50 |
| `r0_sched_s1` | 0.36 | 26 | 1 | 1 / 497 / 2 | 0.00 (0.00–0.01) | 0.50 |
| | | | 2 | 249 / 247 / 4 | 0.50 (0.45–0.54) | 0.75 |
| `r1_s1` | 0.48 | 18 | 1 | 0 / 500 / 0 | 0.00 (0.00–0.01) | 0.50 |
| | | | 2 | 0 / 500 / 0 | 0.00 (0.00–0.01) | 0.50 |
| `r2_s1` | 0.43 | 23 | 1 | 499 / 0 / 1 | 1.00 (0.99–1.00) | 1.00 |
| | | | 2 | 500 / 0 / 0 | 1.00 (0.99–1.00) | 1.00 |
| `r3_s1` | 0.40 | 24 | 1 | 494 / 0 / 6 | 0.99 (0.97–0.99) | 0.99 |
| | | | 2 | 157 / 224 / 119 | 0.31 (0.27–0.36) | 0.54 |

- **The estimate depends on the learner seed.** For `r0_s1` and `r0_sched_s1` one learner gets out of the draws and the other does not; for `r3_s1` one learner wins almost always and the other only partly, oscillating until the end (figure `report_br_curves.png`). The learners escape late: the one against R0 only after about 14M steps, the one against Sched. after about 8M, so a learner still drawing at 20M might escape with more budget. The learner is trained with the same PPO and the same terminal reward as the main agents, without the floor, so it plausibly suffers from the same stalls (§5.4). Since the best response is a lower bound, each target is judged by its **best learner**. Two learner seeds are still few: a third learner could find more.

  The curves count the learner's wins over the matches **finished** in each update, where timeouts (4000 steps) are underrepresented (§6.3), so they overestimate the final level: the learner against R0 is at about 95% during training but wins 48% in the final evaluation, with 222 draws. They show when a learner escapes the draws, not how much it wins; the table above is the measure to use.
- **R2 and R3 are clearly exploitable.** Both `r2_s1` learners win 499 and 500 matches out of 500; the best `r3_s1` learner wins 494. In head-to-head these agents look roughly on par with the others (0.43 and 0.40). Their scores are not lower only because many of their pairs end in 100% draws (§8.3): a draw against another passive agent costs nothing, but an opponent that adapts to them wins almost always. For `r2_s1` this matches the fixed opponents (18% against random, 5% against FootsiesBot, the outlier seed of §8.1).
- **R0 and R0 Sched. are exploitable, but much less.** The best learner wins about half of the matches against both (BR win rate 0.48 and 0.50) and loses very few (37 and 4 out of 500). The BR scores of 0.70 and 0.75 show a real, partial weakness, while 44–49% of the matches still end in a timeout. The rule of thumb of §6.3 ("close to 0.5 or lower means no systematic weakness") has to be read together with the draws.
- **`r1_s1` is never exploited, but this does not show that it is robust.** Both learners draw all 500 matches: in 1000 matches they never win and never lose. This is the case anticipated in §6.3: against a passive target, a learner that only draws has not exploited it, and the draws can be an artefact of a stalled learner: both learners drop to 0% wins within about 2M steps and never recover (`report_br_curves.png`). The correct statement is that **no best response exploited R1**, not that R1 is robust. `r1_s1` is itself passive in head-to-head (100% draws against several agents).
- **Four of the five targets are stalled in self-play** (§8.7): all except `r0_s1`. A stall therefore does not protect against an opponent that adapts: `r2_s1` and `r3_s1` are stalled and still lose almost every match to their best learner. This is one more reason not to read the draws of `r1_s1`, also stalled, as robustness.
- **Answer to the guiding question.** Winning against simple opponents does not imply being hard to exploit, and head-to-head strength only partly predicts exploitability. At the bottom of the ranking the two agree for R2 and R3, which are almost completely exploitable. But `r0_sched_s1`, **last** in head-to-head (0.36), is only moderately exploitable, about as much as `r0_s1` (15th). Against the fixed opponents `r0_s1` (88% against random, 66% against FootsiesBot) and `r1_s1` (90% and 76%) win comparably, yet one is moderately exploitable and the other gives no information. All of this rests on one target seed per variant; best responses against the floor runs, which do not stall, have not been trained.

### 8.5 Ablations

One seed each, final checkpoints, 100 matches per opponent, `results/eval_<run>.csv`. The reference is `r0_s1`. For comparison, the R0 range over its 5 seeds (§8.1) is 41–88% against random and 53–90% against FootsiesBot.

| Run | What changes | vs random | vs FootsiesBot | vs noop (wins) | vs back (wins) |
|---|---|---|---|---|---|
| `r0_s1` (reference) | — | 88% | 66% | 0% | 0% |
| `r0_lam0_s1` | GAE $\lambda = 0$ (one-step TD) | 5% | 4% | 0% | 0% |
| `r0_lam1_s1` | GAE $\lambda = 1$ (Monte Carlo) | 81% | 83% | 0% | 0% |
| `r0_gamma099_s1` | $\gamma = 0.99$ | 69% | 85% | 0% | 0% |
| `r0_charge_s1` | charge actions on | 82% | 76% | 86% | 0% |
| `r0_delay12_s1` | `action_delay` 12 | 70% | 39% | 100% | 50% |
| `r0_delay0_s1` | `action_delay` 0 | 64% | 55% | 0% | 0% |

- **$\lambda = 0$ never learns.** It wins 4–9% against random at every checkpoint (5% at 5M, 4% at 10M, 9% at 15M, 5% at the end) and 4% against FootsiesBot (74 losses, 22 draws). In self-play it uses only `ATTACK` and `BACK`, about half each (§8.7). This is consistent with the bias side of §5.3: with pure one-step TD and only a terminal reward, the win reaches the early actions only through the critic's bootstrap, across up to thousands of steps (§4.3), and in 20M steps this does not happen. The explanation is a hypothesis consistent with the data, not a proof.
- **$\lambda = 1$ and $\gamma = 0.99$ do not hurt, at least on one seed.** Against FootsiesBot they reach 83% and 85%, above `r0_s1` (66%) but inside the R0 range (53–90%); against random 81% and 69%, also inside the range. With one seed the effect cannot be separated from the spread between seeds. The only clear bias–variance result is therefore that too much bias blocks learning. Like many comparison runs, `r0_lam1_s1` beats noop and back at 5M (98% and 94%) and loses this ability by the end (§8.2).
- **With charge actions the agent attacks a passive opponent again, but not because of the specials.** It beats noop in 86% of the matches at the end, more than any R0 seed (at most 34%). The ability goes from 100% at 5M to 0% at 10M and returns at the end, while against back it stays at 0%. The diagnostics (§8.7) show that the charge actions are rarely used: 0.5% of the steps in self-play, 0.2% against noop, 1.7% against random. The result is therefore not explained by a strategic use of the specials, and has to be attributed also to the different action and observation spaces.
- **Delay 12: more initiative, weaker against FootsiesBot.** It is the only agent that, at the end of training, beats noop in 100% of the matches and back in 50%, but against FootsiesBot it falls to 39%, below the whole R0 range. With $D = 12$ the reaction time is 12–16 frames, and by the rule of §2.4 the recovery of `B_ATTACK` (15) becomes only *sometimes* punishable on reaction: whiffing is cheaper, which fits the hypothesis that passivity comes from the cost of whiffing (not verified).
- **Delay 0: no initiative, mid-range strength.** It wins 64% against random and 55% against FootsiesBot, inside the R0 range, and draws every match against noop and back. Compared with delay 12 this agrees with the paper on no-op (§2.4), but not on random, where delay 0 does not win more. `r0_s1` also draws every match against noop and back, so with one seed the ablation cannot show that the delay is the cause.

### 8.6 Stall fix

R0 and R1 with `--adv-std-floor 0.1` (§5.4), 5 seeds each, compared with the base variants. Difference floor minus base with a bootstrap over seeds (95% interval), and ratio between the standard deviations across seeds:

| Comparison | Base (mean, std) | Floor (mean, std) | Difference [95% CI] | Std ratio |
|---|---|---|---|---|
| R0, vs random | 0.74, 0.19 | 0.88, 0.04 | +0.14 [+0.02, +0.31] | 0.20 |
| R0, vs FootsiesBot | 0.68, 0.15 | 0.75, 0.09 | +0.07 [−0.08, +0.20] | 0.60 |
| R1, vs random | 0.90, 0.04 | 0.91, 0.04 | +0.02 [−0.03, +0.06] | 1.10 |
| R1, vs FootsiesBot | 0.74, 0.19 | 0.77, 0.04 | +0.04 [−0.08, +0.21] | 0.23 |

- **Total stalls disappear in the observed seeds**: 2 of 8 measurable runs end in a stall without the floor, 0 of 9 with it. Figure `report_stall.png` shows the mechanism on seed 5 of R0. Without the floor, `r0_s5` stalls twice: early (about 0.7–1M steps) it recovers, but from about 13.7M steps to the end all matches are timeouts and the advantage std stays at ~$10^{-3}$. This fits the role of the learning rate in §5.4: an early stall can be escaped, a late one, with the learning rate close to 0, cannot. With the floor, `r0_floor_s5` also goes through phases with up to 100% timeouts and drops in the (unfloored) advantage std down to ~$10^{-2}$, but it gets out every time; it ends with about 75% of its matches in timeout, so the floor prevents the lock-in, not the timeouts.
- **The mean barely changes.** Only R0 against random has an interval that excludes zero; the other three are not distinguishable from zero. With 5 seeds per group the smallest detectable difference is about 0.3 in win rate, so no difference does not mean equivalence.
- **The spread between seeds drops** in 3 cases out of 4, by a factor of 2–5 (with 5 seeds this is a weak estimate).

The floor does not raise the expected strength; it reduces the variance of training. This fits the bias–variance theme of §5.3: the problem it fixes is variance added by the normalization, not a wrong objective.

### 8.7 Behaviour diagnostics

`results/diag_<run>.csv`, seed 1 of each variant (§6.4), figures `diag_stall.png` and `diag_actions.png`.

- **Stalled agents are not passive.** Five of the nine checkpoints decide no self-play match at all: seed 1 of R1, R2, R3, R0 Sched. and R0 $\lambda = 0$. These agents attack in 43–54% of the steps, more than those without a stall (14–38%: R0 charge 14%, R0 floor 28%, R0 35%, R1 floor 38%), but their attacks never land a decisive hit. The correct description of a stall is therefore "timeouts with attacks that do not conclude", not "both agents wait".
- **Stalled agents attack, but do not approach.** In `diag_actions.png` the share of `FORWARD` is about 4–11% for the stalled agents (R1, R2, R3, R0 Sched.) against about 22–31% for the others (R0, R0 floor, R1 floor, R0 charge). An agent that attacks often but rarely moves forward plausibly attacks out of range, which would explain why the attacks never conclude. Without a measure of the distance (§6.4) this remains a hypothesis.
- **$\lambda = 0$ is degenerate**: in self-play it uses only `ATTACK` and `BACK`, about 50% each, and every match is a timeout.
- **The charge actions are rarely used** (§8.5): at most about 2% of the steps, depending on the opponent.
- **The agents without a stall keep matches decided**: `r0_s1` decides 99% of its self-play matches, `r0_floor_s1` 97% and `r1_floor_s1` 95%, with attack shares of 35%, 28% and 38%.

Three facts link the stalls to optimization rather than to the reward: in a stall the reward is almost always zero and the advantage std falls to ~$10^{-3}$, so the normalization amplifies noise; the entropy bonus pushes the policy towards uniform; and the learning rate going to 0 prevents escaping (§8.6). These are hypotheses consistent with the data, not proven.

---

## 9. Discussion and limits

### 9.1 Passivity and stalls

The results show two different phenomena, which are easy to confuse because both end in timeouts:

- **stalls in self-play** (§8.7): no match is decided, although the agents attack a lot. They are linked to optimization (§5.4), and the advantage floor removes them in the observed seeds (§8.6);
- **reactivity against passive opponents** (§8.2): the final agents rarely start the fight against noop and back, even when they have learned to do it earlier in training. The floor does not change this (R0 floor 10% against noop, R1 floor 2%).

Our hypothesis for the second (§2.4) is that with the default delay attacks must be predicted, while whiffed attacks can always be punished on reaction: whiffing is costly and waiting is not, so against an opponent that is itself reactive the best response is to wait. The delay-12 run (§8.5), where whiffing is cheaper and the agent does take the initiative, is consistent with it; the hypothesis that mutual inactivity is a fixed point of symmetric self-play is not verified.

A **timeout penalty** would discourage passivity directly. It was excluded on purpose: if it is equal for both players it breaks zero-sum, and with it the exact meaning of best response and exploitability (§4.4), which are the measures the guiding question relies on.

### 9.2 What the results support and what they do not

They support that:

- no reward clearly beats the others, and R3 is the worst;
- the spread between seeds is larger than the differences between variants;
- stalls are frequent, and the floor removes them in the observed seeds without changing the mean strength;
- R2 and R3 are highly exploitable, even though they are stalled in self-play;
- $\lambda = 0$ does not learn;
- head-to-head rank only partly predicts exploitability;

They do not support that:

- R1 is more robust (no best response exploited it, but only through draws, and its learners stalled);
- the floor improves mean strength;
- the charge actions are used strategically;
- the variants are equivalent (5 seeds cannot detect differences below about 0.3);

### 9.3 Limits

- The ablations and the diagnostics use **one seed**; exploitability uses **one target seed** per variant and **two learner seeds**.
- The best response is a lower bound, and the learners themselves can stall; their training curves overestimate the final win rate (§6.3, §8.4).