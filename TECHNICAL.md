# arm-adapt: technical summary (week 1)

This document describes exactly what was built and why. It's written at expert level on purpose, so you can hand it to
a separate chat and have it teach you at your own pace (see the prompt at the bottom).

**Stack:** MuJoCo 3.17 (physics) · Gymnasium 1.4 (environment API) · Stable-Baselines3 2.9 with PyTorch 2.14 (SAC) ·
imageio and matplotlib (outputs) · uv (Python 3.12 environment).

**Pipeline:** `arm_model.py` produces MJCF XML → `reach_env.py` wraps it as a `gym.Env` → `train.py` fits a SAC policy →
`evaluate.py` scores it against a random baseline and renders outputs.

---

## 1. `arm_model.py`: morphology as a parameter

`make_arm_xml(link_lengths, motor_gear)` returns an MJCF string for a **planar serial-chain manipulator**.

- **Kinematic tree.** Each link is a `<body>` nested inside its parent and offset by the parent's length along the
  parent's local x-axis. That nesting is what makes it a serial chain, so forward kinematics are handled by MuJoCo's body
  tree. The XML is built inside-out in a loop, so the number of links is just `len(link_lengths)`.
- **Joints.** All joints are hinges about +z. The shoulder (`joint0`) is unlimited. Every later joint is limited to ±170° so
  the arm can't fold through itself (there's no collision checking: `contype=conaffinity=0`).
- **Geometry and mass.** Each link is a capsule with radius 1 cm and length 10 cm, at density 1000 kg/m³, giving
  **0.036 kg per link**. The fingertip is a `site` (a reference point with no physics) at the end of the last link.
- **Actuation.** There's one `<motor>` per joint with `ctrlrange [-1, 1]` and `gear 0.05`, so peak torque is **0.05 N·m**.
  These are direct torque motors, not position servos: the policy controls torque, not angles.
- **Joint dynamics.** `damping 0.002` (viscous friction) and `armature 0.0005` (reflected rotor inertia). The armature is
  larger than the links' own rotational inertia about their joints (about 1.2e-4 kg·m²), so it makes the arm feel more
  like a geared motor and keeps the dynamics well-conditioned.
- **Simulation.** `timestep 0.01 s` with the RK4 integrator. The plane is horizontal and all motion is in-plane, so gravity
  does no work: this isolates the control problem from load-bearing.
- **Target.** A `mocap` body (moved by code, not physics) shown as a green sphere.
- **Camera.** A fixed top-down camera, with its height scaled to the total reach.

**Why it's parameterized:** weeks 2 and 3 change the body. Since morphology is a function argument, a 3-joint arm is
`make_arm_xml((0.1, 0.1, 0.1))` with nothing else to edit.

## 2. `reach_env.py`: the MDP

The task is a Markov Decision Process: at each step the agent sees an observation `o`, picks an action `a`, and gets a
reward `r`.

| | Definition | Size for n joints |
|---|---|---|
| **Observation** | `[cos q, sin q, q̇, target_xy, target_xy − tip_xy]` | 3n + 4 (**10** for n = 2) |
| **Action** | normalized torques in [-1, 1]ⁿ, written straight to `data.ctrl` | n (**2**) |
| **Reward** | `−‖tip − target‖ / reach − 0.05·‖a‖²` | scalar |

- **Control rate.** `frame_skip = 2`, so the agent acts every 0.02 s (50 Hz). Episodes are truncated (not terminated) at
  100 steps, which is 2 s. There is no early termination on success.
- **Angle encoding.** cos/sin removes the jump at ±π and keeps nearby poses close together as inputs to the network.
- **Error vector.** `target − tip` is redundant (the network could work it out from q and the target), but giving it
  directly makes the task close to linear near the goal, which speeds up learning a lot. This is a deliberate
  shaping-by-observation choice.
- **Reward shaping.** It's a dense reward: distance is normalized by total reach so the same scale works for different
  bodies (this matters in week 3). The quadratic effort penalty stops bang-bang chattering around the target.
- **Reset distribution.** The shoulder angle is uniform in [-π, π], the other joints are uniform in ±0.1 rad, and the
  velocities are zero. The target is at radius ∈ [0.15, 0.95] × reach and a uniform angle, so it's always reachable and
  never right on the base.
- **Success metric** (reported only, not part of the reward): final distance < 2 cm, which is 10% of reach.

## 3. `train.py`: Soft Actor-Critic

SAC is an **off-policy, maximum-entropy actor-critic** algorithm for continuous action spaces. Its objective maximizes
expected return plus an entropy bonus, `E[Σ γᵗ (rₜ + α·H(π(·|oₜ)))]`.

- **Actor π(a|o):** a squashed Gaussian policy. The MLP outputs a mean and log-std, a sample is drawn, and then `tanh` maps
  it into [-1, 1]. At evaluation time `deterministic=True` uses the `tanh(mean)`.
- **Critics Q₁, Q₂(o, a):** twin Q-networks trained against the Bellman target
  `r + γ(min(Q̄₁, Q̄₂)(o′, a′) − α log π(a′|o′))`. Taking the minimum of two critics counters overestimation (as in TD3).
  Q̄ are **target networks**, updated by Polyak averaging with τ.
- **Temperature α:** tuned automatically so the policy entropy tracks a target of −dim(A), which is −2. This sets the
  exploration–exploitation balance without hand tuning.
- **Replay buffer:** stores every transition (o, a, r, o′). Gradient updates sample random minibatches from it, which
  breaks the correlation between consecutive steps and reuses data, so it's very sample-efficient compared with on-policy
  methods like PPO.

| Hyperparameter | Value | Meaning |
|---|---|---|
| `net_arch` | [128, 128] | two hidden layers per network (default is 256×256; this task doesn't need that) |
| `learning_starts` | 2000 | random actions to seed the buffer before any updates |
| `learning_rate` | 3e-4 | Adam step size for the actor, critics and α (default) |
| `buffer_size` | 1e6 | larger than the run, so nothing is evicted (default) |
| `batch_size` | 256 | transitions per gradient step (default) |
| `gamma` | 0.99 | discount factor, with an effective horizon of about 100 steps ≈ one episode (default) |
| `tau` | 0.005 | target-network averaging rate (default) |
| `train_freq`, `gradient_steps` | 1, 1 | one gradient update per environment step (default) |
| `ent_coef` | auto | learned α (default) |
| total steps | 200,000 | 2,000 episodes |
| seed | 0 | |

`ProgressLog` tracks the mean final distance over the last 20 episodes every 5k steps and saves that as the learning curve.

## 4. `evaluate.py`: measurement

- 200 episodes with seeded resets (seeds 1000–1199), so the trained and random policies face the **same targets**.
- It reports mean and worst final distance, and success at < 2 cm.
- The GIF shows 6 targets (seeds 2000–2005), with every 2nd frame at 25 fps.
- The curve plots the training-time metric from `train.py`.

## 5. Results

| | Mean final distance | < 2 cm | Worst |
|---|---|---|---|
| Random | 17.9 cm | 0.5% | 37.4 cm |
| SAC, 60k steps | 2.5 cm | 47.5% | 8.3 cm |
| **SAC, 200k steps** | **1.5 cm** | **77.5%** | **5.1 cm** |

Wall-clock time was 5.5 min for 200k steps on an Apple M5 CPU, about 600 steps/s, limited by gradient updates rather than
physics.

**Limitations and things worth trying**
- **The precision plateau** is around 1.2–1.5 cm. Linear distance reward gives the same push at every distance, so there's
  weak pressure to close the last centimeter. Candidates: a success bonus, `−log(d + ε)`, or `exp(−d/σ)` shaping.
- **Single seed.** A credible result needs 3–5 seeds with mean ± std. This matters for the week 2–3 comparisons.
- **No sensor noise, delay or actuator limits** beyond torque saturation, so it's a large sim-to-real gap for now.

## 6. Where this is going (weeks 2–3)

The observation and action dimensions depend on the joint count (3n + 4 and n). A policy trained for n = 2 **can't even be
applied** to n = 3 without an input/output adapter. Week 2 measures the failure with simple adapters (zero-padding,
or freezing the new joint). Week 3 trains one **morphology-conditioned** policy over randomized bodies (padded
observations plus a body descriptor), in the spirit of Huang et al. 2020, *One Policy to Control Them All*, which used a
shared modular policy per joint.

---

## Prompt to learn this in a separate chat

> I'm a first-year mechanical engineering student at Cornell. I know calculus, intro physics, and Python basics, and I've
> started Karpathy's Zero to Hero (micrograd). I built the project below with an AI agent and need to understand it well
> enough to explain it to a professor without notes. Teach me **top-down**: first the big picture in 5 sentences, then
> go one layer deeper at a time (MDP → reward design → neural network policy → SAC's actor/critic/entropy → replay buffer).
> After each layer, ask me 2 questions and don't move on until I answer correctly. Use the arm as the running example.
> Here's the technical summary: [paste this file]
