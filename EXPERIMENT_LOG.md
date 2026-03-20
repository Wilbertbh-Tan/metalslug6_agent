# Experiment Log

Training log for Metal Slug 6 RL agent. Documents problems encountered, hypotheses, fixes applied, and results.

---

## PPO_26 — Value Function Collapse (2026-03-19)

**Status:** Active problem. Training plateaued at 17M steps.

### Problem

After 17M steps (5+ days), the agent is stuck oscillating between curriculum levels 2 and 3 (-8 and -12 death penalty per event). Global avg score plateaued at ~35k. High score reached 161,530 early but hasn't improved since.

**Root cause: PlasticityResetCallback destroyed the value function.**

The `PlasticityResetCallback` resets the policy and value linear heads every 500k steps. Data from training logs:

| Iteration | Step | Explained Variance | Event |
|-----------|------|-------------------|-------|
| 244 | ~500k | 0.424 | Just before 1st reset |
| 245 | 500k | -0.066 | **1st plasticity reset** — dips but recovers |
| 488 | ~1M | **0.707** | Value function at its best |
| 490 | 1M | **-0.075** | **2nd plasticity reset — permanent collapse** |
| 500 | ~1.02M | 0.00005 | Never recovers |
| 8241 | ~16.9M | 0.0000025 | Still dead 15.9M steps later |

The 1st reset (500k) was survivable — the CNN features were still general enough for the value head to relearn quickly. By the 2nd reset (1M), the CNN had specialized enough that a randomly initialized linear head could not recover. From that point on, the critic output was essentially constant, meaning PPO's advantage estimates were pure noise.

### Impact

- **Wasted ~15M steps** of training (~4.5 days of GPU time)
- The policy still improved somewhat via raw reward signal, but without a functioning critic, PPO degenerates to a noisy policy gradient method
- Curriculum oscillation at levels 2↔3 is a downstream symptom — the agent can't learn efficiently enough to sustain higher penalties

### Fix for Next Run

- **Disable `PlasticityResetCallback`** entirely, or only reset `action_net` (never `value_net`)
- Weight decay (1e-4) already addresses plasticity loss without the nuclear option of resetting learned weights
- If plasticity resets are needed in the future, use a warm restart (scale weights by 0.1) instead of full orthogonal re-init

---

## PPO_26 — Curriculum Oscillation (2026-03-18)

**Status:** Ongoing, partially caused by value collapse above.

### Problem

The curriculum death penalty system oscillates between levels 2 and 3:
1. Rolling avg score crosses 40k → upgrade to level 3 (-12 each, -36 total)
2. Higher penalties tank the rolling avg below 32k (80% regression threshold)
3. Downgrade back to level 2 (-8 each, -24 total)
4. Agent recovers → cycle repeats

This happened continuously from ~2M steps through 17M steps.

### Analysis

Two contributing factors:
1. **Value function collapse** (above) — the agent can't adapt efficiently to new penalty levels
2. **Penalty jump is steep** — going from -24 total to -36 total (50% increase) with the regression threshold at 80% of the upgrade score creates a narrow stability band

### Potential Fixes

- Fix the value function first — a working critic may let the agent adapt to level 3 naturally
- If oscillation persists with a working critic: smooth the penalty transitions (linear interpolation between levels instead of discrete jumps)
- Widen the hysteresis band (e.g., upgrade at 40k but only downgrade below 25k instead of 32k)

---

## PPO_26 — Reward Rebalancing (2026-03-14)

**Status:** Resolved. Working as designed.

### Problem

In PPO_16/22, the survival bonus accounted for **43% of total positive reward**. The agent was incentivized to stay alive and move slowly rather than push forward and score. This contributed to score plateaus around 45-50k.

### Fix Applied

Rebalanced reward components:

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| survival_bonus | 0.01/step | 0.003/step | Was 43% of reward, reduced to ~3% |
| progress_scale | 0.01 | 0.03 | Tripled forward momentum incentive |
| time_penalty | -0.0005 | -0.002 | 4x increase for urgency |
| scroll_novelty | 0.005 | 0.02 | Reward exploring new areas |
| score_reward | unchanged | unchanged | Now naturally dominant at 50-65% |

### Result

Reward breakdown in PPO_26 episodes confirms the rebalancing works:
- Score reward: **50-65%** of positive reward (was ~30%)
- Survival bonus: **2-3%** (was 43%)
- Progress: **11-15%** (was ~10%)

The agent pushes forward more aggressively and scores higher per episode when it performs well.

---

## PPO_25 — CNN Architecture Upgrade (2026-03-14)

**Status:** Resolved. Superseded by PPO_26.

### Problem

PPO_16/22 used a narrow ImpalaCNN (16→32→32 channels, 256-dim features). This limited the model's capacity to represent complex game states (projectile patterns, enemy types, terrain variations).

### Fix Applied

- Widened CNN: 16→32→32 → **32→64→64** channels
- Features dim: 256 → **512**
- Added **DrAC random crop augmentation** (pad 4px, random crop) for visual generalization
- Added **orthogonal initialization** (gain=sqrt(2)) for all conv/linear layers
- Longer rollouts: n_steps 512 → **2048** (compensates for single env limitation)

### Result

PPO_25 was a test run that confirmed the architecture works. PPO_26 is the full training run with these changes. The wider CNN + DrAC showed healthy learning in the first 1M steps (before the plasticity reset issue).

---

## PPO_21/22 — Curriculum Regression (2026-03-10)

**Status:** Resolved.

### Problem

PPO_21 reached curriculum level 4 (-18 each) and catastrophically regressed — scores dropped from 55k to 2k with no recovery mechanism. The agent couldn't handle the penalty increase and spiraled.

### Fix Applied

Added regression protection to `DeathPenaltyCurriculumCallback`:
- **Downgrade trigger**: If rolling avg drops below 80% of current level's threshold for 100 consecutive checks
- **Cooldown**: After downgrade, block upgrades for 200 episodes to prevent rapid oscillation
- More granular milestones (9 levels instead of 5) with smaller jumps

### Result

PPO_22 confirmed the regression protection works — the agent no longer catastrophically collapses. However, it revealed the oscillation problem documented above.

---

## PPO_18 — Aggressive Death Penalties (2026-03-08)

**Status:** Resolved. Key lesson learned.

### Problem

Started training with -50 death penalty per event (-150 total for a full death). The agent plateaued at 25k avg score (breakeven was 75k), with every episode deeply negative (-110 avg reward). Scores were declining over time.

### Analysis

With -150 total death penalty, the agent needed to score ~75k just to break even on an episode with one death. Since early training episodes rarely reach 75k, virtually every episode had negative reward. The agent learned to be extremely cautious (surviving longer but not scoring) rather than taking risks to progress.

### Fix Applied

Curriculum system: start with mild penalties (-5 each, -15 total, breakeven ~8k) and gradually increase as the agent demonstrates it can score higher. This gives the agent positive reward signal to learn from in early training.

### Key Lesson

**Never start with the final penalty level.** RL agents need positive reward signal to learn. If the penalty makes most episodes negative from the start, the agent has no gradient toward better behavior.

---

## PPO_13/14 — Entropy Collapse (2026-03-06)

**Status:** Resolved.

### Problem

PPO_13 used `ent_coef=0.01`. Entropy collapsed over training, causing the policy to become deterministic too early. The agent memorized specific action sequences that worked for early levels but couldn't generalize.

PPO_14 attempted to fix this by resuming with `ent_coef=0.02`, but loading a checkpoint with collapsed entropy into a higher entropy coefficient caused instability — high clip fractions (0.414) and regression from 70k to 44k avg score.

### Fix Applied

- Set `ent_coef=0.02` from the start of fresh runs
- Monitor entropy during training (should stay above -4.0 for this action space)
- Don't try to "rescue" an entropy-collapsed checkpoint by cranking up ent_coef mid-training

---

## Next Steps: Learning from Human Demonstrations

The agent has reached a plateau where pure RL exploration is slow to discover optimal strategies. Human demonstrations could accelerate learning by providing high-quality trajectories that show effective movement, dodging, and scoring patterns.

### Approach 1: Behavioral Cloning (BC) Pre-training

Pre-train the policy on human gameplay before PPO takes over. The agent starts with a reasonable policy instead of random actions.

- **How**: Record human gameplay via RetroArch's replay/recording system, extract (observation, action) pairs, train the policy network with supervised learning
- **Pros**: Simple to implement, gives the agent a strong starting point
- **Cons**: BC alone suffers from compounding errors (distribution shift). Best used as initialization for PPO, not as the final policy
- **Library**: [imitation](https://github.com/HumanCompatibleAI/imitation) library integrates with SB3

### Approach 2: DAgger (Dataset Aggregation)

Iterative imitation learning that addresses BC's distribution shift problem. The agent plays, a human labels the "correct" action at visited states, and the dataset grows over rounds.

- **How**: Agent plays → human corrects via keyboard overlay → retrain policy on aggregated dataset
- **Pros**: More robust than pure BC, handles states the human demo didn't visit
- **Cons**: Requires interactive human labeling sessions, more complex to set up

### Approach 3: Reward Shaping from Demonstrations

Use human trajectories to define auxiliary rewards rather than directly imitating actions.

- **How**: Extract state visitation patterns from human play, reward the agent for visiting similar states or achieving similar progress milestones
- **Pros**: Compatible with existing PPO pipeline, doesn't constrain the policy to human actions
- **Cons**: Requires defining a meaningful state similarity metric

### Approach 4: Hybrid — BC Warm-start + PPO Fine-tuning

The most practical approach for this project:

1. Record 5-10 human playthroughs via RetroArch (cover different strategies/routes)
2. Extract observation-action pairs at the same resolution/frame-skip as training (160x120, skip=3)
3. Pre-train the ImpalaCNN policy with BC for ~50k gradient steps
4. Switch to PPO training with the pre-trained weights (lower initial learning rate to avoid catastrophic forgetting)
5. Curriculum death penalties start mild as before

### Implementation Requirements

- RetroArch recording setup (`.bsv` replay files or screen capture + input logging)
- Script to convert replays into SB3-compatible `(obs, action, reward, done)` transitions
- BC training loop (or use `imitation` library's `bc.BC` class)
- Modified `train_ppo.py` to accept a pre-trained policy checkpoint

### Priority

**Medium-high.** Should be attempted if PPO_27 (with fixed value function) still plateaus below 50k avg. The current reward structure is solid; the bottleneck may be exploration efficiency in later game stages where enemies and terrain become more complex.

---

## Open Questions

1. **Should curriculum use linear interpolation?** Instead of discrete penalty levels, smoothly increase the penalty as a function of rolling avg score. This would eliminate the oscillation problem.

2. **Is VecNormalize interfering with value learning?** With `norm_reward=True`, the reward distribution shifts over time. The critic sees normalized rewards but the normalization stats are non-stationary. Worth testing with `norm_reward=False`.

3. **How many human demos are needed?** Literature suggests 5-20 demonstrations are sufficient for BC warm-start in Atari-like environments. Quality matters more than quantity — a few expert runs are better than many mediocre ones.
