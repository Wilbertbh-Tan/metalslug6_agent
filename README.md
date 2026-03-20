# Metal Slug 6 RL Agent

[![Secret Detection](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/secret-scan.yml/badge.svg)](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/secret-scan.yml)
[![Lint](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/lint.yml/badge.svg)](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/lint.yml)
[![Docker Build](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/docker-build.yml/badge.svg)](https://github.com/Wilbertbh-Tan/metalslug6_agent/actions/workflows/docker-build.yml)

> **Work in Progress** — Active development. Architecture and hyperparameters are being iterated on.

PPO-based reinforcement learning agent for Metal Slug 6 (Atomiswave/Flycast) using a custom Gymnasium environment. The agent learns to play from raw pixels and RAM-based rewards inside a headless Docker container running RetroArch.

## Current Results

| Run | Architecture | Max Score | Avg Score | Steps | Notes |
|-----|-------------|-----------|-----------|-------|-------|
| PPO_16 | ImpalaCNN 16-32-32 | 196,360 | 47,784 | ~5.8M | Best avg score baseline |
| PPO_22 | ImpalaCNN 16-32-32 | 150,080 | 45,000 | ~5.5M | Curriculum death penalties |
| PPO_26 | ImpalaCNN 32-64-64 + DrAC | 161,530 | 35,153 | ~17M | Wider CNN, reward rebalance. Value function collapsed at 1M steps due to plasticity reset — see [experiment log](EXPERIMENT_LOG.md) |

## Architecture

- **Policy**: SB3 PPO with ImpalaCNN (IMPALA ResNet + Global Average Pooling)
- **CNN**: 3 ConvSequence blocks (32→64→64), orthogonal init, DrAC random crop augmentation
- **Observations**: 160x120 grayscale, 4-frame stack
- **Action space**: MultiDiscrete([5, 3, 3]) — movement, attack, modifier
- **Reward shaping**: Score delta, progress, survival bonus, curriculum death penalties (9 levels)
- **Exploration**: Entropy bonus, sticky actions (25%), episodic scroll novelty bonus
- **Plasticity**: Weight decay (1e-4)

## Setup

### Requirements

- Docker with NVIDIA Container Toolkit (for GPU training)
- A Metal Slug 6 ROM (Atomiswave format, not included)

### Build

```bash
docker build -t metalslug-rl .
```

### Train

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  --rm -d -p 5900:5900 -p 6080:6080 \
  -v /path/to/your/games:/games \
  -v $(pwd)/outputs:/app/outputs \
  -e RETROARCH_CORE=flycast_libretro \
  -e RETROARCH_CONTENT=/games/mslug6.zip \
  -e TIMESTEPS=100000000 -e DEVICE=cuda \
  --name mslug6 metalslug-rl
```

View in browser: http://localhost:6080/vnc.html

### Monitor

```bash
# Follow training logs
docker logs -f mslug6

# Or read the training stdout log directly
tail -f outputs/runs/PPO_*/logs/training_stdout.log
```

## Key Training Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESTEPS` | `100000000` | Total training timesteps |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `CHECKPOINT_EVERY` | `1000000` | Save checkpoint every N steps |
| `N_STEPS` | `2048` | Rollout steps per PPO update |
| `BATCH_SIZE` | `256` | Mini-batch size |
| `LEARNING_RATE` | `5e-5` | Learning rate (linear schedule) |
| `ENT_COEF` | `0.02` | Entropy coefficient |
| `CLIP_RANGE` | `0.1` | PPO clip range (linear schedule) |
| `CLIP_RANGE_VF` | `-1` | Value function clipping (-1 = disabled) |
| `WEIGHT_DECAY` | `1e-4` | Adam weight decay |
| `FRAME_SKIP` | `3` | Frames to skip per action |
| `RESUME` | (empty) | Path to checkpoint to resume from |
| `NUM_ENVS` | `1` | Parallel environments (max 2 for Flycast) |

See `CONTAINER.md` for the full environment variable reference.

## Project Structure

```
src/
  env/mslug_env.py      # Gymnasium environment (screen capture, RAM reads, rewards)
  env/rewards.py         # Reward computation
  env/ram_decode.py      # BCD score decoding
  impala_cnn.py          # IMPALA ResNet feature extractor
  agents/                # PPO and random agent wrappers
scripts/
  train_ppo.py           # Main training script with callbacks and crash recovery
  run_virtual_desktop.sh # Container entrypoint (Xvfb, RetroArch, auto-boot)
  eval_ppo.py            # Model evaluation
config/
  retroarch-container.cfg # Headless RetroArch config
```

## Experiment Log

See [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for a detailed training journal documenting problems encountered, root cause analysis, fixes applied, and results across all training runs.

## What This Repo Does NOT Include

- Game ROM files
- Emulator cores (built in Docker image)
- Saved game states (created at runtime)
