# metalslug6_agent

PPO-based reinforcement learning agent for Metal Slug 6 via a custom Gymnasium environment.

## Project Structure

- `src/env/` — Gymnasium environment, rewards, RAM decoding
- `src/agents/` — Agent classes (RandomAgent, PPOAgent)
- `scripts/` — Training, evaluation, and utility scripts
- `config/` — RetroArch container configuration
- `tests/` — Unit tests

## Quick Start

### Test environment (random agent, max verbosity)
```bash
python scripts/random_agent.py --max-episodes 2 --verbose-level 2 --disable-in-game-checks
```

### Train
```bash
python scripts/train_ppo.py --timesteps 100000
```

### Evaluate
```bash
python scripts/eval_ppo.py --model outputs/models/ppo_mslug6_final.zip --episodes 3
```

## Container

See `CONTAINER.md` for full Docker details (env vars, GPU setup, debugging).

### Build

```bash
docker build -t metalslug-rl .
```

### Train

```bash
docker run --gpus all --rm -it \
  -p 5900:5900 -p 6080:6080 \
  -e RETROARCH_CORE="/usr/lib/libretro/flycast_libretro.so" \
  -e RETROARCH_CONTENT="/games/mslug6.zip" \
  -e RUN_UNTIL_INTERRUPT=1 \
  -v /path/to/your/games:/games \
  -v $(pwd)/outputs:/app/outputs \
  --name mslug6 \
  metalslug-rl
```

Drop `--gpus all` on machines without an NVIDIA GPU (training falls back to CPU).

The emulator runs uncapped by default — no fast-forward flag needed.

Training parameters are set via `-e` env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `TIMESTEPS` | `100000` | Total training timesteps |
| `RUN_UNTIL_INTERRUPT` | (empty) | If set, train indefinitely until Ctrl+C |
| `CHUNK` | `50000` | Steps per chunk when using `RUN_UNTIL_INTERRUPT` |
| `CHECKPOINT_EVERY` | `10000` | Save a checkpoint every N steps |
| `LEARNING_RATE` | `3e-4` | PPO learning rate |
| `ENT_COEF` | `0.05` | Entropy coefficient (exploration) |
| `N_STEPS` | `512` | Rollout steps per PPO update |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `N_EPOCHS` | `4` | PPO epochs per update |
| `GAMMA` | `0.99` | Discount factor |
| `SCORE_SCALE` | `0.005` | Score reward scaling |
| `DEATH_PENALTY` | `-5.0` | Penalty on death |
| `TIME_PENALTY` | `-0.005` | Per-step time penalty |
| `FRAME_SKIP` | `4` | Frames to skip per action |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `RESUME` | (empty) | Path to checkpoint to resume from |

Example with custom params:

```bash
docker run --gpus all --rm -it \
  -p 5900:5900 -p 6080:6080 \
  -e RETROARCH_CORE="/usr/lib/libretro/flycast_libretro.so" \
  -e RETROARCH_CONTENT="/games/mslug6.zip" \
  -e RUN_UNTIL_INTERRUPT=1 \
  -e LEARNING_RATE=1e-4 \
  -e ENT_COEF=0.01 \
  -e N_STEPS=1024 \
  -e BATCH_SIZE=128 \
  -v /path/to/your/games:/games \
  -v $(pwd)/outputs:/app/outputs \
  --name mslug6 \
  metalslug-rl
```

### Running eval inside the container

```bash
# Inside container — 2 episodes, full step detail
docker exec mslug6 python3 scripts/eval_ppo.py --random --episodes 2 --verbose-level 2 --no-calibration --disable-in-game-checks

# Run indefinitely until Ctrl+C
docker exec -it mslug6 python3 scripts/eval_ppo.py --random --episodes 0 --verbose-level 2 --no-calibration --disable-in-game-checks

# Episode summaries only (no per-step spam)
docker exec mslug6 python3 scripts/eval_ppo.py --random --episodes 0 --verbose-level 1 --no-calibration --disable-in-game-checks
```

Note: the 5-second countdown + "Click on RetroArch" message shows because `MSLUG_HEADLESS` isn't set in the exec shell. That's harmless — it's just a delay. If you want to skip it, add `-e MSLUG_HEADLESS=1`:

```bash
docker exec -e MSLUG_HEADLESS=1 mslug6 python3 scripts/eval_ppo.py --random --episodes 0 --verbose-level 2 --no-calibration --disable-in-game-checks
```

### Viewing Container Logs

```bash
# View all logs
docker logs mslug6

# Follow logs in real-time
docker logs -f mslug6

# View last 50 lines
docker logs --tail 50 mslug6

# Filter for auto-boot messages
docker logs mslug6 2>&1 | grep -i "auto-boot\|sent:\|gameplay\|save state"

# Filter for key sequence
docker logs mslug6 2>&1 | grep -E "sent:|key sequence|BOOT_KEYS"

# View debug logs (instrumentation)

# Check container status
docker ps -a --filter name=mslug6
```

## What this repo does NOT include

- Game content files
- Emulator cores
- Saved game-state files
