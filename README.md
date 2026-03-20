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

See `CONTAINER.md` for Docker build and run instructions.

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

## What this repo does NOT include

- Game content files
- Emulator cores
- Saved game-state files
