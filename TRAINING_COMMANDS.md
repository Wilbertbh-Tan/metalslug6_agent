# Training Commands Reference

## Basic Training Command

```bash
python scripts/train_ppo.py --timesteps 100000 --device cuda
```

## Common Training Options

### Basic Training
```bash
# Train for 100k steps on GPU
python scripts/train_ppo.py --timesteps 100000 --device cuda

# Train for 100k steps on CPU
python scripts/train_ppo.py --timesteps 100000 --device cpu

# Resume from checkpoint
python scripts/train_ppo.py --timesteps 100000 --device cuda --resume outputs/models/ppo_mslug6_10000_steps.zip
```

### Training with Checkpoints
```bash
# Save checkpoint every 5000 steps
python scripts/train_ppo.py --timesteps 100000 --device cuda --checkpoint-every 5000

# Save checkpoint every 10000 steps (default)
python scripts/train_ppo.py --timesteps 100000 --device cuda --checkpoint-every 10000
```

### Continuous Training (until interrupted)
```bash
# Train in chunks of 50000 steps until Ctrl+C
python scripts/train_ppo.py --run-until-interrupt --device cuda --chunk 50000

# Train in chunks of 100000 steps
python scripts/train_ppo.py --run-until-interrupt --device cuda --chunk 100000
```

### Training with Fast Forward
```bash
# Enable fast forward (if configured in RetroArch)
python scripts/train_ppo.py --timesteps 100000 --device cuda --fast-forward-key space

# Fast forward on every reset
python scripts/train_ppo.py --timesteps 100000 --device cuda --fast-forward-key space --fast-forward-on-reset
```

### Training with Verbose Logging
```bash
# Level 0: Quiet (minimal output)
python scripts/train_ppo.py --timesteps 100000 --device cuda --verbose-level 0

# Level 1: Training summary (default)
python scripts/train_ppo.py --timesteps 100000 --device cuda --verbose-level 1

# Level 2: Environment + score detail
python scripts/train_ppo.py --timesteps 100000 --device cuda --verbose-level 2

# Level 3: Maximum debug output
python scripts/train_ppo.py --timesteps 100000 --device cuda --verbose-level 3
```

### Training with Episode Control
```bash
# Stop after 50000 episodes
python scripts/train_ppo.py --timesteps 2000000000 --device cuda --target-episodes 50000

# Record video every 10000 episodes
python scripts/train_ppo.py --timesteps 100000 --device cuda --video-every-episodes 10000
```

### Training with Custom PPO Parameters
```bash
# Custom learning rate
python scripts/train_ppo.py --timesteps 100000 --device cuda --learning-rate 1e-4

# Custom entropy coefficient (exploration)
python scripts/train_ppo.py --timesteps 100000 --device cuda --ent-coef 0.01

# Custom batch size
python scripts/train_ppo.py --timesteps 100000 --device cuda --batch-size 128
```

### Training with Stall Detection
```bash
# Warn if no episodes complete within 10000 steps
python scripts/train_ppo.py --timesteps 100000 --device cuda --stall-threshold 10000

# Disable stall warnings
python scripts/train_ppo.py --timesteps 100000 --device cuda --stall-threshold 0
```

## Training Inside Docker Container

```bash
# Start training in running container
docker exec -d mslug6 python3 /app/scripts/train_ppo.py --timesteps 100000 --device cuda

# Start training with all options
docker exec -d mslug6 python3 /app/scripts/train_ppo.py \
  --timesteps 100000 \
  --device cuda \
  --checkpoint-every 5000 \
  --score-log-every 500 \
  --stall-threshold 5000 \
  --resume /app/outputs/models/ppo_mslug6_10000_steps.zip

# Interactive training (see output in real-time)
docker exec -it mslug6 python3 /app/scripts/train_ppo.py --timesteps 100000 --device cuda
```

## Monitoring Training

### View Training Logs
```bash
# View all training output
cat outputs/logs/training_stdout.log

# Follow training in real-time
tail -f outputs/logs/training_stdout.log

# View last 100 lines
tail -n 100 outputs/logs/training_stdout.log

# Filter for monitor lines (progress updates)
tail -f outputs/logs/training_stdout.log | grep -E "\[monitor\]|\[episode\]"

# Filter for stalls
tail -f outputs/logs/training_stdout.log | grep -i "stall"

# Filter for errors
tail -f outputs/logs/training_stdout.log | grep -iE "error|exception|traceback"
```

### Use Monitoring Script
```bash
# Monitor training for issues (stalls, errors, n/a values)
./scripts/monitor_training.sh

# Monitor specific log file
./scripts/monitor_training.sh outputs/logs/training_stdout.log
```

### Check Training Status
```bash
# Check if training process is running
ps aux | grep train_ppo

# Check latest checkpoint
ls -lht outputs/models/ | head -5

# TensorBoard (metrics for all runs)
tensorboard --logdir outputs/runs --port 6006
# Or use the monitor script:
# ./scripts/monitor_training.sh --tensorboard
```

## Common Issues and Solutions

### Training Shows "n/a" Values
- **Cause**: Missing episode data (no episodes completed yet, or episode ended without score)
- **Solution**: Wait for episodes to complete, or check if game is stuck in transition

### Training Stalls (No Episodes Completing)
- **Cause**: Game stuck in transition, RetroArch crashed, or environment issue
- **Solution**:
  - Check container logs: `docker logs mslug6`
  - Check if RetroArch is running
  - Restart container if needed
  - Lower `--stall-threshold` to detect earlier

### Training Stops Unexpectedly
- **Check logs for errors**: `tail -100 outputs/logs/training_stdout.log | grep -i error`
- **Check if container is running**: `docker ps -a --filter name=mslug6`
- **Check GPU memory**: `nvidia-smi` (if using GPU)
- **Check disk space**: `df -h`

## Full Example Training Command

```bash
python scripts/train_ppo.py \
  --timesteps 100000 \
  --device cuda \
  --checkpoint-every 5000 \
  --score-log-every 500 \
  --stall-threshold 5000 \
  --verbose-level 1 \
  --resume outputs/models/ppo_mslug6_10000_steps.zip
```
