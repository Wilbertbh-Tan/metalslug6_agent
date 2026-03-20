# Running training in a container

You can run Metal Slug training inside Docker so it doesn’t use your main display or keyboard. The default container launcher now starts a virtual desktop session (Xvfb + Openbox + optional VNC) and runs RetroArch + training inside it.

## Prerequisites

- Docker installed and running
- For full training: RetroArch **core** and **game content** (game content file). You must provide these; they are not included.
- **For GPU training (NVIDIA):** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host

## Build

```bash
docker build -t metalslug-rl .
```

## Quick test (no game content)

Verify the container and headless env work without mounting any game files:

```bash
docker run --rm metalslug-rl /app/scripts/test_headless_container.sh
```

You should see "Headless container test passed." If that works, the image and Xvfb + Python setup are fine.

## Run (training only, RetroArch on host)

If you prefer to run RetroArch on your **host** and only run the training script in Docker, that won’t work as-is: the container cannot see your host screen or send keys to the host. So for “background” training you have two options:

1. **Run everything in the container** (recommended): RetroArch + training inside the same container (see below).
2. **Run on host with “send keys to RetroArch only”**: On macOS you can use Quartz to send key events to RetroArch’s PID so your keyboard isn’t affected; the game still runs on the host.

## GPU training (NVIDIA)

The container image is based on `nvidia/cuda:12.2.2-runtime-ubuntu22.04` and installs CUDA-enabled PyTorch. To use your GPU:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host:

   ```bash
   # Ubuntu/Debian
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. Add `--gpus all` to your `docker run` command:

   ```bash
   docker run --gpus all --rm -it \
     -p 5900:5900 -p 6080:6080 \
     -e RETROARCH_CORE="/usr/lib/libretro/flycast_libretro.so" \
     -e RETROARCH_CONTENT="/games/mslug6.zip" \
     -e RUN_UNTIL_INTERRUPT=1 \
     -v /your/games:/games \
     -v $(pwd)/outputs:/app/outputs \
     --name mslug6 \
     metalslug-rl
   ```

3. Training auto-detects CUDA and prints the GPU name at startup:

   ```
   Training device: cuda
     GPU: NVIDIA GeForce RTX 3080
   ```

   To force CPU even when a GPU is available: `-e DEVICE=cpu` or pass `--device cpu` to the training script.

Without `--gpus all` (or on machines without NVIDIA GPUs), training falls back to CPU automatically.

## Run (RetroArch + training in virtual desktop container)

1. The FBNeo libretro core is baked into the image at `/usr/lib/libretro/fbneo_libretro.so`. You only need your game content file (not included in this repository).

2. The container ships with a headless RetroArch config at `/etc/retroarch.cfg` that already enables network commands (port 55355), uses the `gl` video driver with Mesa software rendering, and disables audio. No extra config mount is needed.

3. Run the container with env and mounts:

   ```bash
   docker run --rm -it \
     -p 5900:5900 \
     -p 6080:6080 \
     -e RETROARCH_CORE="/usr/lib/libretro/fbneo_libretro.so" \
     -e RETROARCH_CONTENT="/games/content.file" \
     -e TIMESTEPS=100000 \
     -v /your/games:/games \
     metalslug-rl
   ```

   Adjust `RETROARCH_CONTENT` so it points at the game content file inside the container (e.g. `/games/content.file`).

4. **Capture region**: The virtual-desktop launcher auto-detects RetroArch window geometry and exports `CAPTURE_LEFT/TOP/WIDTH/HEIGHT` when possible. Avoid forcing `CAPTURE_LEFT/TOP` unless you need a manual override.

5. **View in browser (noVNC)**: The container starts a noVNC server. Open `http://localhost:6080/vnc.html` in any browser, click **Connect** (no password), and you can see the virtual desktop live. Map the port with `-p 6080:6080`.

6. **Outputs**: To keep logs and checkpoints, mount a volume:

   ```bash
   -v $(pwd)/outputs:/app/outputs
   ```

## Env vars

| Variable              | Default   | Description |
|-----------------------|-----------|-------------|
| `DISPLAY`             | `:99`     | Xvfb display. |
| `CAPTURE_LEFT`        | auto      | Left of capture region; auto-detected from RetroArch window when possible. |
| `CAPTURE_TOP`         | auto      | Top of capture region; auto-detected from RetroArch window when possible. |
| `CAPTURE_WIDTH`       | auto/640  | Capture width; auto from RetroArch window or fallback 640. |
| `CAPTURE_HEIGHT`      | auto/480  | Capture height; auto from RetroArch window or fallback 480. |
| `MSLUG_HEADLESS`      | (set by script) | Skips “click RetroArch” prompt and 5s delay. |
| `START_VNC`           | `1`       | Start embedded VNC server (`x11vnc`) for observing the virtual desktop. |
| `VNC_PORT`            | `5900`    | VNC port exposed by the container. |
| `NOVNC_PORT`          | `6080`    | noVNC web port. Open `http://localhost:6080/vnc.html` in a browser to view the virtual desktop. |
| `SCREEN_GEOMETRY`     | `1280x720x24` | Virtual desktop geometry for Xvfb. |
| `TIMESTEPS`           | `100000`  | Passed to `train_ppo --timesteps`. |
| `RESUME`              | (empty)   | If set, passed as `--resume` to load a checkpoint. |
| `RETROARCH_CORE`      | (empty)   | Path inside container to core .so. If set with `RETROARCH_CONTENT`, RetroArch is started automatically. |
| `RETROARCH_CONTENT`   | (empty)   | Path inside container to game content file. |
| `FAST_FORWARD_KEY`   | (empty)   | Key to press once to toggle RetroArch fast forward in headless (e.g. `space`). Focus is set to RetroArch first. |
| `RUN_UNTIL_INTERRUPT`| (empty)   | If set (e.g. `1`), training runs until Ctrl+C with no step cap. |
| `CHUNK`              | (empty)   | Steps per chunk when `RUN_UNTIL_INTERRUPT` is set (default in script: 50000). |
| `VERBOSE_LEVEL`      | (empty)   | Training verbosity level `0-3` (`0` quiet, `1` summary, `2` env+score detail, `3` max debug). |
| `VERBOSE`            | (empty)   | Legacy switch that forces detailed logs (`--verbose`). |
| `NO_SCORE_LOG_STDOUT`| (empty)   | If set, disable periodic score monitor lines in stdout. |
| `TARGET_EPISODES`    | (empty)   | Stop automatically after N episodes (e.g. `50000`). |
| `VIDEO_EVERY_EPISODES` | (empty) | Save an eval video every N episodes (e.g. `10000`). |
| `VIDEO_MAX_STEPS`    | (empty)   | Max steps per saved video clip (default `800`). |
| `VIDEO_FPS`          | (empty)   | FPS for saved clips (default `20`). |
| `VIDEO_DIR`          | (empty)   | Output folder for clips (default `/app/outputs/videos`). |
| `AUTO_BOOT_GAME`     | `1`       | Send coin+start keys to get past the title screen and create an initial save state. Set to `0` if you mount your own save state. |
| `BOOT_WAIT`          | `15`      | Seconds to wait for BIOS + ROM to load before sending boot keys. Increase if the game needs more time to boot. |
| `BOOT_KEYS`          | `Shift_R:2:Return:4:x:4:x:4` | Key sequence for auto-boot. Format: `key:delay:key:delay:...` (xdotool key names). Adjust for your game. |

## Fast forward (run game faster)

To get more training steps per second, run RetroArch in fast forward so the emulator is not limited to real-time.

- **In RetroArch:** Settings → Input → Hotkeys → set **Fast Forward** to a key (e.g. **Tab**). When that key is held (or toggled), the game runs uncapped.
- **When training on host:** After the 5-second countdown, the script can send the key once if you pass `--fast-forward-key tab` (use the same key you set in RetroArch). Make sure RetroArch has focus when the countdown ends.
- **Config (optional):** In `retroarch.cfg` you can set `fastforward_ratio = -1.0` (uncapped). If you use a “training” config that has this set, fast forward can be on by default when you launch for training.

**Host:** `python scripts/train_ppo.py --run-until-interrupt --fast-forward-key space`

**Headless (Docker):** Set `FAST_FORWARD_KEY=space` (and optionally `RUN_UNTIL_INTERRUPT=1`). The script focuses the RetroArch window then sends the key before training.

## Example full run

```bash
docker run --rm -it \
  -e RETROARCH_CORE="/usr/lib/libretro/fbneo_libretro.so" \
  -e RETROARCH_CONTENT="/games/content.file" \
  -e TIMESTEPS=50000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/my_retroarch_config:/cfg \
  metalslug-rl
```

(The FBNeo core is compiled into the image at `/usr/lib/libretro/fbneo_libretro.so`.)

## Episode-based stop + periodic videos

If you want to train for episode counts (instead of timesteps) and save a video every fixed number of episodes:

```bash
docker run --rm -it \
  -p 5900:5900 \
  -e RETROARCH_CORE="/usr/lib/libretro/fbneo_libretro.so" \
  -e RETROARCH_CONTENT="/games/content.file" \
  -e TARGET_EPISODES=50000 \
  -e TIMESTEPS=2000000000 \
  -e VIDEO_EVERY_EPISODES=10000 \
  -e VIDEO_MAX_STEPS=800 \
  -e VIDEO_FPS=20 \
  -e FAST_FORWARD_KEY=space \
  -e FAST_FORWARD_MODE=set_once_persist \
  -v $(pwd)/outputs:/app/outputs \
  -v /your/games:/games \
  metalslug-rl
```

Videos are saved under `outputs/videos/` on your host (because `outputs` is mounted).

## Debug checklist for transition errors

Use this quick checklist when training gets stuck in `TRANSITION (not in game)`:

1. Rebuild image without cache:

   ```bash
   docker build --no-cache -t metalslug-rl .
   ```

2. Run with debug logs and confirm startup probe prints non-black frame stats:
   - look for: `[startup-check] frame_stats=min:... max:... mean:...`
   - if max is very low (for example <= 5), capture is wrong.

3. Confirm RetroArch window detection line appears:
   - `RetroArch window detected: id=... left=... top=... width=... height=...`

4. Do not force `CAPTURE_LEFT/TOP` unless needed.

5. If needed, watch virtual desktop via noVNC at `http://localhost:6080/vnc.html` (or VNC client at `localhost:5900`) to verify gameplay is visible.

### Verbosity presets

- Quiet:
  - `-e VERBOSE_LEVEL=0 -e NO_SCORE_LOG_STDOUT=1`
- Normal:
  - `-e VERBOSE_LEVEL=1`
- Debug (recommended when validating checks/RAM):
  - `-e VERBOSE_LEVEL=3`
