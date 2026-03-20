#!/bin/bash
# Run training in a virtual desktop (Xvfb + Openbox + optional VNC).
# Starts RetroArch in that desktop and auto-detects capture region when possible.
# Supports NUM_ENVS > 1 for parallel environments (each gets its own display + RetroArch).

set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export MSLUG_HEADLESS=1
NUM_ENVS="${NUM_ENVS:-1}"
# Flycast (Dreamcast) is resource-heavy; 3+ instances often cause bus error on the 3rd. Cap at 2.
if [ -n "${RETROARCH_CORE:-}" ] && echo "${RETROARCH_CORE}" | grep -qi flycast; then
  if [ "$NUM_ENVS" -gt 2 ] 2>/dev/null; then
    echo "Warning: Capping NUM_ENVS at 2 for Flycast core (3+ instances can cause bus error)."
    NUM_ENVS=2
  fi
fi
SCREEN_GEOMETRY="${SCREEN_GEOMETRY:-1280x720x24}"
VNC_PORT="${VNC_PORT:-5900}"
START_VNC="${START_VNC:-1}"
RETROARCH_LOG="${RETROARCH_LOG:-/tmp/retroarch.log}"
ALLOW_NO_WINDOW="${ALLOW_NO_WINDOW:-0}"

echo "Starting $NUM_ENVS environment(s)..."

# Start Xvfb for each environment: display :99, :100, :101, ...
for _env_i in $(seq 0 $((NUM_ENVS - 1))); do
  _env_display=":$((99 + _env_i))"
  _env_log="/tmp/xvfb_env_${_env_i}.log"
  Xvfb "$_env_display" -screen 0 "$SCREEN_GEOMETRY" +extension MIT-SHM >"$_env_log" 2>&1 &
  echo "Started Xvfb on $_env_display"
done
sleep 2

# Primary display for pyautogui/Xlib and VNC
export DISPLAY=":99"

# So pyautogui/Xlib can connect.
touch "$HOME/.Xauthority"
for _env_i in $(seq 0 $((NUM_ENVS - 1))); do
  _env_display=":$((99 + _env_i))"
  xauth add "$_env_display" . "$(mcookie 2>/dev/null || echo "0")"
done

# Start a lightweight desktop session on each display.
for _env_i in $(seq 0 $((NUM_ENVS - 1))); do
  _env_display=":$((99 + _env_i))"
  DISPLAY="$_env_display" openbox >/tmp/openbox_env_${_env_i}.log 2>&1 &
done
sleep 1

# VNC only on the primary display (:99)
if [ "$START_VNC" != "0" ]; then
  x11vnc \
    -display ":99" \
    -rfbport "$VNC_PORT" \
    -nopw \
    -shared \
    -forever \
    -quiet \
    >/tmp/x11vnc.log 2>&1 &
  echo "Virtual desktop ready on DISPLAY=:99 (VNC port $VNC_PORT, no password)."

  # Start noVNC so the virtual desktop is viewable in any browser.
  NOVNC_PORT="${NOVNC_PORT:-6080}"
  NOVNC_DIR=""
  for d in /usr/share/novnc /usr/share/novnc/utils/../ /snap/novnc/current; do
    [ -f "$d/vnc.html" ] || [ -f "$d/vnc_lite.html" ] && NOVNC_DIR="$d" && break
  done
  if [ -n "$NOVNC_DIR" ]; then
    websockify --web="$NOVNC_DIR" "$NOVNC_PORT" localhost:"$VNC_PORT" \
      >/tmp/novnc.log 2>&1 &
    echo "noVNC available at http://localhost:$NOVNC_PORT/vnc.html"
  else
    echo "Warning: noVNC web directory not found; skipping noVNC."
  fi
fi

if [ -n "${RETROARCH_CORE:-}" ] && [ -n "${RETROARCH_CONTENT:-}" ]; then
  # Force Mesa software rendering and X11 backend for SDL2.
  export LIBGL_ALWAYS_SOFTWARE=1
  export SDL_VIDEODRIVER=x11

  # Resolve core name to full path if not already a file path.
  _CORE="$RETROARCH_CORE"
  if [ ! -f "$_CORE" ]; then
    # Try appending .so and looking in standard libretro directory.
    for _candidate in \
      "/usr/lib/libretro/${_CORE}.so" \
      "/usr/lib/libretro/${_CORE}" \
      "/usr/local/lib/libretro/${_CORE}.so" \
      "/usr/local/lib/libretro/${_CORE}"; do
      if [ -f "$_candidate" ]; then
        _CORE="$_candidate"
        break
      fi
    done
  fi

  # ------------------------------------------------------------------
  # Launch RetroArch + auto-boot for each environment sequentially.
  # env 0 → display :99, cmd port 55355
  # env 1 → display :100, cmd port 55356
  # env N → display :$((99+N)), cmd port $((55355+N))
  # ------------------------------------------------------------------
  for _env_i in $(seq 0 $((NUM_ENVS - 1))); do
    _env_display=":$((99 + _env_i))"
    _env_cmd_port=$((55355 + _env_i))
    _env_ra_log="/tmp/retroarch_env_${_env_i}.log"
    export DISPLAY="$_env_display"
    echo ""
    echo "=== Environment $_env_i: display=$_env_display cmd_port=$_env_cmd_port ==="

    # Create per-env RetroArch config override with unique network_cmd_port
    # and per-env save/state directories so instances don't conflict.
    _env_data_dir="/games/env_${_env_i}"
    mkdir -p "$_env_data_dir" 2>/dev/null || true
    # Symlink BIOS/system files from /games into per-env directory
    for _bios_file in /games/dc_boot.bin /games/dc_flash.bin /games/dc /games/Flycast; do
      if [ -e "$_bios_file" ] && [ ! -e "$_env_data_dir/$(basename "$_bios_file")" ]; then
        ln -sf "$_bios_file" "$_env_data_dir/$(basename "$_bios_file")" 2>/dev/null || true
      fi
    done
    _env_cfg="/tmp/retroarch_env_${_env_i}.cfg"
    cat > "$_env_cfg" <<ENVCFG
network_cmd_port = "${_env_cmd_port}"
savestate_directory = "${_env_data_dir}"
savefile_directory = "${_env_data_dir}"
ENVCFG

    echo "Starting RetroArch env $_env_i with core=$_CORE content=$RETROARCH_CONTENT port=$_env_cmd_port"
    retroarch -v --config /etc/retroarch.cfg --appendconfig "$_env_cfg" -L "$_CORE" "$RETROARCH_CONTENT" ${RETROARCH_APPEND_ARGS:-} >"$_env_ra_log" 2>&1 &
    RA_PID=$!
    sleep 6

    RA_WIN_ID=""
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
      RA_WIN_ID="$(xdotool search --pid "$RA_PID" 2>/dev/null | head -n 1 || true)"
      if [ -z "$RA_WIN_ID" ]; then
        RA_WIN_ID="$(xdotool search --name "RetroArch" 2>/dev/null | head -n 1 || true)"
      fi
      if [ -z "$RA_WIN_ID" ]; then
        RA_WIN_ID="$(wmctrl -lp 2>/dev/null | awk -v pid="$RA_PID" '$3 == pid {print $1; exit}' || true)"
      fi
      if [ -n "$RA_WIN_ID" ]; then
        break
      fi
      sleep 0.5
    done

    if [ -n "$RA_WIN_ID" ]; then
      xdotool windowactivate "$RA_WIN_ID" 2>/dev/null || true
      sleep 0.5
      eval "$(xdotool getwindowgeometry --shell "$RA_WIN_ID" 2>/dev/null || true)"

      # Parse screen dimensions from SCREEN_GEOMETRY (e.g. 1280x720x24).
      SCREEN_W="${SCREEN_GEOMETRY%%x*}"
      _rest="${SCREEN_GEOMETRY#*x}"
      SCREEN_H="${_rest%%x*}"

      # Clamp detected geometry to valid screen bounds.
      _x="${X:-0}"; [ "$_x" -lt 0 ] 2>/dev/null && _x=0
      _y="${Y:-0}"; [ "$_y" -lt 0 ] 2>/dev/null && _y=0
      _w="${WIDTH:-$SCREEN_W}"; [ "$((_x + _w))" -gt "$SCREEN_W" ] && _w="$((SCREEN_W - _x))"
      _h="${HEIGHT:-$SCREEN_H}"; [ "$((_y + _h))" -gt "$SCREEN_H" ] && _h="$((SCREEN_H - _y))"

      # Set capture region from the first env (env 0) — all envs have same geometry.
      if [ "$_env_i" -eq 0 ]; then
        if [ -z "${CAPTURE_LEFT:-}" ]; then
          export CAPTURE_LEFT="$_x"
        fi
        if [ -z "${CAPTURE_TOP:-}" ]; then
          export CAPTURE_TOP="$_y"
        fi
        if [ -z "${CAPTURE_WIDTH:-}" ]; then
          export CAPTURE_WIDTH="$_w"
        fi
        if [ -z "${CAPTURE_HEIGHT:-}" ]; then
          export CAPTURE_HEIGHT="$_h"
        fi
      fi
      echo "RetroArch env $_env_i window detected: id=$RA_WIN_ID geometry=${_x},${_y},${_w}x${_h}"

      # Auto-boot: send coin + start keys to get past title screen, then save state.
      if [ "${AUTO_BOOT_GAME:-1}" != "0" ]; then
        BOOT_WAIT="${BOOT_WAIT:-15}"
        BOOT_KEYS="${BOOT_KEYS:-Shift_R:2:Return:4:z:3:z:2}"
        echo "Auto-boot env $_env_i: waiting ${BOOT_WAIT}s for BIOS + ROM to load..."
        sleep "$BOOT_WAIT"
        echo "Auto-boot env $_env_i: sending key sequence to start game ($BOOT_KEYS)..."
        xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
        sleep 1
        IFS=':' read -ra _parts <<< "$BOOT_KEYS"
        _ki=0
        _key_count=0
        _any_key_failed=0
        while [ "$_ki" -lt "${#_parts[@]}" ]; do
          _key="${_parts[$_ki]}"
          xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
          sleep 0.1
          _xdotool_exit=0
          xdotool key --window "$RA_WIN_ID" "$_key" 2>&1 || _xdotool_exit=$?
          if [ "$_xdotool_exit" -ne 0 ]; then
            _any_key_failed=1
            echo "  WARNING: xdotool exit code $_xdotool_exit for key $_key"
            if [ "$_key" = "z" ]; then
              echo "  Retrying 'z' key after xdotool failure..."
              sleep 0.3
              xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
              sleep 0.1
              _xdotool_exit=0
              xdotool key --window "$RA_WIN_ID" "$_key" 2>&1 || _xdotool_exit=$?
              if [ "$_xdotool_exit" -ne 0 ]; then
                echo "  ERROR: xdotool still failed on retry for key $_key"
              fi
            fi
          fi
          echo "  sent: $_key"
          _key_count=$((_key_count + 1))
          if [ "$_key" = "z" ]; then
            _z_delay="${BOOT_Z_DELAY:-0.5}"
            sleep "$_z_delay"
          else
            sleep 0.2
          fi
          _ki=$((_ki + 1))
          if [ "$_ki" -lt "${#_parts[@]}" ]; then
            _delay="${_parts[$_ki]}"
            sleep "$_delay"
            _ki=$((_ki + 1))
          fi
        done
        _post_keys_wait="${BOOT_POST_KEYS_WAIT:-3}"
        echo "Auto-boot env $_env_i: waiting ${_post_keys_wait}s after key sequence before checking gameplay..."
        sleep "$_post_keys_wait"
        echo "Auto-boot env $_env_i: waiting for gameplay to start (polling RAM via port $_env_cmd_port)..."
        _attempt=0
        _gameplay_ready=0
        _consecutive_ready=0
        _required_consecutive="${BOOT_CONSECUTIVE_CHECKS:-10}"
        while [ "$_attempt" -lt 400 ] && [ "$_gameplay_ready" -eq 0 ]; do
          sleep 0.15
          _attempt=$((_attempt + 1))
          export SAVESTATE_POLL_ATTEMPT="$_attempt"
          _check_result=$(python3 -c "
import sys
sys.path.insert(0, '/app')
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
import os
os.environ.setdefault('DISPLAY', '${_env_display}')
region = CaptureRegion(
    left=int(os.environ.get('CAPTURE_LEFT', '${CAPTURE_LEFT:-0}')),
    top=int(os.environ.get('CAPTURE_TOP', '${CAPTURE_TOP:-0}')),
    width=int(os.environ.get('CAPTURE_WIDTH', '${CAPTURE_WIDTH:-640}')),
    height=int(os.environ.get('CAPTURE_HEIGHT', '${CAPTURE_HEIGHT:-480}'))
)
env = MetalSlugEnv(region=region, retroarch_cmd_port=${_env_cmd_port}, verbose=0)
gs = env._read_ram('003868D0', 1)
score = env._read_ram('003869BC', 4)
lives = env._read_ram('003868D1', 1)
if gs and len(gs) == 1 and gs[0] == 0x00:
    if score and len(score) == 4 and score == [0, 0, 0, 0]:
        if lives and len(lives) == 1 and lives[0] >= 0x02:
            print('READY')
            sys.exit(0)
print('NOT_READY')
sys.exit(1)
" 2>/dev/null || echo "NOT_READY")
          if [ "$_check_result" = "READY" ]; then
            _consecutive_ready=$((_consecutive_ready + 1))
            if [ "$_consecutive_ready" -ge "$_required_consecutive" ]; then
              _gameplay_ready=1
              break
            fi
          else
            _consecutive_ready=0
          fi
          if [ $((_attempt % 20)) -eq 0 ]; then
            echo "Auto-boot env $_env_i:   ...$_attempt polls, consecutive ready: $_consecutive_ready/$_required_consecutive, still waiting..."
          fi
        done
        if [ "$_gameplay_ready" -eq 1 ]; then
          sleep 0.5
          echo "Auto-boot env $_env_i: creating save state after gameplay verified..."
          xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
          sleep 0.2
          # Use network command (more reliable than F2 key) — target correct port
          _save_result=$(python3 -c "
import sys
sys.path.insert(0, '/app')
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
import os
os.environ.setdefault('DISPLAY', '${_env_display}')
region = CaptureRegion(
    left=int(os.environ.get('CAPTURE_LEFT', '${CAPTURE_LEFT:-0}')),
    top=int(os.environ.get('CAPTURE_TOP', '${CAPTURE_TOP:-0}')),
    width=int(os.environ.get('CAPTURE_WIDTH', '${CAPTURE_WIDTH:-640}')),
    height=int(os.environ.get('CAPTURE_HEIGHT', '${CAPTURE_HEIGHT:-480}'))
)
env = MetalSlugEnv(region=region, retroarch_cmd_port=${_env_cmd_port}, verbose=0)
try:
    env._send_retroarch_cmd('SAVE_STATE')
    print('SAVE_SENT')
except Exception as e:
    print(f'SAVE_ERROR:{e}')
" 2>/dev/null || echo "SAVE_ERROR")
          sleep 1.0
          # Verify save state file was created
          _new_states=$(python3 -c "
import glob
import os
dirs = ['/games/env_${_env_i}/Flycast', '/games/env_${_env_i}', '/games/Flycast', '/games']
states = []
for d in dirs:
    states.extend(glob.glob(os.path.join(d, '*.state')))
found = [f for f in states if os.path.exists(f) and os.path.getsize(f) > 0]
if found:
    latest = max(found, key=lambda f: os.path.getmtime(f))
    print(f'{latest}:{os.path.getsize(latest)}:{os.path.getmtime(latest)}')
else:
    print('NO_STATE')
" 2>/dev/null || echo "NO_STATE")
          if [ "$_new_states" != "NO_STATE" ] && [ -n "$_new_states" ]; then
            _state_file=$(echo "$_new_states" | cut -d: -f1)
            _state_size=$(echo "$_new_states" | cut -d: -f2)
            echo "Auto-boot env $_env_i: save state created: $_state_file ($_state_size bytes)"
          else
            echo "Auto-boot env $_env_i: WARNING - SAVE_STATE didn't create file, trying F2 key..."
            xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
            sleep 0.2
            xdotool key --window "$RA_WIN_ID" F2
            sleep 1.0
          fi
          echo "Auto-boot env $_env_i: save state process complete."
        else
          echo "Auto-boot env $_env_i: WARNING - gameplay not detected after $_attempt attempts, saving state anyway..."
          xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
          sleep 0.2
          xdotool key --window "$RA_WIN_ID" F2
        fi
      fi
    else
      if kill -0 "$RA_PID" 2>/dev/null; then
        echo "Warning: RetroArch env $_env_i is running but no X11 window was found."
        wmctrl -lp 2>/dev/null || true
      else
        echo "Error: RetroArch env $_env_i exited before a window was created."
      fi
      if [ -f "$_env_ra_log" ]; then
        echo "RetroArch env $_env_i log tail:"
        tail -n 40 "$_env_ra_log" || true
      fi
      if [ "$ALLOW_NO_WINDOW" != "1" ]; then
        echo "Aborting: no RetroArch window for env $_env_i means capture will likely be invalid."
        exit 1
      fi
      echo "ALLOW_NO_WINDOW=1 set; continuing with fallback capture origin."
    fi
  done  # end per-env loop

  # Reset DISPLAY to primary for the training script
  export DISPLAY=":99"
  echo ""
  echo "All $NUM_ENVS environment(s) initialized."
fi

export CAPTURE_LEFT="${CAPTURE_LEFT:-0}"
export CAPTURE_TOP="${CAPTURE_TOP:-0}"
export CAPTURE_WIDTH="${CAPTURE_WIDTH:-640}"
export CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-480}"

# --- Mode selection ---
case "${MODE:-train}" in
  game-only)
    echo "Game-only mode: RetroArch + VNC running. No agent."
    echo "To start training:  docker exec -d <name> python3 /app/scripts/train_ppo.py --timesteps 100000 --device cuda"
    echo "To start eval:      docker exec -d <name> python3 /app/scripts/eval_ppo.py --model /app/outputs/models/<checkpoint>.zip --episodes 0"
    exec sleep infinity
    ;;
  eval)
    cd /app
    EVAL_ARGS=""
    [ -n "${EVAL_MODEL:-}" ] && EVAL_ARGS="$EVAL_ARGS --model $EVAL_MODEL"
    [ -n "${EVAL_EPISODES:-}" ] && EVAL_ARGS="$EVAL_ARGS --episodes $EVAL_EPISODES"
    [ -n "${EVAL_RANDOM:-}" ] && EVAL_ARGS="$EVAL_ARGS --random"
    [ -n "${VERBOSE_LEVEL:-}" ] && EVAL_ARGS="$EVAL_ARGS --verbose-level $VERBOSE_LEVEL"
    [ -n "${FAST_FORWARD_KEY:-}" ] && EVAL_ARGS="$EVAL_ARGS --fast-forward-key $FAST_FORWARD_KEY"
    [ -n "${FAST_FORWARD_MODE:-}" ] && EVAL_ARGS="$EVAL_ARGS --fast-forward-mode $FAST_FORWARD_MODE"
    exec python3 scripts/eval_ppo.py $EVAL_ARGS
    ;;
esac
# MODE=train (default) falls through to existing training block below

TIMESTEPS="${TIMESTEPS:-100000}"
RESUME="${RESUME:-}"
RUN_UNTIL_INTERRUPT="${RUN_UNTIL_INTERRUPT:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"

cd /app
ARGS="--timesteps $TIMESTEPS --checkpoint-every $CHECKPOINT_EVERY"
[ -n "$RESUME" ] && ARGS="$ARGS --resume $RESUME"
[ -n "$RUN_UNTIL_INTERRUPT" ] && ARGS="$ARGS --run-until-interrupt"
[ -n "${CHUNK:-}" ] && ARGS="$ARGS --chunk $CHUNK"
[ -n "${CALIBRATION_JSON:-}" ] && ARGS="$ARGS --calibration-json $CALIBRATION_JSON"
[ -n "${NO_CALIBRATION:-}" ] && ARGS="$ARGS --no-calibration"
[ -n "${VERBOSE_LEVEL:-}" ] && ARGS="$ARGS --verbose-level $VERBOSE_LEVEL"
[ -n "${VERBOSE:-}" ] && ARGS="$ARGS --verbose"
[ -n "${FAST_FORWARD_KEY:-}" ] && ARGS="$ARGS --fast-forward-key $FAST_FORWARD_KEY"
[ -n "${FAST_FORWARD_MODE:-}" ] && ARGS="$ARGS --fast-forward-mode $FAST_FORWARD_MODE"
[ -n "${FAST_FORWARD_ON_RESET:-}" ] && ARGS="$ARGS --fast-forward-on-reset"
[ -n "${FAST_FORWARD_EVERY_RESET:-}" ] && ARGS="$ARGS --fast-forward-every-reset"
[ -n "${LEARNING_RATE:-}" ] && ARGS="$ARGS --learning-rate $LEARNING_RATE"
[ -n "${ENT_COEF:-}" ] && ARGS="$ARGS --ent-coef $ENT_COEF"
[ -n "${N_STEPS:-}" ] && ARGS="$ARGS --n-steps $N_STEPS"
[ -n "${BATCH_SIZE:-}" ] && ARGS="$ARGS --batch-size $BATCH_SIZE"
[ -n "${N_EPOCHS:-}" ] && ARGS="$ARGS --n-epochs $N_EPOCHS"
[ -n "${GAMMA:-}" ] && ARGS="$ARGS --gamma $GAMMA"
[ -n "${GAE_LAMBDA:-}" ] && ARGS="$ARGS --gae-lambda $GAE_LAMBDA"
[ -n "${CLIP_RANGE_VF:-}" ] && ARGS="$ARGS --clip-range-vf $CLIP_RANGE_VF"
[ "${NO_LINEAR_SCHEDULE:-}" = "1" ] && ARGS="$ARGS --no-linear-schedule"
[ -n "${SCORE_LOG_EVERY:-}" ] && ARGS="$ARGS --score-log-every $SCORE_LOG_EVERY"
[ -n "${NO_SCORE_LOG_STDOUT:-}" ] && ARGS="$ARGS --no-score-log-stdout"
[ -n "${TARGET_EPISODES:-}" ] && ARGS="$ARGS --target-episodes $TARGET_EPISODES"
[ -n "${VIDEO_EVERY_EPISODES:-}" ] && ARGS="$ARGS --video-every-episodes $VIDEO_EVERY_EPISODES"
[ -n "${VIDEO_MAX_STEPS:-}" ] && ARGS="$ARGS --video-max-steps $VIDEO_MAX_STEPS"
[ -n "${VIDEO_FPS:-}" ] && ARGS="$ARGS --video-fps $VIDEO_FPS"
[ -n "${VIDEO_DIR:-}" ] && ARGS="$ARGS --video-dir $VIDEO_DIR"
[ -n "${DEVICE:-}" ] && ARGS="$ARGS --device $DEVICE"
[ -n "${STALL_THRESHOLD:-}" ] && ARGS="$ARGS --stall-threshold $STALL_THRESHOLD"
[ -n "${RUN_NAME:-}" ] && ARGS="$ARGS --run-name $RUN_NAME"
[ "${NUM_ENVS:-1}" != "1" ] && ARGS="$ARGS --num-envs $NUM_ENVS"
exec python3 scripts/train_ppo.py $ARGS
