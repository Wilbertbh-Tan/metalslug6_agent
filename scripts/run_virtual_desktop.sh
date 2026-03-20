#!/bin/bash
# Run training in a virtual desktop (Xvfb + Openbox + optional VNC).
# Starts RetroArch in that desktop and auto-detects capture region when possible.

set -euo pipefail

export DISPLAY="${DISPLAY:-:99}"
export MSLUG_HEADLESS=1
SCREEN_GEOMETRY="${SCREEN_GEOMETRY:-1280x720x24}"
VNC_PORT="${VNC_PORT:-5900}"
START_VNC="${START_VNC:-1}"
RETROARCH_LOG="${RETROARCH_LOG:-/tmp/retroarch.log}"
ALLOW_NO_WINDOW="${ALLOW_NO_WINDOW:-0}"

Xvfb "$DISPLAY" -screen 0 "$SCREEN_GEOMETRY" &
sleep 2

# So pyautogui/Xlib can connect.
touch "$HOME/.Xauthority"
xauth add "$DISPLAY" . "$(mcookie 2>/dev/null || echo "0")"

# Start a lightweight desktop session.
openbox >/tmp/openbox.log 2>&1 &
sleep 1

if [ "$START_VNC" != "0" ]; then
  x11vnc \
    -display "$DISPLAY" \
    -rfbport "$VNC_PORT" \
    -nopw \
    -shared \
    -forever \
    -quiet \
    >/tmp/x11vnc.log 2>&1 &
  echo "Virtual desktop ready on DISPLAY=$DISPLAY (VNC port $VNC_PORT, no password)."

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

  echo "Starting RetroArch with core=$_CORE content=$RETROARCH_CONTENT"
  retroarch -v --config /etc/retroarch.cfg -L "$_CORE" "$RETROARCH_CONTENT" ${RETROARCH_APPEND_ARGS:-} >"$RETROARCH_LOG" 2>&1 &
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
    echo "RetroArch window detected: id=$RA_WIN_ID left=${CAPTURE_LEFT:-} top=${CAPTURE_TOP:-} width=${CAPTURE_WIDTH:-} height=${CAPTURE_HEIGHT:-}"

    # Auto-boot: send coin + start keys to get past title screen, then save state.
    # On a fresh container there is no save state, so F4 in reset() would do nothing.
    # Default key sequence for Metal Slug (FBNeo, default RetroArch bindings):
    #   Shift_R = Select (coin), Return = Start, x = A button (confirm).
    # Format: "key:delay_secs:key:delay_secs:..."
    if [ "${AUTO_BOOT_GAME:-1}" != "0" ]; then
      # Wait for the BIOS + ROM to finish loading before sending keys.
      BOOT_WAIT="${BOOT_WAIT:-15}"
      BOOT_KEYS="${BOOT_KEYS:-Shift_R:2:Return:4:x:4:x:4}"
      echo "Auto-boot: waiting ${BOOT_WAIT}s for BIOS + ROM to load..."
      sleep "$BOOT_WAIT"
      echo "Auto-boot: sending key sequence to start game ($BOOT_KEYS)..."
      xdotool windowfocus --sync "$RA_WIN_ID" 2>/dev/null || true
      sleep 1
      IFS=':' read -ra _parts <<< "$BOOT_KEYS"
      _i=0
      while [ "$_i" -lt "${#_parts[@]}" ]; do
        _key="${_parts[$_i]}"
        xdotool key --window "$RA_WIN_ID" "$_key"
        echo "  sent: $_key"
        _i=$((_i + 1))
        if [ "$_i" -lt "${#_parts[@]}" ]; then
          sleep "${_parts[$_i]}"
          _i=$((_i + 1))
        fi
      done
      # Wait for gameplay to settle, then create save state (F2) for reset().
      sleep 2
      xdotool key --window "$RA_WIN_ID" F2
      echo "Auto-boot: save state created (F2). Subsequent F4 resets will reload this state."
    fi
  else
    if kill -0 "$RA_PID" 2>/dev/null; then
      echo "Warning: RetroArch is running but no X11 window was found."
      echo "Known windows from wmctrl (if any):"
      wmctrl -lp 2>/dev/null || true
    else
      echo "Error: RetroArch exited before a window was created."
    fi
    if [ -f "$RETROARCH_LOG" ]; then
      echo "RetroArch log tail ($RETROARCH_LOG):"
      tail -n 80 "$RETROARCH_LOG" || true
    fi
    if [ "$ALLOW_NO_WINDOW" != "1" ]; then
      echo "Aborting: no RetroArch window means capture will likely be invalid."
      exit 1
    fi
    echo "ALLOW_NO_WINDOW=1 set; continuing with fallback capture origin."
  fi
fi

export CAPTURE_LEFT="${CAPTURE_LEFT:-0}"
export CAPTURE_TOP="${CAPTURE_TOP:-0}"
export CAPTURE_WIDTH="${CAPTURE_WIDTH:-640}"
export CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-480}"

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
[ -n "${SCORE_LOG_EVERY:-}" ] && ARGS="$ARGS --score-log-every $SCORE_LOG_EVERY"
[ -n "${NO_SCORE_LOG_STDOUT:-}" ] && ARGS="$ARGS --no-score-log-stdout"
[ -n "${TARGET_EPISODES:-}" ] && ARGS="$ARGS --target-episodes $TARGET_EPISODES"
[ -n "${VIDEO_EVERY_EPISODES:-}" ] && ARGS="$ARGS --video-every-episodes $VIDEO_EVERY_EPISODES"
[ -n "${VIDEO_MAX_STEPS:-}" ] && ARGS="$ARGS --video-max-steps $VIDEO_MAX_STEPS"
[ -n "${VIDEO_FPS:-}" ] && ARGS="$ARGS --video-fps $VIDEO_FPS"
[ -n "${VIDEO_DIR:-}" ] && ARGS="$ARGS --video-dir $VIDEO_DIR"
[ -n "${DEVICE:-}" ] && ARGS="$ARGS --device $DEVICE"
exec python3 scripts/train_ppo.py $ARGS
