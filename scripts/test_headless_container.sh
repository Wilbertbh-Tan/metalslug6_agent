#!/bin/bash
# Test that the headless container works: Xvfb, Python, env import.
# Run with: docker run --rm metalslug-rl /app/scripts/test_headless_container.sh
# For a full training test, use the default CMD and mount your game content (see CONTAINER.md).
set -e
export DISPLAY="${DISPLAY:-:99}"
export CAPTURE_LEFT="${CAPTURE_LEFT:-0}"
export CAPTURE_TOP="${CAPTURE_TOP:-0}"

echo "Starting Xvfb..."
Xvfb "$DISPLAY" -screen 0 1280x720x24 &
sleep 2

# So pyautogui/Xlib can connect (they expect ~/.Xauthority)
touch "$HOME/.Xauthority"
xauth add "$DISPLAY" . $(mcookie 2>/dev/null || echo "0")

echo "Checking Python and env import..."
cd /app
python3 -c "
from src.env.mslug_env import CaptureRegion, MetalSlugEnv
r = CaptureRegion(left=0, top=0)
env = MetalSlugEnv(region=r, in_game_checks=[], continue_checks=[])
print('Env created (observation_space=%s)' % (env.observation_space.shape,))
env.close()
print('OK: headless container env is ready.')
"

echo "Headless container test passed."
