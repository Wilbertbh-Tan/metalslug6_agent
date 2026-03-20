"""Reward computation for Metal Slug 6 RL environment."""


def compute_reward(
    action,
    curr_score: int,
    prev_score: int,
    score_scale: float = 0.005,
    score_clip: float = 2.0,
    progress_scale: float = 0.05,
    time_penalty: float = -0.005,
) -> tuple[float, dict]:
    """Compute step reward from score delta, progress, and time penalty.

    action: MultiDiscrete array [movement, attack, modifier] or legacy int.
    """
    score_delta = curr_score - prev_score
    if abs(score_delta) > 50000:
        score_delta = 0
    score_reward = min(max(score_delta, 0) * score_scale, score_clip)

    # Extract movement from MultiDiscrete action (dim 0) or legacy int
    movement = int(action[0]) if hasattr(action, '__len__') else int(action)
    progress_reward = 0.0
    if movement == 2:  # right
        progress_reward = progress_scale
    elif movement == 1:  # left
        progress_reward = -progress_scale * 0.2

    total = score_reward + progress_reward + time_penalty
    info = {
        "score": curr_score,
        "score_delta": score_delta,
        "score_reward": score_reward,
        "score_clip": score_clip,
        "progress_reward": progress_reward,
        "time_penalty": time_penalty,
        "total_reward": total,
    }
    return total, info
