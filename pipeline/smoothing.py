"""One-Euro filter & temporal smoothing for noisy landmark sequences and tracking dropouts."""

import math
import numpy as np
from typing import List, Dict, Any, Optional

class OneEuroFilter:
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq if self.freq > 0 else 1.0 / 30.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = np.copy(x)
            self.dx_prev = np.zeros_like(x)
            return x

        # Estimate derivative
        dx = (x - self.x_prev) * self.freq
        edx = self.alpha(self.dcutoff) * dx + (1.0 - self.alpha(self.dcutoff)) * self.dx_prev
        self.dx_prev = edx

        # Calculate cutoff frequency
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        
        # Calculate dynamic alpha per component
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) * self.freq)
        x_filtered = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_filtered
        return x_filtered


def interpolate_missing_landmarks(frames_data: List[Dict[str, Any]], key: str) -> None:
    """Linearly interpolate missing landmark arrays across frames in-place."""
    n = len(frames_data)
    if n == 0:
        return

    # Find valid frames
    valid_indices = [i for i in range(n) if frames_data[i].get(key) is not None]
    if not valid_indices:
        return

    # Extrapolate beginning if needed
    first_valid = valid_indices[0]
    for i in range(first_valid):
        frames_data[i][key] = frames_data[first_valid][key]

    # Extrapolate end if needed
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, n):
        frames_data[i][key] = frames_data[last_valid][key]

    # Interpolate intermediate gaps
    for idx in range(len(valid_indices) - 1):
        start_idx = valid_indices[idx]
        end_idx = valid_indices[idx + 1]
        gap = end_idx - start_idx
        if gap > 1:
            start_lms = frames_data[start_idx][key]
            end_lms = frames_data[end_idx][key]
            for step in range(1, gap):
                alpha = step / float(gap)
                curr_idx = start_idx + step
                interpolated = []
                for p0, p1 in zip(start_lms, end_lms):
                    interp_p = {
                        "x": (1.0 - alpha) * p0["x"] + alpha * p1["x"],
                        "y": (1.0 - alpha) * p0["y"] + alpha * p1["y"],
                        "z": (1.0 - alpha) * p0["z"] + alpha * p1["z"]
                    }
                    if "visibility" in p0 and "visibility" in p1:
                        interp_p["visibility"] = (1.0 - alpha) * p0["visibility"] + alpha * p1["visibility"]
                    interpolated.append(interp_p)
                frames_data[curr_idx][key] = interpolated
