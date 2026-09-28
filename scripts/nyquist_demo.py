"""Generates a visual for the Nyquist segment of the video (script section 8).

Plots the hoop's 4 s horizontal cycle (0.25 Hz) and the 0.25 s samples the bot takes (4 Hz).
Output: assets/nyquist.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PERIOD = 4.0
SAMPLE_INTERVAL = 0.25

t = np.linspace(0, 8, 1000)
ts = np.arange(0, 8 + SAMPLE_INTERVAL, SAMPLE_INTERVAL)
signal = lambda x: np.sin(2 * np.pi * x / PERIOD)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, signal(t), label="Hoop x position (0.25 Hz)", linewidth=2)
ax.stem(ts, signal(ts), linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Samples (4 Hz)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Horizontal position (normalised)")
ax.set_title("Sampling 16x faster than the hoop moves")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("assets/nyquist.png", dpi=200)
print("Wrote assets/nyquist.png")
