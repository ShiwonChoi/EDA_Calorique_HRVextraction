import matplotlib.pyplot as plt


def plot_sound_touch(touch, trial, category, sound_name, start_s, end_s, show=True):
    if not show:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(touch["monotonic_s"], touch["vas"], "-o", ms=2, lw=1)
    ax.axvline(start_s, color="k", ls="--", lw=1)
    ax.axvline(end_s, color="r", ls="--", lw=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("monotonic_s (zero-referenced, s)")
    ax.set_ylabel("VAS position (0-100)")
    ax.set_title(f"Trial {trial} - {category} / {sound_name}")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
