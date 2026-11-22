# selection.py
# Variants shown in the figures. We keep the data of every run in results/, but in the
# plots we only show these ones (plain R1 and the ablations are left out).

import re

PLOT_VARIANTS = ("r0", "r0_sched", "r0_floor", "r1_floor", "r2", "r3")


def variant_of_run(run: str) -> str:
    """Run name without the seed suffix: r1_floor_s3 -> r1_floor."""
    return re.sub(r"_s\d+$", "", run)


def keep_run(run: str) -> bool:
    """True if the run belongs to one of the variants we plot."""
    return variant_of_run(run) in PLOT_VARIANTS
