r"""Warmup-stable-decay (WSD) learning rate schedule.

transformers gained a native ``warmup_stable_decay`` scheduler in 4.45, but this
repository is pinned to 4.37.2 (the trainers subclass Trainer internals that
changed after it, and every script passes ``--evaluation_strategy``, removed in
4.46). ``SchedulerType`` is an enum validated in ``TrainingArguments.__post_init__``,
so ``lr_scheduler_type: warmup_stable_decay`` cannot simply be requested here.

``lr_scheduler_kwargs`` *does* exist in 4.37.2 and is passed through untouched, so
it is used as the configuration channel. Enable WSD from a config with:

    warmup_ratio: 0.02          # or warmup_steps: 200
    lr_scheduler_kwargs:
      schedule: wsd
      decay_ratio: 0.2          # last 20%% of training decays
      decay_type: 1-sqrt        # linear | cosine | 1-sqrt
      min_lr_ratio: 0.0         # final LR as a fraction of peak

``lr_scheduler_type`` is ignored when ``schedule: wsd`` is set.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from llmtuner.extras.logging import get_logger


logger = get_logger(__name__)

WSD_ALIASES = {"wsd", "warmup_stable_decay", "warmup-stable-decay", "trapezoid"}
DECAY_TYPES = ("linear", "cosine", "1-sqrt")


def _decay_factor(progress: float, decay_type: str) -> float:
    r"""Fraction of the peak LR remaining, going 1.0 -> 0.0 as progress goes 0 -> 1."""
    if decay_type == "linear":
        return 1.0 - progress
    if decay_type == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if decay_type == "1-sqrt":  # the MiniCPM WSD decay shape
        return 1.0 - math.sqrt(progress)
    raise ValueError("Unknown decay_type {!r}, expected one of {}.".format(decay_type, DECAY_TYPES))


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
    decay_type: str = "cosine",
    last_epoch: int = -1,
) -> LambdaLR:
    r"""Linear warmup, then a constant plateau, then a decay to ``min_lr_ratio``."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0

        if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * _decay_factor(progress, decay_type)

        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WSDSchedulerMixin:
    r"""Adds WSD support to a Trainer, driven by ``args.lr_scheduler_kwargs``.

    Mix in *before* the Trainer base class. Falls back to the stock scheduler
    whenever ``schedule`` is absent from ``lr_scheduler_kwargs``, so existing
    configs are unaffected.
    """

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[Optimizer] = None):
        kwargs = dict(getattr(self.args, "lr_scheduler_kwargs", None) or {})
        schedule = str(kwargs.pop("schedule", "")).lower()

        if schedule not in WSD_ALIASES:
            if schedule:
                raise ValueError(
                    "Unknown `schedule` {!r} in lr_scheduler_kwargs, expected one of {}.".format(
                        schedule, sorted(WSD_ALIASES)
                    )
                )
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is not None:
            return self.lr_scheduler

        decay_type = str(kwargs.pop("decay_type", "cosine")).lower()
        if decay_type not in DECAY_TYPES:
            raise ValueError("Unknown decay_type {!r}, expected one of {}.".format(decay_type, DECAY_TYPES))

        min_lr_ratio = float(kwargs.pop("min_lr_ratio", 0.0))
        if not 0.0 <= min_lr_ratio < 1.0:
            raise ValueError("`min_lr_ratio` must be in [0, 1), got {}.".format(min_lr_ratio))

        num_decay_steps = kwargs.pop("num_decay_steps", None)
        decay_ratio = kwargs.pop("decay_ratio", None)
        if (num_decay_steps is None) == (decay_ratio is None):
            raise ValueError("Set exactly one of `decay_ratio` or `num_decay_steps` in lr_scheduler_kwargs.")
        if num_decay_steps is None:
            decay_ratio = float(decay_ratio)
            if not 0.0 < decay_ratio <= 1.0:
                raise ValueError("`decay_ratio` must be in (0, 1], got {}.".format(decay_ratio))
            num_decay_steps = round(decay_ratio * num_training_steps)
        num_decay_steps = int(num_decay_steps)

        if kwargs:
            raise ValueError("Unused lr_scheduler_kwargs for the wsd schedule: {}.".format(sorted(kwargs)))

        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        if num_warmup_steps + num_decay_steps > num_training_steps:
            raise ValueError(
                "warmup ({}) + decay ({}) exceeds the {} training steps; lower `warmup_ratio`/`warmup_steps` "
                "or `decay_ratio`.".format(num_warmup_steps, num_decay_steps, num_training_steps)
            )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

        logger.info(
            "Using WSD learning rate schedule over {} steps: {} warmup, {} stable, {} {} decay "
            "to {:g} x peak LR.".format(
                num_training_steps, num_warmup_steps, num_stable_steps, num_decay_steps, decay_type, min_lr_ratio
            )
        )

        self.lr_scheduler = get_wsd_schedule(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable_steps,
            num_decay_steps=num_decay_steps,
            min_lr_ratio=min_lr_ratio,
            decay_type=decay_type,
        )
        self._created_lr_scheduler = True
        return self.lr_scheduler
