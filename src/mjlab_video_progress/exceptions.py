"""Exceptions: every custom exception the package raises, in one place.

Contract:

- ``ConfigError``: raised by ``config.load_config``.
- ``DiscoveryError`` (a ``LookupError``) and its subclass ``RunDirNotFound``: raised by
  ``discovery``. Catching ``DiscoveryError`` catches both.
- The modules that raise them import them from here, so ``config.ConfigError`` and
  ``discovery.DiscoveryError`` / ``discovery.RunDirNotFound`` are the same classes.
- ``RenderError``: raised by ``render.render_checkpoint`` when ``play`` produced no video.
- ``GpuError``: raised by ``gpu.free_mib`` when ``nvidia-smi`` is there but gave no reading.
- ``SinkError``: raised by a sink (``telegram.deliver_telegram``) when it could not deliver. Its
  message says why and never contains a secret.
"""

from __future__ import annotations


class ConfigError(Exception):
    """The config file is missing, malformed or invalid."""


class DiscoveryError(LookupError):
    """Discovery found nothing to watch; the message says why."""


class RunDirNotFound(DiscoveryError):
    """The run dir of a training process could not be located; the message says why."""


class RenderError(Exception):
    """A render did not produce a finished video; the message says why."""


class SinkError(Exception):
    """A sink could not deliver a video; the message says why and never contains a secret."""


class GpuError(Exception):
    """nvidia-smi is on PATH but did not report the free GPU memory; the message says why."""
