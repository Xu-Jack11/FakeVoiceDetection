"""Compatibility shim for legacy imports.

Historically, this module lived under ``Resnet.cache_mel_features``. The shared
implementation has moved to the project root (``cache_mel_features.py``). This
file simply re-exports the public API to avoid breaking existing imports.
"""

from cache_mel_features import *  # noqa: F401,F403
