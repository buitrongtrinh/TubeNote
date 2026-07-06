"""Shared lock for heavyweight models that must take turns on one GPU."""
from __future__ import annotations

import threading


GPU_MODEL_LOCK = threading.RLock()
