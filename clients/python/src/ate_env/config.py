# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field, fields
import re
from typing import Any, Dict, List, Literal, Optional

from .backend.substrate import validate_grpc_target
from .errors import InvalidArgumentError

BackendType = Literal["substrate", "mock"]
StrategyType = Literal["none", "naive", "sliding", "pipelined"]

# Atespaces (Substrate) are DNS-1123 labels.
_DNS1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")

ALLOWED_BACKENDS = {"substrate", "mock"}
ALLOWED_STRATEGIES = {"none", "naive", "sliding", "pipelined"}
ALLOWED_DATA_PLANES = {"ate_env", "grpc", "router"}


class ValidationError(InvalidArgumentError, ValueError):
    """Configuration validation error."""


class SecretStr:
    """Wraps sensitive strings to prevent accidental leakage in repr or str."""

    def __init__(self, secret_value: str):
        if not isinstance(secret_value, str):
            raise TypeError("SecretStr value must be a str")
        self._secret_value = secret_value

    def get_secret_value(self) -> str:
        return self._secret_value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SecretStr):
            return self._secret_value == other._secret_value
        return False

    def __hash__(self) -> int:
        return hash(self._secret_value)


@dataclass(frozen=True)
class FleetConfig:
    """Configuration for SandboxFleet and associated BackendDriver.

    Unknown fields are rejected so misspelled or unsupported options
    fail loudly instead of being silently ignored.
    """

    backend: BackendType = "substrate"
    endpoint: str = "http://localhost:7777"
    router_url: Optional[str] = None
    grpc_endpoint: Optional[str] = None
    data_plane: Literal["ate_env", "grpc", "router"] = "ate_env"
    tenancy: str = "ate-env"
    strategy: StrategyType = "sliding"

    # Sizing & Rollout parameters
    batch_size: int = 8
    num_generations: int = 8
    max_concurrent: int = 64
    max_warmpool_replicas: int = 4
    window_size: Optional[int] = None

    # Hardware & placement
    worker_family: Optional[str] = "c2"
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    # Timeouts & Breakers
    acquire_timeout_s: float = 180.0
    ready_timeout_s: float = 180.0
    step_timeout_s: float = 120.0
    auth_token: Optional[SecretStr] = None

    def __init__(self, **kwargs: Any):
        valid_field_names = {f.name for f in fields(self)}
        unknown = set(kwargs.keys()) - valid_field_names
        if unknown:
            raise ValidationError(f"Unknown fields for FleetConfig: {sorted(unknown)}")

        # Helper to get or default
        def _get(name: str, default: Any = None):
            return kwargs[name] if name in kwargs else default

        backend = _get("backend", "substrate")
        if backend not in ALLOWED_BACKENDS:
            raise ValidationError(f"Invalid backend {backend!r}, must be one of {ALLOWED_BACKENDS}")

        endpoint = _get("endpoint", "http://localhost:7777")
        router_url = _get("router_url", None)
        grpc_endpoint = _get("grpc_endpoint", None)
        if grpc_endpoint:
            try:
                validate_grpc_target(grpc_endpoint)
            except ValueError as e:
                raise ValidationError(str(e)) from e

        data_plane = _get("data_plane", "ate_env")
        if data_plane not in ALLOWED_DATA_PLANES:
            raise ValidationError(f"Invalid data_plane {data_plane!r}, must be one of {ALLOWED_DATA_PLANES}")
        if data_plane == "grpc" and not grpc_endpoint:
            raise ValidationError("data_plane='grpc' requires grpc_endpoint")

        tenancy = _get("tenancy", "ate-env")
        if not isinstance(tenancy, str) or not _DNS1123_LABEL.match(tenancy):
            raise ValidationError(f"Invalid tenancy {tenancy!r}: must match DNS-1123 label regex")

        strategy = _get("strategy", "sliding")
        if strategy not in ALLOWED_STRATEGIES:
            raise ValidationError(f"Invalid strategy {strategy!r}, must be one of {ALLOWED_STRATEGIES}")

        batch_size = _get("batch_size", 8)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValidationError(f"batch_size must be >= 1, got {batch_size}")

        num_generations = _get("num_generations", 8)
        if not isinstance(num_generations, int) or num_generations < 1:
            raise ValidationError(f"num_generations must be >= 1, got {num_generations}")

        max_concurrent = _get("max_concurrent", 64)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValidationError(f"max_concurrent must be >= 1, got {max_concurrent}")

        max_warmpool_replicas = _get("max_warmpool_replicas", 4)
        if not isinstance(max_warmpool_replicas, int) or max_warmpool_replicas < 0:
            raise ValidationError(f"max_warmpool_replicas must be >= 0, got {max_warmpool_replicas}")

        window_size = _get("window_size", None)
        if window_size is not None and (not isinstance(window_size, int) or window_size < 1):
            raise ValidationError(f"window_size must be >= 1, got {window_size}")

        worker_family = _get("worker_family", "c2")
        node_selector = _get("node_selector", {})
        tolerations = _get("tolerations", [])
        labels = _get("labels", {})

        acquire_timeout_s = _get("acquire_timeout_s", 180.0)
        if acquire_timeout_s <= 0:
            raise ValidationError(f"acquire_timeout_s must be > 0, got {acquire_timeout_s}")

        ready_timeout_s = _get("ready_timeout_s", 180.0)
        if ready_timeout_s <= 0:
            raise ValidationError(f"ready_timeout_s must be > 0, got {ready_timeout_s}")

        step_timeout_s = _get("step_timeout_s", 120.0)
        if step_timeout_s <= 0:
            raise ValidationError(f"step_timeout_s must be > 0, got {step_timeout_s}")

        auth_token_raw = _get("auth_token", None)
        auth_token: Optional[SecretStr] = None
        if auth_token_raw is not None:
            if isinstance(auth_token_raw, SecretStr):
                auth_token = auth_token_raw
            else:
                auth_token = SecretStr(str(auth_token_raw))

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "router_url", router_url)
        object.__setattr__(self, "grpc_endpoint", grpc_endpoint)
        object.__setattr__(self, "data_plane", data_plane)
        object.__setattr__(self, "tenancy", tenancy)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "num_generations", num_generations)
        object.__setattr__(self, "max_concurrent", max_concurrent)
        object.__setattr__(self, "max_warmpool_replicas", max_warmpool_replicas)
        object.__setattr__(self, "window_size", window_size)
        object.__setattr__(self, "worker_family", worker_family)
        object.__setattr__(self, "node_selector", dict(node_selector))
        object.__setattr__(self, "tolerations", list(tolerations))
        object.__setattr__(self, "labels", dict(labels))
        object.__setattr__(self, "acquire_timeout_s", float(acquire_timeout_s))
        object.__setattr__(self, "ready_timeout_s", float(ready_timeout_s))
        object.__setattr__(self, "step_timeout_s", float(step_timeout_s))
        object.__setattr__(self, "auth_token", auth_token)

    def model_dump(self) -> Dict[str, Any]:
        """Serialize configuration to dict, keeping auth_token as SecretStr for safe round-tripping."""
        res: Dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            res[f.name] = val
        return res

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "FleetConfig":
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FleetConfig":
        return cls.model_validate(data)
