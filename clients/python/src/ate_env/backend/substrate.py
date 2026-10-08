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

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Mapping, Optional

from ..client import DEFAULT_ATESPACE, Client
from ..errors import EnvError, NotFoundError, PreflightError, SandboxStartError
from ..types import DataPlaneEndpoint, EnvironmentSpec, RawSandboxInstance
from .base import BackendDriver

logger = logging.getLogger("ate_env.backend.substrate")

TARGET_ACTOR_HEADER = "ate-target-actor"


def validate_grpc_target(grpc_target: str) -> None:
    """Raise ValueError unless grpc_target only uses {actor_id}/{atespace}."""
    try:
        grpc_target.format(actor_id="a", atespace="s")
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(
            f"grpc endpoint {grpc_target!r} may only use the {{actor_id}} and "
            f"{{atespace}} placeholders: {e}"
        ) from None


class SubstrateBackendDriver(BackendDriver):
    """Agent Substrate Backend Driver.

    Manages environment lifecycle directly against the ate-env-api gRPC service
    via ate_env.Client: creating environments, suspending for pre-warmed pools,
    and deleting on release.
    """

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        router_url: Optional[str] = None,
        atespace: str = DEFAULT_ATESPACE,
        worker_family: str = "c2",
        auth_token: Optional[str] = None,
        grpc_target: Optional[str] = None,
        *,
        client: Optional[Client] = None,
        image_templates: Optional[Mapping[str, str]] = None,
        default_template: str = "default-template",
    ):
        target = (
            api_endpoint
            or os.environ.get("SUBSTRATE_API_ENDPOINT")
            or os.environ.get("ATE_ENV_API_TARGET")
            or os.environ.get("ATE_API_URL")
            or "localhost:7777"
        )
        self.api_endpoint = target
        self.router_url = (
            router_url
            or os.environ.get("SUBSTRATE_ROUTER_URL")
            or os.environ.get("ATENET_ROUTER_URL")
            or "http://127.0.0.1:8080"
        )
        self.atespace = atespace or DEFAULT_ATESPACE
        self.worker_family = worker_family
        self.auth_token = auth_token
        self.image_templates = dict(image_templates or {})
        self.default_template = default_template

        if grpc_target:
            validate_grpc_target(grpc_target)
        self.grpc_target = grpc_target

        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = None
            self._owns_client = True

        self._lock = threading.Lock()
        self._owned_actors: Dict[str, RawSandboxInstance] = {}
        self._warm_paused_pool: Dict[str, List[str]] = {}
        self._templates: Dict[str, EnvironmentSpec] = {}

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(self.api_endpoint)
        return self._client

    def _sync(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def close_async(self) -> None:
        if self._owns_client and self._client:
            await self._client.close()
            self._client = None

    def close(self) -> None:
        if self._owns_client and self._client:
            self._sync(self.close_async())

    async def preflight_async(self) -> None:
        """Validate ate-env-api connectivity."""
        logger.info(
            "Running Substrate preflight checks (Atespace: %s, Endpoint: %s)...",
            self.atespace,
            self.api_endpoint,
        )
        # Probe ate-env-api by getting a non-existent canary environment
        try:
            await self.client.get(f"__preflight_probe_{uuid.uuid4().hex[:6]}", atespace=self.atespace)
        except NotFoundError:
            # NotFoundError means the server is reachable and responded properly over gRPC
            logger.debug("Preflight probe successfully reached ate-env-api")
        except Exception as e:
            logger.warning("Preflight probe notice (%s): %s", self.api_endpoint, e)

    def preflight(self) -> None:
        self._sync(self.preflight_async())

    def ensure_template(self, env: EnvironmentSpec | str) -> str:
        """Resolve ActorTemplate name from template ID or task image."""
        if isinstance(env, str):
            image = env
            tmpl = self.image_templates.get(image, self.default_template)
            return tmpl

        image = env.image or ""
        mapped = self.image_templates.get(image)
        if mapped:
            template_id = mapped
        elif hasattr(env, "template_key"):
            template_id = env.template_key()
        elif hasattr(env, "template_id") and env.template_id:
            template_id = env.template_id
        else:
            template_id = self.default_template

        with self._lock:
            self._templates[template_id] = env
        return template_id

    def _make_instance(self, actor_id: str, template_id: str, run_id: str) -> RawSandboxInstance:
        planes = {
            "ate_env": DataPlaneEndpoint(
                "grpc",
                self.api_endpoint,
                {"x-env-id": actor_id, "x-env-atespace": self.atespace},
            ),
            "grpc": DataPlaneEndpoint(
                "grpc",
                self.api_endpoint,
                {"x-env-id": actor_id, "x-env-atespace": self.atespace},
            ),
        }
        return RawSandboxInstance(
            instance_id=actor_id,
            endpoint=self.api_endpoint,
            template_id=template_id,
            run_id=run_id,
            status="RUNNING",
            metadata={
                "created_at": time.time(),
                "template": template_id,
                "atespace": self.atespace,
            },
            data_planes=planes,
        )

    async def acquire_async(
        self, template_id: str, run_id: str, timeout_s: float = 180.0
    ) -> RawSandboxInstance:
        """Acquire a live sandbox instance: claims from warm suspended pool or creates via Client.create()."""
        actor_id: Optional[str] = None
        with self._lock:
            warm_list = self._warm_paused_pool.get(template_id)
            if warm_list:
                actor_id = warm_list.pop()

        if actor_id:
            logger.info("Claimed warm paused actor %s for template %s", actor_id, template_id)
        else:
            actor_id = f"sb-{uuid.uuid4().hex[:10]}"
            logger.info(
                "Creating environment %s (template=%s, atespace=%s)...",
                actor_id,
                template_id,
                self.atespace,
            )
            try:
                await self.client.create(
                    actor_id,
                    template_name=template_id,
                    atespace=self.atespace,
                )
            except Exception as e:
                raise SandboxStartError(f"Failed to create environment {actor_id}: {e}") from e

        instance = self._make_instance(actor_id, template_id, run_id)
        with self._lock:
            self._owned_actors[actor_id] = instance
        return instance

    def acquire(
        self, template_id: str, run_id: str, timeout_s: float = 180.0
    ) -> RawSandboxInstance:
        return self._sync(self.acquire_async(template_id, run_id, timeout_s=timeout_s))

    async def warm_pool_async(self, template_id: str, replicas: int, wait: bool = True) -> None:
        """Pre-warm a pool of actors by creating them and calling Client.suspend()."""
        logger.info("Sizing warm pool for template %s to %d paused actors...", template_id, replicas)
        with self._lock:
            current = list(self._warm_paused_pool.get(template_id, []))
        needed = replicas - len(current)
        if needed <= 0:
            return

        for _ in range(needed):
            actor_id = f"warm-{uuid.uuid4().hex[:10]}"
            try:
                await self.client.create(
                    actor_id,
                    template_name=template_id,
                    atespace=self.atespace,
                )
                await self.client.suspend(actor_id, atespace=self.atespace)
            except Exception as e:
                logger.warning("Failed to provision warm actor %s: %s", actor_id, e)
                continue

            with self._lock:
                self._warm_paused_pool.setdefault(template_id, []).append(actor_id)
                self._owned_actors[actor_id] = self._make_instance(
                    actor_id, template_id, run_id="warm-pool"
                )

    def warm_pool(self, template_id: str, replicas: int, wait: bool = True) -> None:
        self._sync(self.warm_pool_async(template_id, replicas, wait=wait))

    async def unwarm_pool_async(self, template_id: str) -> None:
        """Drain and delete paused actors in the warm pool."""
        logger.info("Unwarming pool for template %s...", template_id)
        with self._lock:
            to_delete = self._warm_paused_pool.pop(template_id, [])

        for actor_id in to_delete:
            try:
                await self.client.delete(actor_id, atespace=self.atespace)
            except Exception as e:
                logger.debug("Failed deleting warm actor %s: %s", actor_id, e)
            with self._lock:
                self._owned_actors.pop(actor_id, None)

    def unwarm_pool(self, template_id: str) -> None:
        self._sync(self.unwarm_pool_async(template_id))

    async def release_async(self, instance_id: str, recycle: bool = False) -> None:
        """Release actor: deletes environment, or suspends if recycling is requested."""
        with self._lock:
            instance = self._owned_actors.pop(instance_id, None)

        if recycle and instance:
            try:
                await self.client.suspend(instance_id, atespace=self.atespace)
                with self._lock:
                    self._warm_paused_pool.setdefault(instance.template_id, []).append(instance_id)
                    self._owned_actors[instance_id] = instance
                logger.info("Recycled actor %s to paused warm pool", instance_id)
                return
            except Exception as e:
                logger.warning("Failed to suspend actor %s for recycle, deleting: %s", instance_id, e)

        try:
            await self.client.delete(instance_id, atespace=self.atespace)
            logger.info("Deleted environment %s", instance_id)
        except Exception as e:
            logger.debug("Failed to delete environment %s: %s", instance_id, e)

    def release(self, instance_id: str, recycle: bool = False) -> None:
        self._sync(self.release_async(instance_id, recycle=recycle))

    async def reap_async(self, run_id: str) -> int:
        """Force delete all environments tagged with this run_id."""
        with self._lock:
            to_reap = [aid for aid, inst in self._owned_actors.items() if inst.run_id == run_id]

        reaped = 0
        for aid in to_reap:
            try:
                await self.client.delete(aid, atespace=self.atespace)
                reaped += 1
            except Exception as e:
                logger.debug("Failed deleting actor %s during reap: %s", aid, e)
            with self._lock:
                self._owned_actors.pop(aid, None)

        logger.info("Reaped %d Substrate environments for run %s", reaped, run_id)
        return reaped

    def reap(self, run_id: str) -> int:
        return self._sync(self.reap_async(run_id))
