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
import builtins
import concurrent.futures
import logging
import shlex
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import grpc

from ..client import Client as AteClient
from ..env import Env as AteEnv
from ..errors import (
    CommandExecutionError,
    CommandStartError,
    CommandTimeoutError,
    EnvError,
    InfrastructureError,
    InvalidArgumentError,
    NotFoundError,
    PermissionDeniedError,
    RpcError,
    SandboxProtocolError,
    SandboxUnavailableError,
)
from ..types import ExecResult
from .base import (
    InteractiveSession,
    RuntimeGuestHook,
    command_to_argv,
)

logger = logging.getLogger("ate_env.runtime.substrate_env_client")


class SubstrateEnvClientRuntime(RuntimeGuestHook):
    """
    Substrate guest hook backed directly by the official ate_env.Env handle.

    Leverages ate_env.Client and ate_env.Env for process execution, streaming,
    and chunked file I/O over gRPC.
    """

    def __init__(
        self,
        endpoint: str = "localhost:7777",
        env_id: str = "",
        atespace: str = "ate-env",
        *,
        client: Optional[AteClient] = None,
        env: Optional[AteEnv] = None,
    ):
        self.endpoint = endpoint
        self.env_id = env_id
        self.atespace = atespace
        if env is not None:
            self._env = env
            self._client = env._client
            self._owns_client = False
        elif client is not None:
            self._client = client
            self._env = client.env(env_id, atespace=atespace)
            self._owns_client = False
        else:
            self._client = AteClient(endpoint)
            self._env = self._client.env(env_id, atespace=atespace)
            self._owns_client = True
        self._closed = False

    async def close_async(self) -> None:
        """Close client if owned."""
        if not self._closed:
            self._closed = True
            if self._owns_client and self._client:
                await self._client.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client and self._client:
                self._sync(self._client.close())

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"runtime for environment {self.env_id} is closed")

    def _sync(self, coro: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def exec_async(
        self,
        command: str | List[str],
        cwd: str = "",
        env: Optional[Dict[str, str]] = None,
        timeout_s: float = 120.0,
    ) -> ExecResult:
        """Native async execution over ate_env.Env.shell() passing guest deadline."""
        self._check_open()
        t0 = time.monotonic()
        cmd_str = command if isinstance(command, str) else shlex.join(command)

        # Client-side grace margin beyond the guest deadline
        grace_s = 5.0
        try:
            # We enforce both the guest-side timeout (SIGKILL in container)
            # and a client-side deadline with grace period.
            try:
                # Python 3.11+ has asyncio.timeout; fall back to wait_for on 3.10
                if hasattr(asyncio, "timeout"):
                    async with asyncio.timeout(timeout_s + grace_s):
                        shell_res = await self._env.shell(
                            cmd_str, cwd=cwd, env=env, timeout=timeout_s
                        )
                else:
                    shell_res = await asyncio.wait_for(
                        self._env.shell(cmd_str, cwd=cwd, env=env, timeout=timeout_s),
                        timeout=timeout_s + grace_s,
                    )
            except (builtins.TimeoutError, asyncio.TimeoutError):
                duration = time.monotonic() - t0
                return ExecResult(
                    exit_code=None,
                    stdout="",
                    stderr="",
                    duration_s=duration,
                    timed_out=True,
                )

            duration = time.monotonic() - t0
            # 137 = 128 + 9 (SIGKILL sent by guest when guest-side timeout elapsed)
            timed_out = shell_res.exit_code == 137 and duration >= timeout_s
            return ExecResult(
                exit_code=None if timed_out else shell_res.exit_code,
                stdout=shell_res.stdout,
                stderr=shell_res.stderr,
                duration_s=duration,
                timed_out=timed_out,
            )
        except InvalidArgumentError as e:
            raise SandboxProtocolError(str(e), sandbox_id=self.env_id) from e
        except NotFoundError as e:
            msg = str(e).lower()
            if "environment" in msg or "actor" in msg:
                raise SandboxUnavailableError(str(e), sandbox_id=self.env_id) from e
            raise CommandStartError(str(e), sandbox_id=self.env_id) from e
        except RpcError as e:
            status_name = f"grpc {e.code.name}" if e.code else "grpc RpcError"
            if e.code in (
                grpc.StatusCode.NOT_FOUND,
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.ABORTED,
            ):
                raise SandboxUnavailableError(
                    str(e), sandbox_id=self.env_id, status=status_name
                ) from e
            raise SandboxProtocolError(str(e), sandbox_id=self.env_id, status=status_name) from e
        except InfrastructureError:
            raise
        except Exception as e:
            raise InfrastructureError(str(e), sandbox_id=self.env_id) from e

    def exec(
        self,
        command: str | List[str],
        cwd: str = "",
        env: Optional[Dict[str, str]] = None,
        timeout_s: float = 120.0,
    ) -> ExecResult:
        """Synchronous wrapper for exec_async."""
        return self._sync(self.exec_async(command, cwd=cwd, env=env, timeout_s=timeout_s))

    async def stream_process_async(
        self,
        argv: List[str],
        cwd: str = "",
        env: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[str]:
        """Stream real-time output chunks from running process inside guest."""
        self._check_open()
        try:
            proc = await self._env.start_process(argv, cwd=cwd, env=env)
            async for chunk in proc.output(follow=True):
                if chunk.stdout is not None:
                    yield chunk.stdout.decode("utf-8", errors="replace")
                elif chunk.stderr is not None:
                    yield chunk.stderr.decode("utf-8", errors="replace")
        except (InvalidArgumentError, NotFoundError, RpcError) as e:
            raise SandboxUnavailableError(str(e), sandbox_id=self.env_id) from e
        except Exception as e:
            raise InfrastructureError(str(e), sandbox_id=self.env_id) from e

    def stream_process(
        self,
        argv: List[str],
        cwd: str = "",
        env: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Synchronously stream output chunks."""
        self._check_open()

        async def _collect():
            chunks = []
            async for item in self.stream_process_async(argv, cwd=cwd, env=env):
                chunks.append(item)
            return chunks

        yield from self._sync(_collect())

    async def write_file_async(self, path: str, content: bytes | str) -> None:
        """Write file content directly into guest filesystem via FileSystemService."""
        self._check_open()
        raw = content.encode("utf-8") if isinstance(content, str) else content
        try:
            await self._env.write_file(path, raw)
        except InvalidArgumentError as e:
            raise SandboxProtocolError(str(e), sandbox_id=self.env_id) from e
        except (NotFoundError, RpcError) as e:
            raise SandboxUnavailableError(str(e), sandbox_id=self.env_id) from e
        except Exception as e:
            raise InfrastructureError(str(e), sandbox_id=self.env_id) from e

    def write_file(self, path: str, content: bytes | str) -> None:
        self._sync(self.write_file_async(path, content))

    async def read_file_bytes_async(self, path: str) -> bytes:
        """Read binary file content from guest filesystem via FileSystemService."""
        self._check_open()
        try:
            return await self._env.read_file_bytes(path)
        except NotFoundError:
            raise FileNotFoundError(f"File not found in sandbox: {path}")
        except InvalidArgumentError as e:
            raise SandboxProtocolError(str(e), sandbox_id=self.env_id) from e
        except RpcError as e:
            raise SandboxUnavailableError(str(e), sandbox_id=self.env_id) from e
        except Exception as e:
            raise InfrastructureError(str(e), sandbox_id=self.env_id) from e

    def read_file_bytes(self, path: str) -> bytes:
        return self._sync(self.read_file_bytes_async(path))

    def open_session(self) -> InteractiveSession:
        return _InteractiveSession(self)


class _InteractiveSession(InteractiveSession):
    def __init__(self, runtime: SubstrateEnvClientRuntime):
        self._runtime = runtime

    def run(self, command: str | List[str], timeout_s: Optional[float] = None) -> str:
        res = self._runtime.exec(command, timeout_s=timeout_s or 120.0)
        if res.timed_out:
            raise CommandTimeoutError(f"Command timed out: {command}", result=res)
        if res.exit_code != 0:
            raise CommandExecutionError(
                f"Command failed with exit code {res.exit_code}:\n{res.stderr}", result=res
            )
        return res.stdout

    def close(self) -> None:
        pass
