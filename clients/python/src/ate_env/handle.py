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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .backend.base import BackendDriver
from .exceptions import CommandExecutionError, CommandTimeoutError
from .runtime.base import InteractiveSession, RuntimeGuestHook
from .types import DataPlaneEndpoint, ExecResult, Task


@dataclass
class SandboxHandle:
    """
    Unified handle to a claimed sandbox bound to a specific task.
    
    Provides high-level command execution, file operations, session attachment,
    and release/recycling through the fleet.

    ``data_plane`` holds this sandbox's own connection coordinates (address
    plus routing/auth headers) for harnesses that talk to the sandbox
    directly; it is None for the mock backend.
    """
    sandbox_id: str
    endpoint: str
    task: Task
    run_id: str
    backend: BackendDriver = field(repr=False)
    runtime: RuntimeGuestHook = field(repr=False)
    data_plane: Optional[DataPlaneEndpoint] = None
    _session: Optional[InteractiveSession] = field(default=None, repr=False)

    @property
    def host(self) -> str:
        """Extract host or IP from endpoint."""
        clean = self.endpoint.replace("http://", "").replace("https://", "")
        return clean.split(":")[0]

    @property
    def ip_address(self) -> str:
        """Alias for host / IP address."""
        return self.host

    @property
    def port(self) -> int:
        """Extract port from endpoint (defaults to 80 or 8080)."""
        clean = self.endpoint.replace("http://", "").replace("https://", "")
        if ":" in clean:
            try:
                return int(clean.split(":")[1].split("/")[0])
            except ValueError:
                pass
        return 8080

    def exec(self, command: str | List[str], cwd: str = "",
             env: Optional[Dict[str, str]] = None, timeout_s: float = 120.0,
             check: bool = True) -> ExecResult:
        """
        Execute command inside the sandbox.
        
        If check=True, raises CommandExecutionError on non-zero exit code
        (CommandTimeoutError if the command hit its deadline).
        Infrastructure failures always raise InfrastructureError, regardless
        of ``check``.
        Returns ExecResult.
        """
        if self._session is not None:
            stdout = self._session.run(command, timeout_s=timeout_s)
            return ExecResult(exit_code=0, stdout=stdout, stderr="")

        res = self.runtime.exec(command, cwd=cwd, env=env, timeout_s=timeout_s)
        if check and res.timed_out:
            raise CommandTimeoutError(
                f"Command '{command}' timed out after {timeout_s:g}s:\n"
                f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}",
                result=res,
            )
        if check and res.exit_code != 0:
            raise CommandExecutionError(
                f"Command '{command}' failed with exit code {res.exit_code}:\n"
                f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}",
                result=res,
            )
        return res

    async def exec_async(self, command: str | List[str], cwd: str = "",
                         env: Optional[Dict[str, str]] = None, timeout_s: float = 120.0,
                         check: bool = True) -> ExecResult:
        """Asynchronously execute command inside the sandbox."""
        res = await self.runtime.exec_async(command, cwd=cwd, env=env, timeout_s=timeout_s)
        if check and res.timed_out:
            raise CommandTimeoutError(
                f"Command '{command}' timed out after {timeout_s:g}s:\n"
                f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}",
                result=res,
            )
        if check and res.exit_code != 0:
            raise CommandExecutionError(
                f"Command '{command}' failed with exit code {res.exit_code}:\n"
                f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}",
                result=res,
            )
        return res

    async def write_file_async(self, path: str, content: bytes | str) -> None:
        """Asynchronously write file content directly into sandbox filesystem."""
        await self.runtime.write_file_async(path, content)

    async def read_file_bytes_async(self, path: str) -> bytes:
        """Asynchronously read binary file content from sandbox filesystem."""
        return await self.runtime.read_file_bytes_async(path)

    def initialize(self, init_script: Optional[str] = None, cwd: str = "",
                   timeout_s: float = 120.0) -> ExecResult:
        """Run post-boot initialization script inside the guest container."""
        return self.runtime.initialize(init_script=init_script, cwd=cwd, timeout_s=timeout_s)

    def open_session(self) -> InteractiveSession:
        """Open and attach a persistent shell session for fast iterative commands."""
        if self._session is None:
            self._session = self.runtime.open_session()
        return self._session

    def close_session(self) -> None:
        """Close persistent shell session if open."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def release(self) -> None:
        """Release this sandbox back to the backend or terminate it."""
        self.close_session()
        try:
            self.backend.release(self.sandbox_id, recycle=False)
        finally:
            self.runtime.close()

    async def release_async(self) -> None:
        """Asynchronously release this sandbox back to the backend or terminate it."""
        self.close_session()
        try:
            if hasattr(self.backend, "release_async"):
                await self.backend.release_async(self.sandbox_id, recycle=False)
            else:
                self.backend.release(self.sandbox_id, recycle=False)
        finally:
            if hasattr(self.runtime, "close_async"):
                await self.runtime.close_async()
            else:
                self.runtime.close()

    def recycle(self) -> None:
        """Return sandbox to warm pool after cleaning working state."""
        self.close_session()
        try:
            self.backend.release(self.sandbox_id, recycle=True)
        finally:
            self.runtime.close()

    async def recycle_async(self) -> None:
        """Asynchronously return sandbox to warm pool after cleaning working state."""
        self.close_session()
        try:
            if hasattr(self.backend, "release_async"):
                await self.backend.release_async(self.sandbox_id, recycle=True)
            else:
                self.backend.release(self.sandbox_id, recycle=True)
        finally:
            if hasattr(self.runtime, "close_async"):
                await self.runtime.close_async()
            else:
                self.runtime.close()

    def __enter__(self) -> "SandboxHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    async def __aenter__(self) -> "SandboxHandle":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release_async()
