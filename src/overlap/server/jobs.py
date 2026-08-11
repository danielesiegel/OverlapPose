"""Background job engine for the web UI.

Jobs wrap the same core entry points the CLI calls synchronously
(:func:`overlap.ingest.index_paths`, :func:`overlap.match.compare_manifest_file`,
:func:`overlap.match.self_dedupe`) and forward their progress callbacks into
an event list that the SSE endpoint replays - the UI sees exactly the event
stream `--json` mode prints.

Cancellation is cooperative: the progress callback raises when a cancel is
requested, which aborts between files; indexing is resumable by design, so a
cancelled job loses nothing.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from overlap.paths import reports_dir


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    job_id: str
    kind: str
    params: dict[str, Any]
    created: float = field(default_factory=time.time)
    status: str = "running"  # running | done | error | cancelled
    error: str | None = None
    result: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "params": self.params,
            "created": self.created,
            "status": self.status,
            "error": self.error,
            "result": self.result,
            "n_events": len(self.events),
        }


class JobManager:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def _spawn(self, kind: str, params: dict[str, Any], target: Any) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, params=params)
        with self._lock:
            self.jobs[job.job_id] = job

        def runner() -> None:
            try:
                job.result = target(job)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001 - report, don't kill the server
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            job.events.append({"event": "job_end", "status": job.status})

        threading.Thread(target=runner, daemon=True, name=f"job-{job.job_id}").start()
        return job

    def _progress(self, job: Job) -> Any:
        def cb(event: dict[str, Any]) -> None:
            job.events.append(event)
            if job.cancel_requested:
                raise JobCancelled()

        return cb

    # -- job kinds -------------------------------------------------------------

    def start_index(
        self,
        paths: list[str],
        *,
        sample_fps: float,
        workers: int,
        crop_ladder: str = "",
        crop_edges: str = "",
        reindex: bool = False,
    ) -> Job:
        from overlap.ingest import index_paths

        def target(job: Job) -> dict[str, Any]:
            stats = index_paths(
                [Path(p) for p in paths],
                self.index_dir,
                sample_fps=sample_fps,
                crop_ladder=crop_ladder,
                crop_edges=crop_edges,
                workers=workers,
                reindex=reindex,
                progress=self._progress(job),
            )
            return {
                "indexed": stats.indexed,
                "skipped": stats.skipped,
                "errors": stats.errors,
                "frames": stats.frames,
            }

        return self._spawn("index", {"paths": paths, "reindex": reindex}, target)

    def start_compare(
        self,
        manifest_path: Path,
        *,
        min_run_s: float,
        nprobe: int,
        max_manifest_bytes: int,
        label: str | None = None,
    ) -> Job:
        from overlap.match import compare_manifest_file

        def target(job: Job) -> dict[str, Any]:
            report = compare_manifest_file(
                manifest_path,
                self.index_dir,
                min_run_s=min_run_s,
                nprobe=nprobe,
                max_manifest_bytes=max_manifest_bytes,
                progress=self._progress(job),
            )
            report_id = save_report(report)
            return {"report_id": report_id, "summary": report["summary"]}

        return self._spawn("compare", {"manifest": manifest_path.name, "label": label}, target)

    def start_self_dedupe(self, *, min_run_s: float) -> Job:
        from overlap.match import self_dedupe

        def target(job: Job) -> dict[str, Any]:
            report = self_dedupe(self.index_dir, min_run_s=min_run_s, progress=self._progress(job))
            report_id = save_report(report)
            return {"report_id": report_id, "summary": report["summary"]}

        return self._spawn("self-dedupe", {}, target)

    # -- queries ---------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        job.cancel_requested = True
        return True


# -- report persistence (shared with the report routes) -------------------------


def save_report(report: dict[str, Any]) -> str:
    reports_dir().mkdir(parents=True, exist_ok=True)
    report_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    (reports_dir() / f"{report_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_id


def load_report(report_id: str) -> dict[str, Any] | None:
    if not report_id.replace("-", "").isalnum():  # ids are path components
        return None
    path = reports_dir() / f"{report_id}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def list_reports() -> list[dict[str, Any]]:
    if not reports_dir().is_dir():
        return []
    out = []
    for path in sorted(reports_dir().glob("*.json"), reverse=True):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            out.append(
                {
                    "report_id": path.stem,
                    "generated": doc.get("generated"),
                    "mode": doc.get("mode", "compare"),
                    "summary": doc.get("summary", {}),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return out


def delete_report(report_id: str) -> bool:
    if not report_id.replace("-", "").isalnum():
        return False
    path = reports_dir() / f"{report_id}.json"
    if path.is_file():
        path.unlink()
        return True
    return False
