"""The catalog: SQLite source of truth for an index directory.

Layout of an index directory (``*.ovl``)::

    corpus.ovl/
      catalog.sqlite     # this module (WAL mode; the source of truth)
      ann/               # derived FAISS artifacts (overlap.store.annindex)

The ANN index is always reconstructible from ``hash_chunks``; the catalog is
authoritative. One transaction per file is the resumability unit: a re-run
skips files whose (path, size, mtime) already sit at status ``done``.
"""

from __future__ import annotations

import sqlite3
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from overlap.errors import IndexError_

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from overlap.ingest.model import FileResult

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  file_id    INTEGER PRIMARY KEY,
  abspath    TEXT NOT NULL UNIQUE,
  root       TEXT NOT NULL,
  relpath    TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  mtime_ns   INTEGER NOT NULL,
  sha256     BLOB,
  container  TEXT,
  status     TEXT NOT NULL,
  error      TEXT,
  indexed_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);
CREATE TABLE IF NOT EXISTS streams (
  stream_id   INTEGER PRIMARY KEY,
  file_id     INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
  stream_key  TEXT NOT NULL,
  codec       TEXT,
  width       INTEGER,
  height      INTEGER,
  native_fps  REAL,
  duration_ms INTEGER,
  sample_fps  REAL NOT NULL,
  algo_id     TEXT NOT NULL,
  prep_id     TEXT NOT NULL,
  border_crop TEXT NOT NULL DEFAULT '0,0,0,0',
  n_frames    INTEGER NOT NULL,
  n_crop_rungs INTEGER NOT NULL DEFAULT 0,
  sketch      BLOB,
  UNIQUE(file_id, stream_key)
);
CREATE TABLE IF NOT EXISTS hash_chunks (
  stream_id INTEGER NOT NULL REFERENCES streams(stream_id) ON DELETE CASCADE,
  seq       INTEGER NOT NULL,
  n         INTEGER NOT NULL,
  hashes    BLOB NOT NULL,
  mirrors   BLOB NOT NULL,
  qualities BLOB NOT NULL,
  flags     BLOB NOT NULL,
  crop_hashes  BLOB,
  crop_mirrors BLOB,
  PRIMARY KEY(stream_id, seq)
);
-- Which ANN shard holds each stream's codes, and the frame/rung counts they
-- were built from. Derived data, but it must live beside the hashes it
-- describes: it is what lets a re-open rebuild only the shards whose streams
-- actually changed instead of the whole index.
CREATE TABLE IF NOT EXISTS ann_shards (
  stream_id    INTEGER PRIMARY KEY REFERENCES streams(stream_id) ON DELETE CASCADE,
  shard        TEXT NOT NULL,
  n_frames     INTEGER NOT NULL,
  n_crop_rungs INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ann_shards_shard ON ann_shards(shard);
"""

CHUNK_FRAMES = 1024
HASH_BYTES = 32


@dataclass(frozen=True)
class CatalogStats:
    files_done: int
    files_error: int
    files_skipped: int
    streams: int
    frames: int
    total_duration_ms: int
    db_bytes: int


@dataclass(frozen=True)
class StreamRow:
    stream_id: int
    file_id: int
    stream_key: str
    sample_fps: float
    n_frames: int
    duration_ms: int | None
    n_crop_rungs: int = 0


class Catalog:
    """Data-access layer over catalog.sqlite. Single-writer by design."""

    def __init__(self, con: sqlite3.Connection, index_dir: Path) -> None:
        self._con = con
        self.index_dir = index_dir

    # -- lifecycle ---------------------------------------------------------

    # Settings that change what a hash *means*: mixing them in one index would
    # make its contents incomparable, so a mismatch is refused.
    IDENTITY_META = ("algo_id", "prep_id")

    @classmethod
    def open(
        cls,
        index_dir: Path,
        expected_meta: dict[str, str] | None = None,
        on_setting_change: Callable[[str, str, str], None] | None = None,
    ) -> Catalog:
        """Open (creating if needed) the catalog under ``index_dir``.

        ``expected_meta`` is written on first open. Afterwards, keys in
        :attr:`IDENTITY_META` are *verified* - an index must never mix hashes
        that mean different things. Everything else (sampling rate, crop
        variants) is a per-stream tuning choice that the schema already records
        per stream, so changing it only affects newly indexed files and is
        reported through ``on_setting_change`` rather than refused: otherwise
        enabling one extra variant would cost a full re-index of the archive.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "catalog.sqlite"
        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(_SCHEMA)
        cat = cls(con, index_dir)
        stored_schema = cat.get_meta("schema_version")
        if stored_schema is None:
            cat.set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(stored_schema) != SCHEMA_VERSION:
            con.close()
            raise IndexError_(
                f"index {index_dir} has schema v{stored_schema}, this build needs "
                f"v{SCHEMA_VERSION}; re-index with this version (pre-1.0 has no migrations)"
            )
        if expected_meta:
            for key, value in expected_meta.items():
                stored = cat.get_meta(key)
                if stored is None or stored == value:
                    cat.set_meta(key, value)
                    continue
                if key in cls.IDENTITY_META:
                    con.close()
                    raise IndexError_(
                        f"index {index_dir} was built with {key}={stored}, "
                        f"but this run uses {key}={value}; these hashes are not "
                        f"comparable - use a separate index directory or re-index"
                    )
                if on_setting_change is not None:
                    on_setting_change(key, stored, value)
                cat.set_meta(key, value)
        con.commit()
        return cat

    def close(self) -> None:
        self._con.commit()
        self._con.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self._con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self._con.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # -- resumability --------------------------------------------------------

    def is_done(self, abspath: str, size_bytes: int, mtime_ns: int) -> bool:
        row = self._con.execute(
            "SELECT status, size_bytes, mtime_ns FROM files WHERE abspath=?", (abspath,)
        ).fetchone()
        return (
            row is not None
            and row[0] == "done"
            and int(row[1]) == size_bytes
            and int(row[2]) == mtime_ns
        )

    # -- writes ---------------------------------------------------------------

    def store_file_result(self, result: FileResult) -> int:
        """Persist one file atomically (the resumability unit). Returns file_id."""
        con = self._con
        with con:  # transaction
            con.execute("DELETE FROM files WHERE abspath=?", (result.abspath,))
            cur = con.execute(
                "INSERT INTO files(abspath, root, relpath, size_bytes, mtime_ns, sha256,"
                " container, status, error, indexed_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    result.abspath,
                    result.root,
                    result.relpath,
                    result.size_bytes,
                    result.mtime_ns,
                    result.sha256,
                    result.container,
                    result.status,
                    result.error,
                    int(time.time()),
                ),
            )
            file_id = int(cur.lastrowid)  # type: ignore[arg-type]
            for s in result.streams:
                cur = con.execute(
                    "INSERT INTO streams(file_id, stream_key, codec, width, height,"
                    " native_fps, duration_ms, sample_fps, algo_id, prep_id, border_crop,"
                    " n_frames, n_crop_rungs, sketch) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        file_id,
                        s.stream_key,
                        s.codec,
                        s.width,
                        s.height,
                        s.native_fps,
                        s.duration_ms,
                        s.sample_fps,
                        s.algo_id,
                        s.prep_id,
                        s.border_crop,
                        s.n_frames,
                        s.n_crop_rungs,
                        s.sketch,
                    ),
                )
                stream_id = int(cur.lastrowid)  # type: ignore[arg-type]
                rung_stride = s.n_crop_rungs * HASH_BYTES
                for seq in range(0, (s.n_frames + CHUNK_FRAMES - 1) // CHUNK_FRAMES):
                    lo = seq * CHUNK_FRAMES
                    hi = min(lo + CHUNK_FRAMES, s.n_frames)
                    con.execute(
                        "INSERT INTO hash_chunks(stream_id, seq, n, hashes, mirrors,"
                        " qualities, flags, crop_hashes, crop_mirrors)"
                        " VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            stream_id,
                            seq,
                            hi - lo,
                            zlib.compress(s.hashes[lo * HASH_BYTES : hi * HASH_BYTES], 1),
                            zlib.compress(s.mirrors[lo * HASH_BYTES : hi * HASH_BYTES], 1),
                            zlib.compress(s.qualities[lo:hi], 1),
                            zlib.compress(s.flags[lo:hi], 1),
                            zlib.compress(s.crop_hashes[lo * rung_stride : hi * rung_stride], 1)
                            if rung_stride
                            else None,
                            zlib.compress(s.crop_mirrors[lo * rung_stride : hi * rung_stride], 1)
                            if rung_stride
                            else None,
                        ),
                    )
        return file_id

    # -- reads -----------------------------------------------------------------

    def stats(self) -> CatalogStats:
        con = self._con
        done, err, skip = (
            int(con.execute("SELECT COUNT(*) FROM files WHERE status=?", (s,)).fetchone()[0])
            for s in ("done", "error", "skipped")
        )
        streams = int(con.execute("SELECT COUNT(*) FROM streams").fetchone()[0])
        frames = int(con.execute("SELECT COALESCE(SUM(n_frames),0) FROM streams").fetchone()[0])
        duration = int(
            con.execute("SELECT COALESCE(SUM(duration_ms),0) FROM streams").fetchone()[0]
        )
        db_bytes = (self.index_dir / "catalog.sqlite").stat().st_size
        return CatalogStats(done, err, skip, streams, frames, duration, db_bytes)

    def iter_streams(self) -> Iterator[StreamRow]:
        for row in self._con.execute(
            "SELECT stream_id, file_id, stream_key, sample_fps, n_frames, duration_ms,"
            " n_crop_rungs FROM streams ORDER BY stream_id"
        ):
            yield StreamRow(*row)

    def stream_hashes(self, stream_id: int) -> tuple[bytes, bytes, bytes, bytes]:
        """Return (hashes, mirrors, qualities, flags) concatenated for a stream."""
        hashes, mirrors, qualities, flags = [], [], [], []
        for row in self._con.execute(
            "SELECT hashes, mirrors, qualities, flags FROM hash_chunks"
            " WHERE stream_id=? ORDER BY seq",
            (stream_id,),
        ):
            hashes.append(zlib.decompress(row[0]))
            mirrors.append(zlib.decompress(row[1]))
            qualities.append(zlib.decompress(row[2]))
            flags.append(zlib.decompress(row[3]))
        return b"".join(hashes), b"".join(mirrors), b"".join(qualities), b"".join(flags)

    def stream_crop_hashes(self, stream_id: int) -> tuple[bytes, bytes]:
        """Return (crop_hashes, crop_mirrors) for a stream (empty when no ladder)."""
        crop_hashes, crop_mirrors = [], []
        for row in self._con.execute(
            "SELECT crop_hashes, crop_mirrors FROM hash_chunks WHERE stream_id=? ORDER BY seq",
            (stream_id,),
        ):
            if row[0] is not None:
                crop_hashes.append(zlib.decompress(row[0]))
            if row[1] is not None:
                crop_mirrors.append(zlib.decompress(row[1]))
        return b"".join(crop_hashes), b"".join(crop_mirrors)

    def absorb(self, other: Catalog, *, progress: Callable[[dict[str, Any]], None] | None = None
    ) -> dict[str, int]:
        """Copy another index's fingerprints into this one.

        This is how a corpus larger than one machine gets built. Hashing is the
        expensive step by orders of magnitude - decode plus one hash per frame
        per geometry - so machines fingerprint slices in parallel and the slices
        are combined here, moving hashes rather than recomputing them.

        Streams arrive without a shard assignment, so the next time the index is
        opened they are sharded incrementally: adding a machine's worth of
        footage costs the FAISS build for that footage alone, not a rebuild.

        Files already present by content (sha256) are skipped, which makes a
        merge idempotent and safe to re-run after an interruption.
        """
        emit = progress or (lambda _e: None)
        for key in self.IDENTITY_META:
            mine, theirs = self.get_meta(key), other.get_meta(key)
            if mine is not None and theirs is not None and mine != theirs:
                raise IndexError_(
                    f"cannot merge: this index has {key}={mine}, the other has {key}={theirs}; "
                    f"these hashes are not comparable"
                )
        have = {
            bytes(r[0])
            for r in self._con.execute("SELECT sha256 FROM files WHERE sha256 IS NOT NULL")
        }
        counts = {"files": 0, "streams": 0, "frames": 0, "skipped": 0}
        cols = (
            "abspath, root, relpath, size_bytes, mtime_ns, sha256, container,"
            " status, error, indexed_at"
        )
        for row in other._con.execute(f"SELECT file_id, {cols} FROM files ORDER BY file_id"):
            src_file_id, values = int(row[0]), row[1:]
            sha = values[5]
            if sha is not None and bytes(sha) in have:
                counts["skipped"] += 1
                continue
            with self._con:
                # A merged file keeps the other machine's path, which is a real
                # location on that machine and the only honest thing to record.
                cur = self._con.execute(
                    f"INSERT OR REPLACE INTO files({cols}) VALUES({','.join('?' * 10)})", values
                )
                new_file_id = int(cur.lastrowid)  # type: ignore[arg-type]
                if sha is not None:
                    have.add(bytes(sha))
                counts["files"] += 1
                counts["streams"] += self._absorb_streams(other, src_file_id, new_file_id, counts)
            if counts["files"] % 500 == 0:
                emit({"event": "merge", "files": counts["files"]})
        return counts

    def _absorb_streams(
        self, other: Catalog, src_file_id: int, new_file_id: int, counts: dict[str, int]
    ) -> int:
        scols = (
            "stream_key, codec, width, height, native_fps, duration_ms, sample_fps,"
            " algo_id, prep_id, border_crop, n_frames, n_crop_rungs, sketch"
        )
        n = 0
        for srow in other._con.execute(
            f"SELECT stream_id, {scols} FROM streams WHERE file_id=? ORDER BY stream_id",
            (src_file_id,),
        ):
            src_stream_id = int(srow[0])
            cur = self._con.execute(
                f"INSERT INTO streams(file_id, {scols}) VALUES({','.join('?' * 14)})",
                (new_file_id, *srow[1:]),
            )
            new_stream_id = int(cur.lastrowid)  # type: ignore[arg-type]
            # Chunks move as stored blobs: still compressed, never decompressed,
            # so a merge is I/O and not a re-encode of every hash.
            self._con.executemany(
                "INSERT INTO hash_chunks(stream_id, seq, n, hashes, mirrors, qualities,"
                " flags, crop_hashes, crop_mirrors) VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (new_stream_id, *crow)
                    for crow in other._con.execute(
                        "SELECT seq, n, hashes, mirrors, qualities, flags, crop_hashes,"
                        " crop_mirrors FROM hash_chunks WHERE stream_id=? ORDER BY seq",
                        (src_stream_id,),
                    )
                ],
            )
            counts["frames"] += int(srow[11])
            n += 1
        return n

    # -- ANN shard bookkeeping -------------------------------------------------

    def shard_assignments(self) -> dict[int, tuple[str, int, int]]:
        """stream_id -> (shard name, n_frames, n_crop_rungs) it was sharded at."""
        return {
            int(r[0]): (str(r[1]), int(r[2]), int(r[3]))
            for r in self._con.execute(
                "SELECT stream_id, shard, n_frames, n_crop_rungs FROM ann_shards"
            )
        }

    def assign_shard(self, shard: str, rows: list[tuple[int, int, int]]) -> None:
        """Record that ``rows`` of (stream_id, n_frames, n_crop_rungs) are in ``shard``."""
        with self._con:
            self._con.executemany(
                "INSERT INTO ann_shards(stream_id, shard, n_frames, n_crop_rungs)"
                " VALUES(?,?,?,?) ON CONFLICT(stream_id) DO UPDATE SET"
                " shard=excluded.shard, n_frames=excluded.n_frames,"
                " n_crop_rungs=excluded.n_crop_rungs",
                [(sid, shard, n, rungs) for sid, n, rungs in rows],
            )

    def drop_shard_assignments(self, shards: set[str]) -> None:
        with self._con:
            self._con.executemany(
                "DELETE FROM ann_shards WHERE shard=?", [(s,) for s in sorted(shards)]
            )

    def stream_codec_dims(self, stream_id: int) -> tuple[str, int, int]:
        row = self._con.execute(
            "SELECT codec, width, height FROM streams WHERE stream_id=?", (stream_id,)
        ).fetchone()
        if row is None:
            raise IndexError_(f"no stream {stream_id} in catalog")
        return str(row[0] or ""), int(row[1] or 0), int(row[2] or 0)

    def files_by_status(self, status: str) -> list[tuple[int, str, str]]:
        return [
            (int(r[0]), str(r[1]), str(r[2]))
            for r in self._con.execute(
                "SELECT file_id, abspath, relpath FROM files WHERE status=? ORDER BY relpath",
                (status,),
            )
        ]

    def file_rows(self) -> list[dict[str, Any]]:
        cols = "file_id, abspath, root, relpath, size_bytes, sha256, container, status, error"
        return [
            dict(zip(cols.replace(" ", "").split(","), row, strict=True))
            for row in self._con.execute(f"SELECT {cols} FROM files ORDER BY relpath")
        ]
