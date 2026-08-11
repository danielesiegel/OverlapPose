# Windows notes

Windows is a first-class platform for overlap - Windows 11 is a primary
development environment and the full test suite is run there before each
release.

## Install

```powershell
pipx install overlap-cli          # or: uv tool install overlap-cli
overlap doctor                    # verify the environment
```

All required native dependencies (PyAV/FFmpeg libraries, OpenCV, FAISS) ship
as wheels - no compiler, no separate FFmpeg install needed for normal use.
The ffmpeg *CLI* is only needed if you run the test suite
(`winget install Gyan.FFmpeg`).

## Long paths

Deep dataset trees can exceed the legacy 260-character path limit. Enable
long paths once (admin PowerShell):

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

## Index location

The default index lives in `%LOCALAPPDATA%\overlap\corpus.ovl`. For large
corpora put it on a fast volume:

```powershell
overlap --index D:\overlap\corpus.ovl index E:\datasets
# or persistently:
$env:OVERLAP_INDEX = "D:\overlap\corpus.ovl"
```

Indexing a NAS share works (`\\nas\captures\...`), but decoding over SMB is
bandwidth-bound - running overlap on the machine that owns the disks and
using `overlap ui` over SSH port-forwarding is usually faster.

## Development

```powershell
git clone https://github.com/World-Archive/overlap.git
cd overlap
uv sync --group dev --extra ros
uv run pytest -m "not integration"   # fast suite, no ffmpeg needed
uv run pytest                        # full suite (needs ffmpeg on PATH)
```

Notes:

- The repo normalizes line endings via `.gitattributes`; no `core.autocrlf`
  configuration is needed.
- Worker processes use the spawn context (the only option on Windows), so
  anything you pass into the pipeline must be picklable; running the suite
  on Windows catches violations that Linux would silently tolerate.
- Rich progress output renders best in Windows Terminal.
