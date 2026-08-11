# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository. We aim to acknowledge reports within a week.

Please do not open public issues for security reports.

## Threat model summary

overlap is a local, offline tool. Its security-relevant surfaces are:

- **Manifest parsing.** Manifests (`.ovlm`) come from counterparties and are
  untrusted input. They are parsed with strict schema validation, size limits,
  and no code execution (no pickle, no eval). Malformed manifests must fail
  closed with an error, never crash the process in an exploitable way.
- **Media parsing.** Video and robotics container decoding is delegated to
  ffmpeg/libav, OpenCV, and the mcap/rosbags libraries. Decoding untrusted
  media inherits those libraries' vulnerability surface; keep them updated.
- **The local web UI.** `overlap ui` binds to `127.0.0.1` by default and
  requires a per-session token (printed in the launch URL) because lab servers
  are often multi-user machines. The API can read arbitrary paths the
  operating user can read - do not bind it to non-loopback interfaces on
  untrusted networks.

## Guarantees

- **No telemetry. No network I/O anywhere in the codebase.** overlap never
  phones home, checks for updates, or loads remote assets. A pull request
  adding any network call will be rejected; a test in the suite
  (`tests/unit/test_project_guards.py`) scans every source file for
  network clients and raw sockets and fails if any appear.
- Reports and manifests contain perceptual hashes and file metadata, not
  frames or pixels. Note that perceptual hashes are not encryption: they leak
  coarse visual structure by design. Treat manifests as confidential business
  documents, not as anonymized data.
