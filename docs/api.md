# Web API reference (`/api/v1`)

`overlap ui` serves this JSON API on loopback; the bundled web pages are a
thin htmx client over it, and scripts can drive it directly. Authentication:
send the session token (printed at launch) as `Authorization: Bearer <token>` - or open the printed URL once to receive a session cookie.
`GET /api/v1/health` is the only unauthenticated route.

## Corpus

| route | description |
|---|---|
| `GET /health` | liveness: `{"status": "ok"}` |
| `GET /version` | tool + manifest schema versions |
| `GET /stats` | corpus statistics (files, hours, fingerprints, meta) |
| `GET /files?limit=N` | indexed file listing |

## Jobs

Long-running work is asynchronous: `POST` returns `202 {"job_id": ...}`;
poll `GET /jobs/{id}` or stream `GET /jobs/{id}/events` (Server-Sent Events - the same event schema the CLI's `--json` mode emits, ending with a
`job_end` event).

| route | description |
|---|---|
| `POST /jobs/index` | `{"paths": [...], "reindex": false}` - fingerprint server-local paths |
| `POST /compare` | multipart upload, field `manifest` - compare an incoming `.ovlm` |
| `POST /self-dedupe` | corpus vs itself |
| `GET /jobs` / `GET /jobs/{id}` | job listing / status (`result.report_id` links the outcome) |
| `POST /jobs/{id}/cancel` | cooperative cancel (between files; indexing is resumable) |
| `GET /jobs/{id}/events` | SSE progress stream |

## Reports & manifests

| route | description |
|---|---|
| `GET /reports` | saved reports (id, summary) |
| `GET /reports/{id}` | full report JSON (schema `report/1`) |
| `GET /reports/{id}/render?format=html\|md` | standalone HTML (email-able) or Markdown |
| `DELETE /reports/{id}` | remove a saved report |
| `POST /export` | `{"label": ..., "anonymize_paths": false}` - build a manifest of the corpus |
| `GET /manifests/{name}` | download a built manifest |

Uploads are capped (`ui.max_upload_mb`, default 512 MB) and manifests are
parsed as untrusted input (strict validation, fail-closed).
