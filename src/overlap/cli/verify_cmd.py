"""`overlap verify` - check delivered files against a manifest's Merkle root."""

from __future__ import annotations

from pathlib import Path

import typer

from overlap.cli._console import emit_document, get_state
from overlap.exit_codes import ExitCode
from overlap.match import verify_delivery


def verify_cmd(
    ctx: typer.Context,
    manifest: Path = typer.Argument(..., exists=True, help="The manifest that was quoted."),
    data: Path = typer.Option(
        ...,
        "--data",
        exists=True,
        file_okay=False,
        help="Directory containing the delivered files.",
    ),
) -> None:
    """Verify a delivery: every file promised in the manifest must arrive
    byte-identical (matching is by content, so renames are fine)."""
    state = get_state(ctx)
    result = verify_delivery(manifest, data)

    if state.json_mode:
        emit_document(result)
    else:
        ok = "[green]OK[/green]" if result["ok"] else "[red]FAILED[/red]"
        state.err.print(
            f"Verification {ok}: {result['files_matched']}/{result['files_expected']} "
            f"files delivered, {result['files_missing']} missing, "
            f"{result['files_extra']} extra."
        )
        if not result["manifest_merkle_ok"]:
            state.err.print("[red]Manifest Merkle root does not match its contents.[/red]")
        for miss in result["missing"][:20]:
            state.err.print(f"  missing: {miss['relpath']}")

    if not result["ok"]:
        raise typer.Exit(code=ExitCode.RUNTIME_ERROR)
