from __future__ import annotations

from pathlib import Path


def expired_exports(
    export_root: Path,
    *,
    now: float,
    max_age_days: int,
) -> list[Path]:
    if not export_root.exists():
        return []
    cutoff = now - max(1, int(max_age_days)) * 86400
    result: list[Path] = []
    for path in export_root.glob("*.zip"):
        try:
            if path.stat().st_mtime < cutoff:
                result.append(path)
        except OSError:
            continue
    return result


def orphan_input_dirs(
    input_root: Path,
    *,
    known_ids: set[str],
) -> list[Path]:
    if not input_root.exists():
        return []
    result: list[Path] = []
    for path in input_root.iterdir():
        if path.is_dir() and path.name not in known_ids:
            result.append(path)
    return result


def format_reclaimed_bytes(reclaimed: int) -> str:
    reclaimed = max(0, int(reclaimed))
    if reclaimed >= 1024**3:
        return f"{reclaimed / 1024**3:.2f} Go"
    return f"{reclaimed / 1024**2:.1f} Mo"


def cleanup_report(
    *,
    removed_exports: int,
    removed_inputs: int,
    gc_files: int,
    stale_artifacts: int,
    recoverable_jobs: int,
    expired_recovery_kernels: int,
    reclaimed: int,
) -> str:
    amount = format_reclaimed_bytes(reclaimed)
    return (
        f"Nettoyage terminé • {removed_exports} export(s) ancien(s) • "
        f"{removed_inputs} dossier(s) d'entrée orphelin(s) • "
        f"{gc_files} source(s) partagée(s) non référencée(s) • "
        f"{stale_artifacts} artefact(s) DB obsolète(s) • "
        f"{recoverable_jobs} job(s) récupérable(s) détecté(s) • "
        f"{expired_recovery_kernels} récupération(s) distante(s) expirée(s) • "
        f"{amount} libéré(s)."
    )
