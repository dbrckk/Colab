from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


HashFile = Callable[[Path], str]
MediaIntegrity = Callable[[Path, str], bool]


def validate_downloaded_outputs(
    job: dict[str, Any],
    artifacts: list[tuple[Path, str]],
    *,
    hash_file: HashFile,
    media_integrity_ok: MediaIntegrity,
) -> dict[str, Any]:
    if not artifacts:
        raise RuntimeError("Le kernel Kaggle s'est terminé sans produire de fichier.")

    expected_kind = {
        "image": "image",
        "image_edit": "image",
        "image_batch": "image",
        "video_faceswap": "video",
    }.get(job.get("task"))

    data: dict[str, Any] = {}
    manifests = [path for path, _kind in artifacts if path.name == "result.json"]
    if manifests:
        try:
            data = json.loads(manifests[-1].read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"result.json illisible: {exc}") from exc
        if data.get("status") != "done":
            raise RuntimeError(
                "Le worker Kaggle n'a pas confirmé la réussite: "
                + str(data.get("error") or data.get("status") or "statut inconnu")
            )

    artifact_names = {path.name for path, _ in artifacts}
    artifact_by_name = {path.name: path for path, _ in artifacts}
    declared_files = [str(x) for x in (data.get("files") or [])]
    missing_declared = [
        name for name in declared_files if Path(name).name not in artifact_names
    ]
    if missing_declared:
        raise RuntimeError(
            "Le manifeste Kaggle référence des fichiers absents: "
            + ", ".join(missing_declared[:10])
        )

    output_manifest = data.get("output_manifest") or []
    worker_version = str(data.get("worker_version") or "")
    try:
        worker_version_tuple = tuple(int(x) for x in worker_version.split("."))
    except ValueError:
        worker_version_tuple = ()

    if worker_version_tuple >= (1, 3) and declared_files and not output_manifest:
        raise RuntimeError(
            "Manifest SHA-256 absent: le worker v1.3+ doit signer tous les fichiers générés."
        )

    for entry in output_manifest:
        name = Path(str(entry.get("name") or "")).name
        path = artifact_by_name.get(name)
        if not name or path is None:
            raise RuntimeError(f"Manifest SHA-256: fichier absent: {name or '?'}")
        expected_size = int(entry.get("size") or -1)
        if expected_size < 0 or path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Manifest SHA-256: taille invalide pour {name} "
                f"({path.stat().st_size} != {expected_size})."
            )
        expected_sha = str(entry.get("sha256") or "").lower()
        if len(expected_sha) != 64 or hash_file(path).lower() != expected_sha:
            raise RuntimeError(f"Manifest SHA-256: checksum invalide pour {name}.")

    if output_manifest and len(output_manifest) != len(declared_files):
        raise RuntimeError(
            "Manifest SHA-256 incomplet: le nombre d'entrées ne correspond pas aux fichiers déclarés."
        )

    if expected_kind:
        media = [
            path for path, kind in artifacts
            if kind == expected_kind and media_integrity_ok(path, kind)
        ]
        if not media:
            raise RuntimeError(
                f"Le job {job.get('task')} est terminé sans média {expected_kind} décodable."
            )

        if job.get("task") == "image_batch":
            meta = job.get("meta") or {}
            expected_count = int(
                meta.get("batch_count") or len(meta.get("prompts") or [])
            )
            declared_count = int(data.get("batch_count") or 0)
            if declared_count != expected_count:
                raise RuntimeError(
                    f"Lot incomplet: le worker annonce {declared_count}/{expected_count} image(s)."
                )
            if len(media) != expected_count:
                raise RuntimeError(
                    f"Lot incomplet: {len(media)}/{expected_count} image(s) décodable(s) récupérée(s)."
                )

    return data
