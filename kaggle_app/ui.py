from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .config import SETTINGS
from .kaggle_runner import KaggleController

controller = KaggleController()

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important; padding-bottom: 48px !important;}
.hero {
  padding: 22px;
  border-radius: 24px;
  border: 1px solid rgba(127,127,127,.22);
  margin-bottom: 12px;
  background: linear-gradient(135deg, rgba(120,80,255,.10), rgba(30,160,255,.06));
  box-shadow: 0 12px 34px rgba(0,0,0,.08);
}
.hero h1 {margin: 0 0 6px 0 !important; font-size: clamp(26px, 5vw, 42px) !important;}
.hero p {margin: 0 !important; opacity: .78; font-size: 15px;}
.panel-card {
  border: 1px solid rgba(127,127,127,.16) !important;
  border-radius: 20px !important;
  padding: 12px !important;
  box-shadow: 0 8px 26px rgba(0,0,0,.05);
}
.primary-action {min-height: 54px !important; font-weight: 750 !important; font-size: 16px !important;}
button {touch-action: manipulation; min-height: 44px !important; border-radius: 12px !important;}
textarea, input {font-size: 16px !important;}
@media (max-width: 720px) {
  .gradio-container {padding-left: 8px !important; padding-right: 8px !important;}
  .hero {padding: 16px; border-radius: 18px;}
  .panel-card {padding: 8px !important; border-radius: 16px !important;}
}
"""

def _task_help(task):
    return {
        "image": "✨ **Création d’image** — écris un prompt. Aucun fichier source requis.",
        "image_edit": "✏️ **Modification d’image** — ajoute une image source et décris précisément le changement.",
        "video_faceswap": "🎬 **Face swap vidéo** — ajoute un visage source et une vidéo cible. Le prompt peut rester vide.",
    }.get(task, "")

def _dashboard():
    try:
        return controller.dashboard_summary()
    except Exception as e:
        return f"Diagnostic tableau de bord indisponible: {type(e).__name__}: {e}"

def _recent_videos(limit=30):
    items = []
    for a in controller.recent_artifacts("video", int(limit)):
        path = a.get("path")
        if path and Path(path).exists():
            items.append(path)
    return items

def _recent_images(limit=60):
    items = []
    for a in controller.recent_artifacts("image", int(limit)):
        path = a.get("path")
        if path and Path(path).exists():
            caption = (a.get("prompt") or a.get("task") or a.get("job_id") or "")[:100]
            items.append((path, caption))
    return items

def _format_elapsed(seconds):
    seconds = max(0, int(seconds or 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"

def _format_job_row(j):
    import time as _time
    created = float(j.get("created_at") or 0)
    updated = float(j.get("updated_at") or created)
    terminal = j.get("status") in {"done", "error", "cancelled", "interrupted"}
    end = updated if terminal else _time.time()
    created_text = _time.strftime("%d/%m %H:%M", _time.localtime(created)) if created else "—"
    return [
        j["id"],
        j["task"],
        j["status"],
        created_text,
        _format_elapsed(end - created if created else 0),
        j["kernel_ref"],
        (j["prompt"] or "")[:90],
        (j["error"] or "")[-180:],
    ]

def _filter_jobs(status_filter="Tous", task_filter="Tous", query=""):
    query = (query or "").strip().lower()
    rows = []
    for j in controller.jobs(500):
        if status_filter and status_filter != "Tous" and j.get("status") != status_filter:
            continue
        if task_filter and task_filter != "Tous" and j.get("task") != task_filter:
            continue
        haystack = " ".join([
            str(j.get("id") or ""),
            str(j.get("prompt") or ""),
            str(j.get("kernel_ref") or ""),
            str(j.get("error") or ""),
        ]).lower()
        if query and query not in haystack:
            continue
        rows.append(_format_job_row(j))
    return rows

def _jobs_table():
    rows = []
    for j in controller.jobs(100):
        rows.append(_format_job_row(j))
    return rows

def _cleanup_storage(days):
    try:
        return controller.cleanup_storage(int(days))
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def _health_check():
    try:
        return controller.health_check()
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def _remote_logs(job_id):
    try:
        return controller.remote_logs(job_id)
    except Exception as e:
        return f"{type(e).__name__}: {e}"

def _save_credentials(username, api_token, legacy_key, persist):
    try:
        msg = controller.save_credentials(username, api_token, legacy_key, persist)
        return msg, "✅ Kaggle prêt"
    except Exception as e:
        return f"{type(e).__name__}: {e}", "❌ Kaggle non configuré"

def _submit_batch(prompts, negative, steps, cfg, seed, aspect):
    try:
        ids = controller.submit_batch(prompts, negative, steps, cfg, seed, aspect)
        count = len([line for line in (prompts or "").splitlines() if line.strip()])
        return "\n".join(ids), f"Lot de {count} image(s) ajouté comme un seul job Kaggle.", _jobs_table()
    except Exception as e:
        raise gr.Error(str(e))

def _submit(task, prompt, negative, steps, cfg, seed, aspect, source, target):
    try:
        job_id = controller.submit(
            task,
            prompt,
            negative,
            steps,
            cfg,
            seed,
            aspect,
            source_image=source,
            target_video=target,
        )
        return job_id, f"Job {job_id} ajouté à la file Kaggle.", _jobs_table()
    except Exception as e:
        raise gr.Error(str(e))

def _load_job_to_form(job_id):
    j = controller.job(job_id)
    if not j:
        raise gr.Error("Job introuvable.")
    meta = j.get("meta") or {}
    source = meta.get("source_image") or None
    target = meta.get("target_video") or None
    if source and not Path(source).exists():
        source = None
    if target and not Path(target).exists():
        target = None

    task_value = j.get("task") or "image"
    batch_text = ""
    prompt_value = j.get("prompt") or ""
    if task_value == "image_batch":
        prompts = [str(x) for x in (meta.get("prompts") or []) if str(x).strip()]
        batch_text = "\n".join(prompts)
        task_value = "image"
        prompt_value = prompts[0] if prompts else ""

    return (
        task_value,
        prompt_value,
        meta.get("negative_prompt") or "",
        int(meta.get("steps", 25)),
        float(meta.get("cfg", 1.0)),
        int(meta.get("seed", -1)),
        meta.get("aspect") or "1:1",
        source,
        target,
        batch_text,
        f"Paramètres du job {job_id} chargés dans le formulaire.",
    )


def _retry_job(job_id):
    try:
        new_id = controller.retry(job_id)
        return new_id, f"Relance créée : {new_id}", _jobs_table()
    except Exception as e:
        raise gr.Error(str(e))

def _export_job(job_id):
    try:
        return controller.export_job_archive(job_id)
    except Exception as e:
        raise gr.Error(str(e))

def _delete_job(job_id):
    try:
        msg = controller.delete_local_job(job_id)
        return msg, _jobs_table()
    except Exception as e:
        raise gr.Error(str(e))

def _cancel_job(job_id):
    try:
        return controller.cancel(job_id), _jobs_table()
    except Exception as e:
        raise gr.Error(str(e))

def _refresh(job_id):
    if not job_id:
        return "Aucun job sélectionné.", [], None, [], _jobs_table()
    j = controller.job(job_id)
    if not j:
        return "Job introuvable.", [], None, [], _jobs_table()

    artifacts = controller.artifacts(job_id)
    images = [a["path"] for a in artifacts if a["kind"] == "image" and Path(a["path"]).exists()]
    videos = [a["path"] for a in artifacts if a["kind"] == "video" and Path(a["path"]).exists()]
    files = [a["path"] for a in artifacts if Path(a["path"]).exists()]

    status = f"{j['status']}\nKaggle: {j['kernel_ref'] or '—'}"
    result_manifest = (j.get("meta") or {}).get("result_manifest") or {}
    if result_manifest:
        details = []
        if result_manifest.get("seed") is not None:
            details.append(f"seed={result_manifest.get('seed')}")
        if result_manifest.get("width") and result_manifest.get("height"):
            details.append(f"{result_manifest.get('width')}×{result_manifest.get('height')}")
        if result_manifest.get("steps") is not None:
            details.append(f"steps={result_manifest.get('steps')}")
        if result_manifest.get("worker_version"):
            details.append(f"worker={result_manifest.get('worker_version')}")
        if details:
            status += "\nRésultat : " + " • ".join(details)
    position = controller.queue_position(job_id)
    if position is not None:
        status += f"\nPosition file locale : {position}"
    size = controller.job_storage_bytes(job_id)
    if size:
        status += f"\nStockage récupéré : {size / (1024**2):.1f} Mo"
    status += f"\nFichiers récupérés : {len(files)}"
    if j["error"]:
        status += "\n" + j["error"]
    return status, images, (videos[0] if videos else None), files, _jobs_table()

def build_ui():
    with gr.Blocks(title="Qwen Kaggle Studio") as demo:
        gr.HTML(
            "<div class='hero'><h1>Qwen Kaggle Studio</h1>"
            "<p>Crée, modifie et traite tes médias sur Kaggle GPU. Les résultats reviennent automatiquement dans ta bibliothèque persistante.</p></div>"
        )
        dashboard = gr.Markdown(_dashboard())

        with gr.Tab("✨ Générer"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["panel-card"]):
                    task = gr.Dropdown(
                        [
                            ("✨ Créer une image", "image"),
                            ("✏️ Modifier une image", "image_edit"),
                            ("🎬 Face swap vidéo", "video_faceswap"),
                        ],
                        value="image",
                        label="Tâche",
                    )
                    task_help = gr.Markdown(_task_help("image"))
                    prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Décris l'image à générer…")
                    negative = gr.Textbox(label="Negative prompt", lines=2)
                    with gr.Row():
                        aspect = gr.Dropdown(
                            ["1:1", "4:3", "3:4", "16:9", "9:16"],
                            value="1:1",
                            label="Format",
                        )
                        steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
                    with gr.Row():
                        cfg = gr.Slider(1, 8, value=1.0, step=.5, label="CFG")
                        seed = gr.Number(value=-1, precision=0, label="Seed")
                    with gr.Accordion("Image/vidéo source", open=False):
                        source = gr.Image(label="Image source / visage source", type="filepath")
                        target = gr.Video(label="Vidéo cible")
                    submit = gr.Button("🚀 Lancer sur Kaggle", variant="primary", elem_classes=["primary-action"])
                    job_id = gr.Textbox(label="Job ID")
                    submit_info = gr.Textbox(label="Envoi", interactive=False)
                    with gr.Accordion("📦 Génération par lot", open=False):
                        gr.Markdown("Un prompt par ligne • maximum 20 images • un seul job Kaggle avec chargement Qwen partagé.")
                        batch_prompts = gr.Textbox(
                            label="Prompts du lot",
                            lines=8,
                            placeholder="Prompt 1\nPrompt 2\nPrompt 3",
                        )
                        batch_submit = gr.Button("Ajouter le lot à la file")
                        batch_ids = gr.Textbox(label="Jobs créés", lines=6, interactive=False)

                with gr.Column(scale=6, elem_classes=["panel-card"]):
                    status = gr.Textbox(label="État", lines=5, interactive=False)
                    with gr.Row():
                        refresh = gr.Button("↻ Actualiser")
                        retry = gr.Button("↻ Relancer")
                        cancel = gr.Button("⛔ Annuler le job", variant="stop")
                    gallery = gr.Gallery(label="Images récupérées", columns=2, height=420)
                    video = gr.Video(label="Vidéo récupérée")
                    files = gr.Files(label="Tous les fichiers du job")

        with gr.Tab("📚 Bibliothèque / Jobs"):
            with gr.Accordion("🖼 Galerie récente", open=True):
                with gr.Row():
                    recent_limit = gr.Slider(12, 100, value=48, step=4, label="Images récentes")
                    recent_reload = gr.Button("↻ Actualiser la galerie")
                recent_gallery = gr.Gallery(
                    label="Images générées récemment",
                    value=_recent_images(48),
                    columns=4,
                    height=520,
                )
            with gr.Accordion("🎬 Vidéos récentes", open=False):
                with gr.Row():
                    recent_video_limit = gr.Slider(5, 50, value=20, step=5, label="Vidéos récentes")
                    recent_video_reload = gr.Button("↻ Actualiser les vidéos")
                recent_videos = gr.Files(
                    label="Vidéos générées récemment",
                    value=_recent_videos(20),
                )
            with gr.Row():
                job_status_filter = gr.Dropdown(
                    ["Tous","queued","waiting_auth","preparing","uploading_inputs","submitting","running","recovering","downloading","done","error","cancelled","interrupted"],
                    value="Tous",
                    label="Statut",
                )
                job_task_filter = gr.Dropdown(
                    ["Tous","image","image_batch","image_edit","video_faceswap"],
                    value="Tous",
                    label="Type",
                )
                job_search = gr.Textbox(label="Rechercher", placeholder="Prompt, Job ID, erreur…")
                job_filter_btn = gr.Button("Filtrer")
            jobs = gr.Dataframe(
                headers=["Job ID", "Type", "État", "Créé", "Durée", "Kernel Kaggle", "Prompt", "Erreur"],
                value=_jobs_table(),
                interactive=False,
                wrap=True,
                label="Historique",
            )
            reload_jobs = gr.Button("↻ Rafraîchir")
            lookup = gr.Textbox(label="Job ID à ouvrir")
            with gr.Row():
                open_job = gr.Button("Ouvrir le job", variant="primary")
                load_form_btn = gr.Button("↙ Charger dans Générer")
                retry_job_btn = gr.Button("↻ Relancer le job")
                logs_btn = gr.Button("📜 Logs Kaggle")
                export_job_btn = gr.Button("📦 Export ZIP")
                delete_job_btn = gr.Button("🗑 Supprimer localement", variant="stop")
            lib_status = gr.Textbox(label="État", lines=5, interactive=False)
            lib_logs = gr.Textbox(label="Logs Kaggle", lines=12, interactive=False, visible=True)
            lib_gallery = gr.Gallery(label="Images", columns=3)
            lib_video = gr.Video(label="Vidéo")
            lib_files = gr.Files(label="Fichiers")
            lib_export = gr.File(label="Archive du job")

        with gr.Tab("⚙️ Paramètres Kaggle"):
            ready = "✅ Kaggle prêt" if controller.credentials_ready() else "❌ Kaggle non configuré"
            gr.Markdown(
                "Les identifiants restent dans le processus local. "
                "Si tu coches la sauvegarde, ils sont écrits dans .env.local, ignoré par Git."
            )
            kaggle_status = gr.Textbox(value=ready, label="État", interactive=False)
            username = gr.Textbox(value=os.getenv("KAGGLE_USERNAME", ""), label="KAGGLE_USERNAME")
            api_token = gr.Textbox(value="", label="KAGGLE_API_TOKEN (recommandé)", type="password")
            with gr.Accordion("Ancienne clé Kaggle (optionnel)", open=False):
                legacy_key = gr.Textbox(value="", label="KAGGLE_KEY legacy", type="password")
            persist = gr.Checkbox(value=True, label="Sauvegarder dans le stockage privé configuré")
            with gr.Row():
                save = gr.Button("Enregistrer et tester Kaggle", variant="primary")
                health_btn = gr.Button("🩺 Diagnostic complet")
            save_msg = gr.Textbox(label="Résultat", lines=3, interactive=False)
            health_out = gr.Textbox(label="Diagnostic", lines=8, interactive=False)
            with gr.Accordion("Entretien du stockage", open=False):
                cleanup_days = gr.Slider(1, 90, value=14, step=1, label="Supprimer les ZIP d'export plus vieux que N jours")
                cleanup_btn = gr.Button("🧹 Nettoyage sûr")
                cleanup_out = gr.Textbox(label="Nettoyage", lines=3, interactive=False)

        task.change(_task_help, [task], [task_help])
        submit.click(
            _submit,
            [task, prompt, negative, steps, cfg, seed, aspect, source, target],
            [job_id, submit_info, jobs],
        )
        batch_submit.click(
            _submit_batch,
            [batch_prompts, negative, steps, cfg, seed, aspect],
            [batch_ids, submit_info, jobs],
        )
        refresh.click(_refresh, [job_id], [status, gallery, video, files, jobs])
        retry.click(_retry_job, [job_id], [job_id, submit_info, jobs])
        cancel.click(_cancel_job, [job_id], [submit_info, jobs])
        reload_jobs.click(_jobs_table, [], [jobs])
        job_filter_btn.click(
            _filter_jobs,
            [job_status_filter, job_task_filter, job_search],
            [jobs],
        )
        recent_reload.click(_recent_images, [recent_limit], [recent_gallery])
        recent_video_reload.click(_recent_videos, [recent_video_limit], [recent_videos])
        open_job.click(_refresh, [lookup], [lib_status, lib_gallery, lib_video, lib_files, jobs])
        load_form_btn.click(
            _load_job_to_form,
            [lookup],
            [task, prompt, negative, steps, cfg, seed, aspect, source, target, batch_prompts, submit_info],
        )
        retry_job_btn.click(_retry_job, [lookup], [lookup, lib_status, jobs])
        logs_btn.click(_remote_logs, [lookup], [lib_logs])
        export_job_btn.click(_export_job, [lookup], [lib_export])
        delete_job_btn.click(_delete_job, [lookup], [lib_status, jobs])
        save.click(
            _save_credentials,
            [username, api_token, legacy_key, persist],
            [save_msg, kaggle_status],
        )
        health_btn.click(_health_check, [], [health_out])
        cleanup_btn.click(_cleanup_storage, [cleanup_days], [cleanup_out])

        try:
            timer = gr.Timer(value=5.0, active=True)
            timer.tick(
                _refresh,
                [job_id],
                [status, gallery, video, files, jobs],
                show_progress="hidden",
            )
            dashboard_timer = gr.Timer(value=10.0, active=True)
            dashboard_timer.tick(_dashboard, [], [dashboard], show_progress="hidden")
        except Exception:
            pass

    return demo
