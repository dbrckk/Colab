from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .config import SETTINGS
from .kaggle_runner import KaggleController

controller = KaggleController()

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important; padding-bottom: 48px !important;}
.hero {padding: 18px; border-radius: 20px; border: 1px solid rgba(127,127,127,.2); margin-bottom: 12px;}
.primary-action {min-height: 52px !important; font-weight: 700 !important;}
button {touch-action: manipulation; min-height: 44px !important;}
textarea, input {font-size: 16px !important;}
@media (max-width: 720px) {
  .gradio-container {padding-left: 8px !important; padding-right: 8px !important;}
  .hero {padding: 14px; border-radius: 16px;}
}
"""

def _jobs_table():
    rows = []
    for j in controller.jobs(100):
        rows.append([
            j["id"],
            j["task"],
            j["status"],
            j["kernel_ref"],
            (j["prompt"] or "")[:90],
            (j["error"] or "")[-180:],
        ])
    return rows

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

def _retry_job(job_id):
    try:
        new_id = controller.retry(job_id)
        return new_id, f"Relance créée : {new_id}", _jobs_table()
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
            "<p>Prompt/upload → Kaggle GPU → récupération automatique → bibliothèque locale.</p></div>"
        )

        with gr.Tab("✨ Générer"):
            with gr.Row():
                with gr.Column(scale=5):
                    task = gr.Dropdown(["image", "image_edit", "video_faceswap"], value="image", label="Tâche")
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

                with gr.Column(scale=6):
                    status = gr.Textbox(label="État", lines=5, interactive=False)
                    with gr.Row():
                        refresh = gr.Button("↻ Actualiser")
                        retry = gr.Button("↻ Relancer")
                        cancel = gr.Button("⛔ Annuler le job", variant="stop")
                    gallery = gr.Gallery(label="Images récupérées", columns=2, height=420)
                    video = gr.Video(label="Vidéo récupérée")
                    files = gr.Files(label="Tous les fichiers du job")

        with gr.Tab("📚 Bibliothèque / Jobs"):
            jobs = gr.Dataframe(
                headers=["Job ID", "Type", "État", "Kernel Kaggle", "Prompt", "Erreur"],
                value=_jobs_table(),
                interactive=False,
                wrap=True,
                label="Historique",
            )
            reload_jobs = gr.Button("↻ Rafraîchir")
            lookup = gr.Textbox(label="Job ID à ouvrir")
            with gr.Row():
                open_job = gr.Button("Ouvrir le job", variant="primary")
                retry_job_btn = gr.Button("↻ Relancer le job")
                logs_btn = gr.Button("📜 Logs Kaggle")
                delete_job_btn = gr.Button("🗑 Supprimer localement", variant="stop")
            lib_status = gr.Textbox(label="État", lines=5, interactive=False)
            lib_logs = gr.Textbox(label="Logs Kaggle", lines=12, interactive=False, visible=True)
            lib_gallery = gr.Gallery(label="Images", columns=3)
            lib_video = gr.Video(label="Vidéo")
            lib_files = gr.Files(label="Fichiers")

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

        submit.click(
            _submit,
            [task, prompt, negative, steps, cfg, seed, aspect, source, target],
            [job_id, submit_info, jobs],
        )
        refresh.click(_refresh, [job_id], [status, gallery, video, files, jobs])
        retry.click(_retry_job, [job_id], [job_id, submit_info, jobs])
        cancel.click(_cancel_job, [job_id], [submit_info, jobs])
        reload_jobs.click(_jobs_table, [], [jobs])
        open_job.click(_refresh, [lookup], [lib_status, lib_gallery, lib_video, lib_files, jobs])
        retry_job_btn.click(_retry_job, [lookup], [lookup, lib_status, jobs])
        logs_btn.click(_remote_logs, [lookup], [lib_logs])
        delete_job_btn.click(_delete_job, [lookup], [lib_status, jobs])
        save.click(
            _save_credentials,
            [username, api_token, legacy_key, persist],
            [save_msg, kaggle_status],
        )
        health_btn.click(_health_check, [], [health_out])

        try:
            timer = gr.Timer(value=5.0, active=True)
            timer.tick(
                _refresh,
                [job_id],
                [status, gallery, video, files, jobs],
                show_progress="hidden",
            )
        except Exception:
            pass

    return demo
