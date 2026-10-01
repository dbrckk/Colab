from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .config import SETTINGS
from .kaggle_runner import KaggleController

controller = KaggleController()

CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.hero {padding: 18px; border-radius: 20px; border: 1px solid rgba(127,127,127,.2); margin-bottom: 12px;}
.primary-action {min-height: 50px !important; font-weight: 700 !important;}
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

def _save_credentials(username, key, persist):
    try:
        msg = controller.save_credentials(username, key, persist)
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
                    task = gr.Dropdown(["image", "video_faceswap"], value="image", label="Tâche")
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
                    with gr.Accordion("Entrées vidéo / face swap", open=False):
                        source = gr.Image(label="Visage source", type="filepath")
                        target = gr.Video(label="Vidéo cible")
                    submit = gr.Button("🚀 Lancer sur Kaggle", variant="primary", elem_classes=["primary-action"])
                    job_id = gr.Textbox(label="Job ID")
                    submit_info = gr.Textbox(label="Envoi", interactive=False)

                with gr.Column(scale=6):
                    status = gr.Textbox(label="État", lines=5, interactive=False)
                    refresh = gr.Button("↻ Actualiser")
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
            open_job = gr.Button("Ouvrir le job", variant="primary")
            lib_status = gr.Textbox(label="État", lines=5, interactive=False)
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
            key = gr.Textbox(value="", label="KAGGLE_KEY", type="password")
            persist = gr.Checkbox(value=True, label="Sauvegarder localement dans .env.local")
            save = gr.Button("Enregistrer et tester Kaggle", variant="primary")
            save_msg = gr.Textbox(label="Résultat", lines=3, interactive=False)

        submit.click(
            _submit,
            [task, prompt, negative, steps, cfg, seed, aspect, source, target],
            [job_id, submit_info, jobs],
        )
        refresh.click(_refresh, [job_id], [status, gallery, video, files, jobs])
        reload_jobs.click(_jobs_table, [], [jobs])
        open_job.click(_refresh, [lookup], [lib_status, lib_gallery, lib_video, lib_files, jobs])
        save.click(_save_credentials, [username, key, persist], [save_msg, kaggle_status])

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
