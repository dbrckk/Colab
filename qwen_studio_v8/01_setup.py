#@title 1) Configuration
USE_DRIVE = True #@param {type:"boolean"}
SHARE_GRADIO = True #@param {type:"boolean"}
FORCE_REDOWNLOAD = False #@param {type:"boolean"}
FORCE_REBUILD_SDCPP = False #@param {type:"boolean"}

# Text encoder EXACT envoyé par l'utilisateur
HERETIC_REPO = "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF"
HERETIC_GGUF = "qwen3vl_8b_heretic-Q4_K_M.gguf"
HERETIC_MMPROJ = "mmproj-qwen3vl_8b_heretic-f16.gguf"

# Qwen-Image 2.1 visuel + VAE nécessaires autour du text encoder
DIT_REPO = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
# Nom actuel du Q4_K_M. La cellule de téléchargement sait aussi le retrouver automatiquement si renommé.
DIT_FILE = "qwen-image-2.1-UC-Q4_K_M.gguf"
VAE_FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"

ROOT = "/content/qwen_heretic_gradio"

# IMPORTANT : ne créer AUCUN dossier sous le point de montage avant drive.mount().
# On utilise /content/gdrive pour éviter le /content/drive éventuellement pollué
# par une ancienne exécution du notebook.
DRIVE_MOUNT = "/content/gdrive"
DRIVE_ROOT = f"{DRIVE_MOUNT}/MyDrive/QwenImage21_Heretic_GGUF"
MODEL_ROOT = f"{DRIVE_ROOT}/models" if USE_DRIVE else f"{ROOT}/models"
OUTPUT_ROOT = f"{DRIVE_ROOT}/outputs" if USE_DRIVE else f"{ROOT}/outputs"
SDCPP_ROOT = "/content/stable-diffusion.cpp"
SDCLI_CACHE = f"{DRIVE_ROOT}/bin/sd-cli" if USE_DRIVE else f"{ROOT}/bin/sd-cli"

import os
# Avant montage Drive, on crée uniquement les dossiers hors Drive.
os.makedirs(ROOT, exist_ok=True)
if not USE_DRIVE:
    for p in [MODEL_ROOT, OUTPUT_ROOT, os.path.dirname(SDCLI_CACHE)]:
        os.makedirs(p, exist_ok=True)

print("Text encoder:", HERETIC_REPO, HERETIC_GGUF)
print("Point de montage Drive:", DRIVE_MOUNT if USE_DRIVE else "désactivé")
print("Modèles persistants:", MODEL_ROOT)

# Jobs persistants
JOB_WORKERS = 1 #@param {type:"integer"}
JOB_POLL_SECONDS = 3 #@param {type:"integer"}

# Optimisations v7.5
FAST_INSTALL = True #@param {type:"boolean"}
DOWNLOAD_WORKERS = 3 #@param {type:"integer"}
BUILD_TIMEOUT_MINUTES = 18 #@param {type:"integer"}
MAX_BUILD_JOBS = 8 #@param {type:"integer"}

# v7.6 : démarrage rapide
LAZY_FACE_SWAP_INSTALL = True #@param {type:"boolean"}
FACE_SWAP_GPU_RUNTIME = True #@param {type:"boolean"}
PIP_TIMEOUT_SECONDS = 180 #@param {type:"integer"}

FAST_FIRST_BUILD = True #@param {type:"boolean"}
BUILD_TIMEOUT_MINUTES = 35 #@param {type:"integer"}
CCACHE_MAX_GB = 6 #@param {type:"integer"}

# v7.8 : bootstrap accéléré
USE_PREBUILT_CUDA_RUNTIME = True #@param {type:"boolean"}
PREBUILT_IMAGE = "ghcr.io/leejet/stable-diffusion.cpp:master-cuda" #@param {type:"string"}
PARALLEL_BOOTSTRAP = True #@param {type:"boolean"}
BACKGROUND_DRIVE_CACHE = True #@param {type:"boolean"}
CRANE_VERSION = "v0.22.1" #@param {type:"string"}
SD_RUNTIME_DIR = "/content/sdcpp-prebuilt"
SD_RUNTIME_CACHE_TAR = f"{DRIVE_ROOT}/bin/sdcpp-master-cuda-runtime.tar.gz" if USE_DRIVE else f"{ROOT}/bin/sdcpp-master-cuda-runtime.tar.gz"


# v7.9 : UX plus rapide
LAZY_BOOTSTRAP = True #@param {type:"boolean"}
AUTO_START_BACKGROUND_PREPARE = True #@param {type:"boolean"}
SHOW_IMAGES_ONLY_IN_GRADIO = True #@param {type:"boolean"}
GRADIO_ANALYTICS = False #@param {type:"boolean"}

# v8 — UX
UI_TITLE = "Qwen Studio — Heretic GGUF"
UI_SUBTITLE = "Qwen-Image 2.1 • Heretic Q4_K_M • Video Face Swap Ultra v8.6"
AUTO_REFRESH_SECONDS = 3 #@param {type:"integer"}


#@title 2) Monter Google Drive — corrigé
if USE_DRIVE:
    import os
    from google.colab import drive

    # Ne rien créer dans DRIVE_MOUNT avant cet appel.
    # Si Drive est déjà monté ici, Colab le détecte normalement et ne remonte pas.
    drive.mount(DRIVE_MOUNT, force_remount=False)

    # Seulement APRÈS le montage, créer nos dossiers persistants.
    for p in [DRIVE_ROOT, MODEL_ROOT, OUTPUT_ROOT, os.path.dirname(SDCLI_CACHE)]:
        os.makedirs(p, exist_ok=True)

    print("Drive prêt:", DRIVE_ROOT)
    print("Cache modèles:", MODEL_ROOT)
else:
    print("Drive désactivé. Les fichiers seront perdus à la fin de la session Colab.")


# 3) Dépendances Python — v7.8 MINIMALES
import subprocess, sys, importlib.util, os, time


def run_pip(packages, timeout=None, binary_only=False):
    if isinstance(packages, str):
        packages=[packages]
    cmd=[sys.executable, '-m', 'pip', 'install',
         '--disable-pip-version-check', '--no-input', '--prefer-binary',
         '--retries', '1', '--timeout', '35']
    if binary_only:
        cmd += ['--only-binary=:all:']
    cmd += list(packages)
    print('>', ' '.join(cmd), flush=True)
    started=time.time()
    p=subprocess.run(cmd, timeout=timeout or 150)
    print(f'pip terminé en {time.time()-started:.1f}s | code={p.returncode}', flush=True)
    if p.returncode != 0:
        raise RuntimeError('Installation pip échouée: ' + ' '.join(packages))

# Seulement ce qui est requis pour ouvrir Gradio et télécharger les modèles.
core_checks = {
    'gradio': 'gradio',
    'huggingface_hub': 'huggingface_hub',
    'PIL': 'pillow',
    'requests': 'requests',
}
missing=[pkg for mod,pkg in core_checks.items() if importlib.util.find_spec(mod) is None]
if missing:
    print('Dépendances manquantes:', missing, flush=True)
    run_pip(missing)
else:
    print('✅ Aucune installation pip nécessaire au démarrage.', flush=True)

# hf_xet n'est PLUS installé ici. S'il est déjà fourni par Colab/HF, il sera utilisé.
# Sinon huggingface_hub retombe sur son transport normal sans bloquer le bootstrap.
os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '12'
print('✅ Dépendances de démarrage prêtes.')
print('ℹ️ Face Swap / ONNX restent différés au premier usage.')


# v8.2 — face swap vidéo robuste
VIDEO_FACE_SWAP_ENABLED = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_CRF = 17 #@param {type:"integer"}
VIDEO_FACE_SWAP_PRESET = "slow" #@param ["medium", "slow", "slower"]
VIDEO_FACE_SWAP_KEEP_AUDIO = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_FRAME_STRIDE = 1 #@param {type:"integer"}
VIDEO_FACE_SWAP_MAX_FRAMES = 0 #@param {type:"integer"}
VIDEO_FACE_SWAP_DETECT_EVERY = 1 #@param {type:"integer"}
VIDEO_FACE_SWAP_STRICT_TRACKING = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_MIN_DET_SCORE = 0.55 #@param {type:"number"}
VIDEO_FACE_SWAP_MIN_EMBED_SIM = 0.18 #@param {type:"number"}
VIDEO_FACE_SWAP_MIN_IOU = 0.05 #@param {type:"number"}
VIDEO_FACE_SWAP_AREA_RATIO_MIN = 0.45 #@param {type:"number"}
VIDEO_FACE_SWAP_AREA_RATIO_MAX = 2.25 #@param {type:"number"}
VIDEO_FACE_SWAP_SKIP_ON_OCCLUSION = True #@param {type:"boolean"}


# v8.4 — Face swap vidéo Ultra
VIDEO_FACE_SWAP_PREVIEW_SECONDS = 0 #@param {type:"integer"}
VIDEO_FACE_SWAP_CLEANUP_TEMP = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_PROGRESS_EVERY = 8 #@param {type:"integer"}
VIDEO_FACE_SWAP_AUTO_RESUME = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_CANCEL_ENABLED = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_PRESERVE_MOUTH = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_MOUTH_BLEND = 0.78 #@param {type:"number"}
VIDEO_FACE_SWAP_RECOVERY_STABLE_FRAMES = 2 #@param {type:"integer"}
VIDEO_FACE_SWAP_AUTO_MODE = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_EASY_DET_SCORE = 0.75 #@param {type:"number"}
VIDEO_FACE_SWAP_HARD_MODE_CRF = 16 #@param {type:"integer"}


# v8.4 — Backend FaceFusion Ultra optionnel (lazy)
VIDEO_FACEFUSION_ENABLED = True #@param {type:"boolean"}
VIDEO_FACEFUSION_VERSION = "3.9.0" #@param {type:"string"}
VIDEO_FACEFUSION_ROOT = "/content/facefusion"
VIDEO_FACEFUSION_VENV = "/content/facefusion-venv"
VIDEO_FACEFUSION_EXPRESSION_RESTORER = True #@param {type:"boolean"}
VIDEO_FACEFUSION_FACE_ENHANCER = True #@param {type:"boolean"}
VIDEO_FACEFUSION_ENHANCER_BLEND = 60 #@param {type:"integer"}
VIDEO_FACEFUSION_EXPRESSION_FACTOR = 80 #@param {type:"integer"}
VIDEO_FACEFUSION_MASK_TYPES = "occlusion region" #@param {type:"string"}


# v8.5 — stabilité / UX vidéo
VIDEO_FACE_SWAP_USE_SEAMLESS_BLEND = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_MASK_DILATE = 18 #@param {type:"integer"}
VIDEO_FACE_SWAP_MASK_BLUR = 21 #@param {type:"integer"}
VIDEO_FACE_SWAP_PAUSE_ENABLED = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_ANALYSIS_SAMPLE_FRAMES = 12 #@param {type:"integer"}
VIDEO_FACE_SWAP_CACHE_MAX_AGE_HOURS = 24 #@param {type:"integer"}
VIDEO_FACE_SWAP_CACHE_KEEP_RECENT = 3 #@param {type:"integer"}
VIDEO_FACE_SWAP_AUTO_REFRESH = True #@param {type:"boolean"}
VIDEO_FACEFUSION_AUTO_INSTALL_ON_HARD = True #@param {type:"boolean"}

VIDEO_FACE_SWAP_PREFLIGHT_SECONDS = 5 #@param {type:"integer"}

VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_CHECKPOINT_EVERY = 120 #@param {type:"integer"}

# v8.6 — pipeline vidéo streaming / faible disque
VIDEO_FACE_SWAP_STREAM_INPUT = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_DELETE_CHECKPOINTED_FRAMES = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_MIN_FREE_DISK_GB = 4 #@param {type:"integer"}

# v8.6 — cache persistant FaceFusion (lazy, n'affecte pas le démarrage normal)
VIDEO_FACEFUSION_CACHE_ENABLED = True #@param {type:"boolean"}
VIDEO_FACEFUSION_CACHE_DIR = f"{DRIVE_ROOT}/facefusion_cache" if USE_DRIVE else f"{ROOT}/facefusion_cache"
VIDEO_FACEFUSION_CACHE_TAR = f"{VIDEO_FACEFUSION_CACHE_DIR}/facefusion_{VIDEO_FACEFUSION_VERSION}.tar.gz"

# v8.6 — priorité qualité pour le choix automatique vidéo
VIDEO_FACE_SWAP_AUTO_QUALITY_FIRST = True #@param {type:"boolean"}
VIDEO_FACE_SWAP_AUTO_FACEFUSION_FOR_AUDIO = True #@param {type:"boolean"}
