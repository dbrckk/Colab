# 4) Modèles — préparation lazy / arrière-plan (v7.9)
import os, shutil, concurrent.futures, functools, time, threading, traceback
from huggingface_hub import hf_hub_download, list_repo_files

os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'

ACTIVE_MODEL_ROOT = os.path.join(ROOT, 'active_models')
os.makedirs(ACTIVE_MODEL_ROOT, exist_ok=True)
os.makedirs(MODEL_ROOT, exist_ok=True)

# On utilise des noms locaux stables, même si le dépôt distant renomme légèrement les fichiers.
HERETIC_PATH = os.path.join(ACTIVE_MODEL_ROOT, 'qwen3vl_8b_heretic-Q4_K_M.gguf')
MMPROJ_PATH  = os.path.join(ACTIVE_MODEL_ROOT, 'mmproj-qwen3vl_8b_heretic-f16.gguf')
DIT_PATH     = os.path.join(ACTIVE_MODEL_ROOT, 'qwen-image-2.1-Q4_K_M.gguf')
VAE_PATH     = os.path.join(ACTIVE_MODEL_ROOT, 'qwen_image_2.1_vae_bf16.safetensors')

PERSIST_HERETIC_PATH = os.path.join(MODEL_ROOT, os.path.basename(HERETIC_PATH))
PERSIST_MMPROJ_PATH  = os.path.join(MODEL_ROOT, os.path.basename(MMPROJ_PATH))
PERSIST_DIT_PATH     = os.path.join(MODEL_ROOT, os.path.basename(DIT_PATH))
PERSIST_VAE_PATH     = os.path.join(MODEL_ROOT, os.path.basename(VAE_PATH))

PREBUILT_PREFETCH_RESULT = None
PREBUILT_PREFETCH_ERROR = None
PREBUILT_PREFETCH_THREAD = None
BOOTSTRAP_THREAD = None
BOOTSTRAP_LOCK = threading.Lock()
BOOTSTRAP_STATE = {
    'models_ready': False,
    'runtime_ready': False,
    'phase': 'idle',
    'details': 'Non démarré',
    'last_error': '',
    'started_at': None,
    'finished_at': None,
}

@functools.lru_cache(maxsize=8)
def repo_files(repo_id):
    return tuple(list_repo_files(repo_id))


def _good_local(path, min_mb=1):
    return os.path.exists(path) and os.path.getsize(path) >= int(min_mb*1024*1024) and not FORCE_REDOWNLOAD


def _copy_in_background(src, dst):
    if not USE_DRIVE or not BACKGROUND_DRIVE_CACHE or not src or not dst:
        return
    def worker():
        try:
            if _good_local(dst):
                return
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + '.part'
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def resolve_remote_filename(repo_id, preferred, kind=None):
    files = repo_files(repo_id)
    if preferred in files:
        return preferred
    low=[(f, f.lower()) for f in files]
    if kind == 'dit_q4km':
        candidates=[f for f,fl in low if fl.endswith('.gguf') and 'q4_k_m' in fl and 'qwen-image-2.1' in fl and 'text' not in fl]
    elif kind == 'heretic_q4km':
        candidates=[f for f,fl in low if fl.endswith('.gguf') and 'q4_k_m' in fl and 'heretic' in fl and 'qwen3vl' in fl]
    elif kind == 'mmproj':
        candidates=[f for f,fl in low if fl.endswith('.gguf') and 'mmproj' in fl and 'heretic' in fl]
    elif kind == 'vae':
        candidates=[f for f,fl in low if fl.endswith('.safetensors') and 'vae' in fl and 'qwen_image_2.1' in fl]
    else:
        candidates=[]
    if not candidates:
        raise FileNotFoundError(f'Fichier introuvable dans {repo_id}: {preferred}')
    return sorted(candidates, key=lambda x:(0 if 'q4_k_m' in x.lower() else 1, len(x)))[0]


def _stage_from_persistent_or_cache(active_path, persistent_path, repo_id, preferred_name, kind, min_mb=1):
    if _good_local(active_path, min_mb=min_mb):
        return active_path
    if _good_local(persistent_path, min_mb=min_mb):
        os.makedirs(os.path.dirname(active_path), exist_ok=True)
        shutil.copy2(persistent_path, active_path)
        return active_path

    remote_name = resolve_remote_filename(repo_id, preferred_name, kind)
    cached = hf_hub_download(repo_id=repo_id, filename=remote_name, force_download=FORCE_REDOWNLOAD)
    os.makedirs(os.path.dirname(active_path), exist_ok=True)
    shutil.copy2(cached, active_path)
    _copy_in_background(active_path, persistent_path)
    return active_path


def ensure_core_models_ready(verbose=True):
    if BOOTSTRAP_STATE.get('models_ready') and all(_good_local(p, min_mb=1) for p in [HERETIC_PATH, MMPROJ_PATH, DIT_PATH, VAE_PATH]):
        return HERETIC_PATH, MMPROJ_PATH, DIT_PATH, VAE_PATH
    with BOOTSTRAP_LOCK:
        if BOOTSTRAP_STATE.get('models_ready') and all(_good_local(p, min_mb=1) for p in [HERETIC_PATH, MMPROJ_PATH, DIT_PATH, VAE_PATH]):
            return HERETIC_PATH, MMPROJ_PATH, DIT_PATH, VAE_PATH
        BOOTSTRAP_STATE['phase'] = 'models'
        BOOTSTRAP_STATE['details'] = 'Préparation des modèles…'
        jobs=[
            ('Heretic GGUF', HERETIC_REPO, HERETIC_GGUF, 'heretic_q4km', HERETIC_PATH, PERSIST_HERETIC_PATH, 100),
            ('Heretic mmproj', HERETIC_REPO, HERETIC_MMPROJ, 'mmproj', MMPROJ_PATH, PERSIST_MMPROJ_PATH, 100),
            ('DiT Qwen-Image 2.1', DIT_REPO, DIT_FILE, 'dit_q4km', DIT_PATH, PERSIST_DIT_PATH, 500),
            ('VAE', DIT_REPO, VAE_FILE, 'vae', VAE_PATH, PERSIST_VAE_PATH, 10),
        ]

        def worker(item):
            label, repo, preferred, kind, active, persistent, min_mb = item
            BOOTSTRAP_STATE['details'] = f'{label}…'
            path = _stage_from_persistent_or_cache(active, persistent, repo, preferred, kind, min_mb=min_mb)
            return label, path

        max_workers = max(1, min(int(DOWNLOAD_WORKERS or 3), 4))
        if PARALLEL_BOOTSTRAP:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures=[ex.submit(worker, item) for item in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    _ = fut.result()
        else:
            for item in jobs:
                worker(item)

        BOOTSTRAP_STATE['models_ready'] = True
        BOOTSTRAP_STATE['details'] = 'Modèles prêts.'
        return HERETIC_PATH, MMPROJ_PATH, DIT_PATH, VAE_PATH


def bootstrap_status_text():
    started = BOOTSTRAP_STATE.get('started_at') or ''
    finished = BOOTSTRAP_STATE.get('finished_at') or ''
    return (
        f"Phase: {BOOTSTRAP_STATE.get('phase','?')}\n"
        f"Détails: {BOOTSTRAP_STATE.get('details','')}\n"
        f"Modèles prêts: {BOOTSTRAP_STATE.get('models_ready', False)}\n"
        f"Runtime prêt: {BOOTSTRAP_STATE.get('runtime_ready', False)}\n"
        f"Début: {started}\n"
        f"Fin: {finished}\n"
        f"Erreur: {BOOTSTRAP_STATE.get('last_error','')}"
    )
