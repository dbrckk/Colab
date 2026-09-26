# 5) stable-diffusion.cpp — runtime lazy / fast path (v7.9)
import os, shutil, subprocess, time, multiprocessing, pathlib, threading, tarfile

SDCLI = None
SDCLI_PREP_THREAD = None
SDCLI_LOCK = threading.Lock()


def sdcli_works(path):
    if not os.path.isfile(path):
        return False
    try:
        os.chmod(path, 0o755)
        env=os.environ.copy()
        bindir=os.path.dirname(path)
        env['LD_LIBRARY_PATH']=bindir+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        p=subprocess.run([path, '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, env=env)
        return p.returncode in (0,1)
    except Exception:
        return False


def find_sdcli(root):
    candidates=[
        os.path.join(root,'sd.cpp','bin','sd-cli'),
        os.path.join(root,'bin','sd-cli'),
        os.path.join(root,'sd-cli'),
    ]
    for p in candidates:
        if sdcli_works(p):
            return p
    for p in pathlib.Path(root).rglob('sd-cli'):
        if sdcli_works(str(p)):
            return str(p)
    return None


def run_live(cmd, cwd=None, timeout_min=10):
    print('\n$', ' '.join(cmd) if isinstance(cmd,list) else cmd, flush=True)
    start=time.time()
    p=subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=isinstance(cmd,str))
    try:
        while True:
            line=p.stdout.readline()
            if line:
                print(line.rstrip(), flush=True)
            if p.poll() is not None:
                rest=p.stdout.read()
                if rest: print(rest, end='', flush=True)
                break
            if time.time()-start > timeout_min*60:
                p.kill(); raise TimeoutError(f'Timeout après {timeout_min} min')
            time.sleep(.1)
        rc=p.wait(timeout=10)
    except Exception:
        try: p.kill()
        except Exception: pass
        raise
    if rc != 0:
        raise RuntimeError(f'Commande échouée ({rc}): {cmd}')
    return rc


def restore_runtime_cache():
    if not os.path.exists(SD_RUNTIME_CACHE_TAR):
        return None
    shutil.rmtree(SD_RUNTIME_DIR, ignore_errors=True)
    os.makedirs(SD_RUNTIME_DIR, exist_ok=True)
    with tarfile.open(SD_RUNTIME_CACHE_TAR, 'r:gz') as tf:
        tf.extractall(SD_RUNTIME_DIR)
    return find_sdcli(SD_RUNTIME_DIR)


def ensure_crane():
    crane='/content/crane'
    if os.path.isfile(crane) and os.access(crane, os.X_OK):
        return crane
    import requests
    url=f'https://github.com/google/go-containerregistry/releases/download/{CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz'
    archive='/content/go-containerregistry.tar.gz'
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(archive, 'wb') as f:
            for ch in r.iter_content(1024*1024):
                if ch: f.write(ch)
    with tarfile.open(archive, 'r:gz') as tf:
        member=next((m for m in tf.getmembers() if os.path.basename(m.name)=='crane'), None)
        if member is None:
            raise RuntimeError('crane absent de l’archive')
        member.name='crane'
        tf.extract(member, '/content')
    os.chmod(crane, 0o755)
    return crane


def pull_prebuilt_cuda_runtime():
    crane=ensure_crane()
    shutil.rmtree(SD_RUNTIME_DIR, ignore_errors=True)
    os.makedirs(SD_RUNTIME_DIR, exist_ok=True)
    p1=subprocess.Popen([crane, 'export', '--platform', 'linux/amd64', PREBUILT_IMAGE, '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2=subprocess.Popen(['tar', '-xf', '-', '-C', SD_RUNTIME_DIR, 'sd.cpp/bin'], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p1.stdout.close()
    _, err2 = p2.communicate(timeout=900)
    rc1 = p1.wait(timeout=30)
    if rc1 != 0 or p2.returncode != 0:
        err1 = p1.stderr.read().decode(errors='ignore') if p1.stderr else ''
        raise RuntimeError('Extraction image CUDA échouée: ' + ((err2 or err1)[-1500:]))
    return find_sdcli(SD_RUNTIME_DIR)


def cache_runtime_background(runtime_root):
    if not USE_DRIVE:
        return
    def worker():
        try:
            os.makedirs(os.path.dirname(SD_RUNTIME_CACHE_TAR), exist_ok=True)
            tmp = SD_RUNTIME_CACHE_TAR + '.part'
            with tarfile.open(tmp, 'w:gz', compresslevel=3) as tf:
                sd_dir=os.path.join(runtime_root, 'sd.cpp')
                if os.path.exists(sd_dir):
                    tf.add(sd_dir, arcname='sd.cpp')
                else:
                    b=os.path.join(runtime_root, 'bin')
                    if os.path.exists(b): tf.add(b, arcname='bin')
            os.replace(tmp, SD_RUNTIME_CACHE_TAR)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()


def compile_fallback():
    required={'git':'git','cmake':'cmake','ninja':'ninja-build','g++':'build-essential','ccache':'ccache'}
    missing=[pkg for exe,pkg in required.items() if shutil.which(exe) is None]
    if missing:
        run_live('apt-get -qq update && apt-get -qq install -y --no-install-recommends ' + ' '.join(missing), timeout_min=6)
    if not os.path.isdir(os.path.join(SDCPP_ROOT, '.git')):
        shutil.rmtree(SDCPP_ROOT, ignore_errors=True)
        run_live(['git','clone','--depth','1','--recurse-submodules','--shallow-submodules','--jobs','4','https://github.com/leejet/stable-diffusion.cpp',SDCPP_ROOT], timeout_min=7)
    arch=[]
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        cc=q.stdout.strip().splitlines()[0].replace('.','') if q.returncode==0 and q.stdout.strip() else ''
        if cc.isdigit(): arch=[f'-DCMAKE_CUDA_ARCHITECTURES={cc}']
    except Exception:
        pass
    configure=['cmake','-S','.', '-B','build','-G','Ninja','-DSD_CUDA=ON','-DCMAKE_BUILD_TYPE=Release','-DSD_WEBP=OFF','-DSD_WEBM=OFF','-DGGML_NATIVE=OFF','-DGGML_CUDA_FORCE_CUBLAS=ON','-DGGML_CUDA_FA=OFF','-DGGML_CUDA_FA_ALL_QUANTS=OFF'] + arch
    run_live(configure, cwd=SDCPP_ROOT, timeout_min=6)
    jobs=max(1, min(int(MAX_BUILD_JOBS), multiprocessing.cpu_count()))
    run_live(['cmake','--build','build','--target','sd-cli','--config','Release','-j', str(jobs)], cwd=SDCPP_ROOT, timeout_min=int(BUILD_TIMEOUT_MINUTES))
    return os.path.join(SDCPP_ROOT, 'build', 'bin', 'sd-cli')


def ensure_sdcli_ready(force=False):
    global SDCLI
    if SDCLI and sdcli_works(SDCLI) and not force:
        BOOTSTRAP_STATE['runtime_ready'] = True
        return SDCLI
    with SDCLI_LOCK:
        if SDCLI and sdcli_works(SDCLI) and not force:
            BOOTSTRAP_STATE['runtime_ready'] = True
            return SDCLI
        BOOTSTRAP_STATE['phase'] = 'runtime'
        BOOTSTRAP_STATE['details'] = 'Préparation du runtime sd-cli…'
        candidate = None
        if not force and sdcli_works(SDCLI_CACHE):
            candidate = SDCLI_CACHE
        if candidate is None and not force:
            try:
                candidate = restore_runtime_cache()
            except Exception:
                candidate = None
        if candidate is None and not force:
            try:
                candidate = pull_prebuilt_cuda_runtime()
                if candidate:
                    cache_runtime_background(SD_RUNTIME_DIR)
            except Exception:
                candidate = None
        if candidate is None:
            candidate = compile_fallback()
        if not sdcli_works(candidate):
            raise RuntimeError('sd-cli indisponible')
        SDCLI = candidate
        bindir = os.path.dirname(SDCLI)
        os.environ['LD_LIBRARY_PATH'] = bindir + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')
        BOOTSTRAP_STATE['runtime_ready'] = True
        BOOTSTRAP_STATE['details'] = 'Runtime sd-cli prêt.'
        return SDCLI


def _background_prepare_worker():
    BOOTSTRAP_STATE['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    BOOTSTRAP_STATE['phase'] = 'bootstrap'
    BOOTSTRAP_STATE['details'] = 'Préparation en arrière-plan…'
    BOOTSTRAP_STATE['last_error'] = ''
    try:
        ensure_core_models_ready(verbose=False)
        ensure_sdcli_ready(force=False)
        BOOTSTRAP_STATE['details'] = 'Préparation terminée.'
    except Exception as e:
        BOOTSTRAP_STATE['last_error'] = f'{type(e).__name__}: {e}'
        BOOTSTRAP_STATE['details'] = 'Préparation en erreur.'
    finally:
        BOOTSTRAP_STATE['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')


def start_background_prepare(force=False):
    global BOOTSTRAP_THREAD
    if BOOTSTRAP_THREAD is not None and BOOTSTRAP_THREAD.is_alive() and not force:
        return 'Préparation déjà en cours.'
    BOOTSTRAP_THREAD = threading.Thread(target=_background_prepare_worker, daemon=True)
    BOOTSTRAP_THREAD.start()
    return 'Préparation lancée en arrière-plan.'

if AUTO_START_BACKGROUND_PREPARE:
    try:
        start_background_prepare(force=False)
    except Exception:
        pass
print('⚡ Mode fast UX prêt : Gradio peut être lancé sans attendre la fin complète du bootstrap.')
