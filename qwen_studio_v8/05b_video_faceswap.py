# v8.4 — Face swap vidéo Ultra (tracking + bouche + checkpoints + progression)
import os, subprocess, shutil, uuid, time, json, signal, tarfile, hashlib, threading
from pathlib import Path
from fractions import Fraction

VIDEO_FACE_ROOT = os.path.join(ROOT, 'video_faceswap')
VIDEO_LOCAL_JOB_ROOT = os.path.join(ROOT, 'video_artifacts')
VIDEO_CHECKPOINT_ROOT = os.path.join(JOB_ROOT, 'video_checkpoints')
VIDEO_JOB_ROOT = VIDEO_LOCAL_JOB_ROOT
VIDEO_CANCEL_ROOT = os.path.join(JOB_ROOT, 'cancel_flags')
VIDEO_PAUSE_ROOT = os.path.join(JOB_ROOT, 'pause_flags')
os.makedirs(VIDEO_FACE_ROOT, exist_ok=True)
os.makedirs(VIDEO_LOCAL_JOB_ROOT, exist_ok=True)
os.makedirs(VIDEO_CHECKPOINT_ROOT, exist_ok=True)
os.makedirs(VIDEO_CANCEL_ROOT, exist_ok=True)
os.makedirs(VIDEO_PAUSE_ROOT, exist_ok=True)
VIDEO_ANALYSIS_CACHE = {}
FACEFUSION_CACHE_THREAD = None

class JobCancelled(Exception):
    pass

def _new_output_video(prefix='faceswap_video'):
    ts=time.strftime('%Y%m%d-%H%M%S')
    return os.path.join(OUTPUT_ROOT, f'{ts}_{prefix}_{uuid.uuid4().hex[:6]}.mp4')

def _video_job_dir(job_id):
    p=os.path.join(VIDEO_JOB_ROOT,str(job_id))
    os.makedirs(p,exist_ok=True)
    return p

def _video_checkpoint_dir(job_id):
    p=os.path.join(VIDEO_CHECKPOINT_ROOT,str(job_id))
    os.makedirs(p,exist_ok=True)
    return p

def _checkpoint_chunk_bounds(path):
    name=Path(path).stem
    try:
        _,start,end=name.split('_',2)
        return int(start),int(end)
    except Exception:
        return None

def _restore_video_checkpoints(job_id,frames_out):
    # v8.6: les chunks persistent sur Drive sans être tous ré-extraits localement.
    # On renvoie seulement la dernière frame persistée; l'encodage final sait
    # lire les archives chunk par chunk pour économiser fortement le disque.
    return _latest_checkpoint_end(job_id)

def _latest_checkpoint_end(job_id):
    if not VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS or not job_id:
        return 0
    checkpoint_dir=_video_checkpoint_dir(job_id)
    latest=0
    for archive in Path(checkpoint_dir).glob('chunk_*.tar'):
        if not _checkpoint_is_valid(archive):
            continue
        bounds=_checkpoint_chunk_bounds(archive)
        if bounds:
            latest=max(latest,bounds[1])
    return latest

def _sha256_file(path,chunk_size=4*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        while True:
            chunk=fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _checkpoint_manifest_path(archive):
    return str(archive)+'.json'

def _checkpoint_is_valid(archive):
    archive=str(archive)
    if not os.path.exists(archive) or os.path.getsize(archive)<=1024:
        return False
    manifest_path=_checkpoint_manifest_path(archive)
    try:
        if os.path.exists(manifest_path):
            data=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            if int(data.get('size',-1))!=os.path.getsize(archive):
                return False
            expected=data.get('sha256')
            if expected and _sha256_file(archive)!=expected:
                return False
        return tarfile.is_tarfile(archive)
    except Exception:
        return False

def _checkpoint_frames(job_id,frames_out,start_idx,end_idx):
    if not VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS or not job_id or end_idx<start_idx:
        return
    checkpoint_dir=_video_checkpoint_dir(job_id)
    dest=os.path.join(checkpoint_dir,f'chunk_{int(start_idx):08d}_{int(end_idx):08d}.tar')
    if _checkpoint_is_valid(dest):
        return
    files=[]
    for idx in range(int(start_idx),int(end_idx)+1):
        p=os.path.join(frames_out,f'{idx:08d}.png')
        if os.path.exists(p):
            files.append((idx,p))
    if not files:
        return

    # Construire l'archive sur le disque local rapide puis la copier vers Drive.
    local_tmp=os.path.join(
        _video_job_dir(job_id),
        f'.checkpoint_{int(start_idx):08d}_{int(end_idx):08d}_{uuid.uuid4().hex[:6]}.tar'
    )
    drive_tmp=dest+'.part'
    manifest_tmp=_checkpoint_manifest_path(dest)+'.part'
    try:
        with tarfile.open(local_tmp,'w') as tf:
            for idx,p in files:
                tf.add(p,arcname=f'{idx:08d}.png',recursive=False)
        digest=_sha256_file(local_tmp)
        size=os.path.getsize(local_tmp)
        shutil.copy2(local_tmp,drive_tmp)
        os.replace(drive_tmp,dest)
        manifest={
            'version':1,
            'job_id':str(job_id),
            'start_frame':int(start_idx),
            'end_frame':int(end_idx),
            'frame_count':len(files),
            'size':int(size),
            'sha256':digest,
            'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        Path(manifest_tmp).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(manifest_tmp,_checkpoint_manifest_path(dest))
        if VIDEO_FACE_SWAP_DELETE_CHECKPOINTED_FRAMES:
            for _,p in files:
                try:
                    os.remove(p)
                except Exception:
                    pass
    finally:
        for tmp in (local_tmp,drive_tmp,manifest_tmp):
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

def _checkpoint_unsaved_frames(job_id,frames_out):
    if not VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS or not job_id:
        return
    latest=_latest_checkpoint_end(job_id)
    existing=sorted(Path(frames_out).glob('*.png'))
    indices=[]
    for p in existing:
        try:
            indices.append(int(p.stem))
        except Exception:
            pass
    indices=[i for i in indices if i>latest]
    if indices:
        _checkpoint_frames(job_id,frames_out,min(indices),max(indices))

def _clear_persistent_video_checkpoints(job_id):
    if not job_id:
        return
    shutil.rmtree(os.path.join(VIDEO_CHECKPOINT_ROOT,str(job_id)),ignore_errors=True)

def _cancel_flag(job_id):
    return os.path.join(VIDEO_CANCEL_ROOT,f'{job_id}.flag')

def cancel_job(job_id):
    if not job_id:
        return 'Job ID vide'
    with open(_cancel_flag(job_id),'w',encoding='utf-8') as fh:
        fh.write('cancel')
    j=load_job(job_id)
    if j:
        j['info']=((j.get('info') or '')+'\nAnnulation demandée…').strip()
        save_job(j)
    return f'Annulation demandée pour {job_id}'

def _pause_flag(job_id):
    return os.path.join(VIDEO_PAUSE_ROOT,f'{job_id}.flag')

def pause_job(job_id):
    if not VIDEO_FACE_SWAP_PAUSE_ENABLED:
        return 'Pause désactivée dans la configuration.'
    if not job_id:
        return 'Job ID vide'
    with open(_pause_flag(job_id),'w',encoding='utf-8') as fh:
        fh.write('pause')
    j=load_job(job_id)
    if j and j.get('status') in ('running','queued'):
        j['status']='paused'
        j['info']=((j.get('info') or '')+'\nPause demandée…').strip()
        save_job(j)
    return f'Pause demandée pour {job_id}'

def resume_job(job_id):
    if not job_id:
        return 'Job ID vide'
    try:
        os.remove(_pause_flag(job_id))
    except Exception:
        pass
    j=load_job(job_id)
    if j and j.get('status')=='paused':
        j['status']='running'
        j['info']=((j.get('info') or '')+'\nReprise demandée…').strip()
        save_job(j)
    return f'Reprise demandée pour {job_id}'

def _is_pause_requested(job_id):
    return bool(job_id) and os.path.exists(_pause_flag(job_id))

def _wait_if_paused(job_id, process=None):
    if not VIDEO_FACE_SWAP_PAUSE_ENABLED or not job_id:
        return
    stopped=False
    while _is_pause_requested(job_id):
        if _is_cancel_requested(job_id):
            if process is not None:
                try: process.terminate()
                except Exception: pass
            raise JobCancelled('Annulation demandée pendant la pause.')
        if process is not None and not stopped:
            try:
                os.kill(process.pid, signal.SIGSTOP)
                stopped=True
            except Exception:
                stopped=False
        j=load_job(job_id)
        if j:
            j['status']='paused'
            j['info']='Job en pause — utilise Reprendre pour continuer.'
            save_job(j)
        time.sleep(0.4)
    if process is not None and stopped:
        try: os.kill(process.pid, signal.SIGCONT)
        except Exception: pass
    j=load_job(job_id)
    if j and j.get('status')=='paused':
        j['status']='running'
        save_job(j)

def _is_cancel_requested(job_id):
    return bool(job_id) and os.path.exists(_cancel_flag(job_id))

def _clear_cancel(job_id):
    try:
        os.remove(_cancel_flag(job_id))
    except Exception:
        pass

def _update_job_progress(job_id,current,total,stage='traitement',extra=''):
    if not job_id:
        return
    j=load_job(job_id)
    if not j:
        return
    current=int(current); total=max(0,int(total))
    pct=0.0 if total<=0 else round((current/total)*100.0,2)
    j['progress_current']=current
    j['progress_total']=total
    j['progress_pct']=pct
    msg=f'{stage}: {current}/{total} ({pct:.2f}%)'
    if extra:
        msg += f' | {extra}'
    j['info']=msg
    save_job(j)

def ensure_ffmpeg():
    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        return
    subprocess.run(
        'apt-get -qq update && apt-get -qq install -y --no-install-recommends ffmpeg',
        shell=True, check=True, timeout=600
    )

def _ffprobe_fps(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0',
         '-show_entries','stream=avg_frame_rate,r_frame_rate',
         '-of','json',video_path],
        capture_output=True,text=True
    )
    try:
        data=json.loads(p.stdout or '{}')
        streams=data.get('streams') or []
        stream=streams[0] if streams else {}
        for key in ('avg_frame_rate','r_frame_rate'):
            val=str(stream.get(key) or '')
            if val and val not in ('0/0','0'):
                fps=float(Fraction(val))
                if fps>0:
                    return fps
    except Exception:
        pass
    return 25.0

def _ffprobe_size(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height',
         '-of','csv=s=x:p=0',video_path],
        capture_output=True,text=True
    )
    try:
        w,h=(p.stdout or '').strip().split('x',1)
        return int(w),int(h)
    except Exception:
        return 0,0

def _ffprobe_duration(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ['ffprobe','-v','error','-show_entries','format=duration','-of','default=nokey=1:noprint_wrappers=1',video_path],
        capture_output=True,text=True
    )
    try:
        return max(0.0,float((p.stdout or '').strip()))
    except Exception:
        return 0.0

def _face_bbox(face):
    return [float(v) for v in face.bbox[:4]]

def _face_area(face):
    x1,y1,x2,y2=_face_bbox(face)
    return max(1.0,(x2-x1)*(y2-y1))

def _face_center(face):
    x1,y1,x2,y2=_face_bbox(face)
    return ((x1+x2)/2.0,(y1+y2)/2.0)

def _bbox_iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih
    aa=max(1.0,(ax2-ax1)*(ay2-ay1))
    ba=max(1.0,(bx2-bx1)*(by2-by1))
    return inter/max(1.0,aa+ba-inter)

def _cosine(a,b):
    import numpy as np
    if a is None or b is None:
        return 0.0
    a=np.asarray(a,dtype='float32').reshape(-1)
    b=np.asarray(b,dtype='float32').reshape(-1)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<1e-8 or nb<1e-8:
        return 0.0
    return float(np.dot(a,b)/(na*nb))

def _sig(face):
    return {
        'bbox':_face_bbox(face),
        'area':_face_area(face),
        'center':_face_center(face),
        'embedding':getattr(face,'normed_embedding',None),
        'det_score':float(getattr(face,'det_score',0.0) or 0.0),
        'kps':getattr(face,'kps',None),
    }

def _select_face(faces,prev=None):
    if not faces:
        return None,None
    if prev is None or not VIDEO_FACE_SWAP_STRICT_TRACKING:
        f=_largest(faces)
        return f,_sig(f)
    best=None; best_score=-1e9
    px,py=prev['center']
    for f in faces:
        s=_sig(f)
        cx,cy=s['center']
        drift=((cx-px)**2+(cy-py)**2)**0.5
        iou=_bbox_iou(s['bbox'],prev['bbox'])
        emb=_cosine(s['embedding'],prev['embedding'])
        score=s['det_score']*2.0+iou*3.7+emb*3.2-drift*0.002
        if score>best_score:
            best_score=score
            best=(f,s)
    return best if best else (None,None)

def _unstable(sig,prev):
    if sig is None:
        return True
    if sig['det_score'] < float(VIDEO_FACE_SWAP_MIN_DET_SCORE):
        return True
    if prev is None:
        return False
    ratio=sig['area']/max(1.0,prev['area'])
    if ratio<float(VIDEO_FACE_SWAP_AREA_RATIO_MIN) or ratio>float(VIDEO_FACE_SWAP_AREA_RATIO_MAX):
        return True
    iou=_bbox_iou(sig['bbox'],prev['bbox'])
    emb=_cosine(sig['embedding'],prev['embedding'])
    if iou<float(VIDEO_FACE_SWAP_MIN_IOU) and emb<float(VIDEO_FACE_SWAP_MIN_EMBED_SIM):
        return True
    return False

def _mouth_preserve(original,swapped,sig):
    import numpy as np, cv2
    if not VIDEO_FACE_SWAP_PRESERVE_MOUTH:
        return swapped
    kps=sig.get('kps')
    if kps is None:
        return swapped
    try:
        pts=np.asarray(kps,dtype=np.float32).reshape(-1,2)
        if len(pts)<5:
            return swapped
        m1,m2=pts[3],pts[4]
        cx=int((m1[0]+m2[0])/2.0)
        cy=int((m1[1]+m2[1])/2.0)
        x1,y1,x2,y2=[int(v) for v in sig['bbox']]
        fw=max(8,x2-x1); fh=max(8,y2-y1)
        rx=max(8,int(fw*0.23)); ry=max(6,int(fh*0.16))
        mask=np.zeros(original.shape[:2],dtype=np.uint8)
        cv2.ellipse(mask,(cx,cy),(rx,ry),0,0,360,255,-1)
        blur=max(7,int(min(rx,ry)*0.75))
        if blur%2==0: blur+=1
        mask=cv2.GaussianBlur(mask,(blur,blur),0).astype('float32')/255.0
        alpha=float(VIDEO_FACE_SWAP_MOUTH_BLEND)
        mask=np.clip(mask*alpha,0.0,1.0)[...,None]
        out=(swapped.astype('float32')*(1.0-mask)+original.astype('float32')*mask)
        return np.clip(out,0,255).astype('uint8')
    except Exception:
        return swapped

def _post_blend_face(original,swapped,sig):
    import numpy as np, cv2
    result=swapped
    try:
        if VIDEO_FACE_SWAP_USE_SEAMLESS_BLEND:
            kps=sig.get('kps')
            if kps is not None:
                hull=cv2.convexHull(np.asarray(kps,dtype=np.float32).reshape(-1,1,2))
                mask=np.zeros(original.shape[:2],dtype=np.uint8)
                cv2.fillConvexPoly(mask,hull.astype(np.int32),255)
                x1,y1,x2,y2=[int(v) for v in sig['bbox']]
                pad=max(8,int(max(x2-x1,y2-y1)*0.10))+int(VIDEO_FACE_SWAP_MASK_DILATE)
                x1=max(0,x1-pad); y1=max(0,y1-pad)
                x2=min(mask.shape[1]-1,x2+pad); y2=min(mask.shape[0]-1,y2+pad)
                cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)
                k=max(3,int(VIDEO_FACE_SWAP_MASK_BLUR))
                if k%2==0: k+=1
                mask=cv2.GaussianBlur(mask,(k,k),0)
                center=(int((x1+x2)/2),int((y1+y2)/2))
                result=cv2.seamlessClone(swapped,original,mask,center,cv2.NORMAL_CLONE)
    except Exception:
        result=swapped
    return _mouth_preserve(original,result,sig)

def _facefusion_python():
    return os.path.join(VIDEO_FACEFUSION_VENV,'bin','python')

def _facefusion_ready():
    return (
        os.path.isfile(os.path.join(VIDEO_FACEFUSION_ROOT,'facefusion.py')) and
        os.path.isfile(_facefusion_python())
    )

def _restore_facefusion_cache():
    if not VIDEO_FACEFUSION_CACHE_ENABLED or not USE_DRIVE:
        return False
    if not os.path.exists(VIDEO_FACEFUSION_CACHE_TAR) or os.path.getsize(VIDEO_FACEFUSION_CACHE_TAR)<1024*1024:
        return False
    try:
        shutil.rmtree(VIDEO_FACEFUSION_ROOT,ignore_errors=True)
        shutil.rmtree(VIDEO_FACEFUSION_VENV,ignore_errors=True)
        with tarfile.open(VIDEO_FACEFUSION_CACHE_TAR,'r:gz') as tf:
            tf.extractall('/content')
        return _facefusion_ready()
    except Exception:
        shutil.rmtree(VIDEO_FACEFUSION_ROOT,ignore_errors=True)
        shutil.rmtree(VIDEO_FACEFUSION_VENV,ignore_errors=True)
        return False

def _cache_facefusion_worker():
    try:
        if not _facefusion_ready() or not USE_DRIVE:
            return
        os.makedirs(VIDEO_FACEFUSION_CACHE_DIR,exist_ok=True)
        local_tmp=f'/content/facefusion_cache_{uuid.uuid4().hex[:8]}.tar.gz'
        drive_tmp=VIDEO_FACEFUSION_CACHE_TAR+'.part'
        try:
            with tarfile.open(local_tmp,'w:gz',compresslevel=1) as tf:
                tf.add(VIDEO_FACEFUSION_ROOT,arcname=os.path.basename(VIDEO_FACEFUSION_ROOT))
                tf.add(VIDEO_FACEFUSION_VENV,arcname=os.path.basename(VIDEO_FACEFUSION_VENV))
            shutil.copy2(local_tmp,drive_tmp)
            os.replace(drive_tmp,VIDEO_FACEFUSION_CACHE_TAR)
        finally:
            for p in (local_tmp,drive_tmp):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass

def cache_facefusion_background():
    global FACEFUSION_CACHE_THREAD
    if not VIDEO_FACEFUSION_CACHE_ENABLED or not USE_DRIVE:
        return
    if FACEFUSION_CACHE_THREAD is not None and FACEFUSION_CACHE_THREAD.is_alive():
        return
    FACEFUSION_CACHE_THREAD=threading.Thread(target=_cache_facefusion_worker,daemon=True)
    FACEFUSION_CACHE_THREAD.start()

def facefusion_status_text():
    ready=_facefusion_ready()
    cache=os.path.exists(VIDEO_FACEFUSION_CACHE_TAR) if VIDEO_FACEFUSION_CACHE_ENABLED else False
    cache_size=(os.path.getsize(VIDEO_FACEFUSION_CACHE_TAR)/1024**3) if cache else 0.0
    return (
        f'FaceFusion local: {"prêt" if ready else "non installé"}\n'
        f'Cache Drive: {"présent" if cache else "absent"}'
        + (f' ({cache_size:.2f} Go)' if cache else '')
    )

def ensure_facefusion_ultra():
    if not VIDEO_FACEFUSION_ENABLED:
        raise RuntimeError('FaceFusion Ultra est désactivé dans la configuration.')
    if _facefusion_ready():
        return
    if _restore_facefusion_cache():
        return
    if not shutil.which('git'):
        subprocess.run('apt-get -qq update && apt-get -qq install -y --no-install-recommends git',shell=True,check=True,timeout=600)
    if not os.path.isdir(os.path.join(VIDEO_FACEFUSION_ROOT,'.git')):
        shutil.rmtree(VIDEO_FACEFUSION_ROOT,ignore_errors=True)
        subprocess.run([
            'git','clone','--depth','1','--branch',str(VIDEO_FACEFUSION_VERSION),
            'https://github.com/facefusion/facefusion.git',VIDEO_FACEFUSION_ROOT
        ],check=True,timeout=900)
    if not os.path.isfile(_facefusion_python()):
        subprocess.run(['python','-m','venv',VIDEO_FACEFUSION_VENV],check=True,timeout=300)
    venv_bin=os.path.join(VIDEO_FACEFUSION_VENV,'bin')
    env=os.environ.copy()
    env['PATH']=venv_bin+os.pathsep+env.get('PATH','')
    subprocess.run([_facefusion_python(),'-m','pip','install','-U','pip'],check=True,timeout=600,env=env)
    subprocess.run(
        [_facefusion_python(),'install.py','cuda@12','--skip-conda'],
        cwd=VIDEO_FACEFUSION_ROOT,check=True,timeout=1800,env=env
    )

def run_facefusion_ultra(source_image,target_video,keep_audio=True,preview_seconds=0,job_id=None):
    ensure_ffmpeg()
    ensure_facefusion_ultra()
    out=_new_output_video('facefusion_ultra_preview' if int(preview_seconds or 0)>0 else 'facefusion_ultra')
    py=_facefusion_python()
    processors=['face_swapper']
    if VIDEO_FACEFUSION_EXPRESSION_RESTORER:
        processors.append('expression_restorer')
    if VIDEO_FACEFUSION_FACE_ENHANCER:
        processors.append('face_enhancer')
    cmd=[
        py,'facefusion.py','headless-run',
        '--workflow-strategy','disk',
        '--processors',*processors,
        '--face-mask-types',*str(VIDEO_FACEFUSION_MASK_TYPES).split(),
        '--face-enhancer-model','gfpgan_1.4',
        '--face-enhancer-blend',str(int(VIDEO_FACEFUSION_ENHANCER_BLEND)),
        '--expression-restorer-model','live_portrait',
        '--expression-restorer-factor',str(int(VIDEO_FACEFUSION_EXPRESSION_FACTOR)),
        '--execution-providers','cuda',
        '-s',source_image,
        '-t',target_video,
        '-o',out,
    ]
    if int(preview_seconds or 0)>0:
        fps=_ffprobe_fps(target_video)
        trim_end=max(1,int(round(float(preview_seconds)*fps)))
        cmd += ['--trim-frame-end',str(trim_end)]
    _update_job_progress(job_id,0,1,'FaceFusion Ultra','initialisation / modèles')
    env=os.environ.copy()
    env['PATH']=os.path.join(VIDEO_FACEFUSION_VENV,'bin')+os.pathsep+env.get('PATH','')
    proc=subprocess.Popen(
        cmd,cwd=VIDEO_FACEFUSION_ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,env=env
    )
    tail=[]
    import select
    while True:
        _wait_if_paused(job_id,proc)
        if _is_cancel_requested(job_id):
            proc.terminate()
            try: proc.wait(timeout=10)
            except Exception: proc.kill()
            raise JobCancelled('Annulation demandée pendant FaceFusion Ultra.')
        line=''
        if proc.stdout:
            ready,_,_=select.select([proc.stdout],[],[],0.25)
            if ready:
                line=proc.stdout.readline()
        if line:
            tail.append(line.rstrip())
            tail=tail[-60:]
            j=load_job(job_id) if job_id else None
            if j:
                j['info']='FaceFusion Ultra en cours…\n'+('\n'.join(tail[-5:]))
                save_job(j)
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    rc=proc.wait()
    if rc!=0 or not os.path.exists(out):
        raise RuntimeError('FaceFusion Ultra a échoué:\n'+'\n'.join(tail[-25:]))
    _update_job_progress(job_id,1,1,'FaceFusion Ultra','terminé')
    cache_facefusion_background()
    return out, f'FaceFusion Ultra terminé | processors={processors} | masks={VIDEO_FACEFUSION_MASK_TYPES}\n{out}'

def analyze_video_difficulty(video_path):
    import cv2, math
    if not video_path:
        return 'Ajoute une vidéo à analyser.', 'builtin-ultra'
    try:
        key=(str(video_path),os.path.getsize(str(video_path)),int(os.path.getmtime(str(video_path))))
        if key in VIDEO_ANALYSIS_CACHE:
            return VIDEO_ANALYSIS_CACHE[key]
    except Exception:
        key=None
    init_faceswap()
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 'Vidéo illisible.', 'builtin-ultra'
    frame_count=max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    samples=max(4,min(32,int(VIDEO_FACE_SWAP_ANALYSIS_SAMPLE_FRAMES or 12)))
    positions=[int(i*(frame_count-1)/max(1,samples-1)) for i in range(samples)]
    detected=multi=0
    scores=[]; area_ratios=[]; motions=[]
    prev_center=None
    width=max(1.0,float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1))
    height=max(1.0,float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1))
    diag=max(1.0,math.hypot(width,height))
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        ok,frame=cap.read()
        if not ok or frame is None:
            continue
        try:
            faces=FS_APP.get(frame)
        except Exception:
            faces=[]
        if not faces:
            continue
        detected+=1
        if len(faces)>1:
            multi+=1
        face=_largest(faces)
        sig=_sig(face)
        scores.append(sig['det_score'])
        area_ratios.append(sig['area']/max(1.0,width*height))
        if prev_center is not None:
            dx=sig['center'][0]-prev_center[0]
            dy=sig['center'][1]-prev_center[1]
            motions.append(math.hypot(dx,dy)/diag)
        prev_center=sig['center']
    cap.release()
    coverage=detected/max(1,len(positions))
    multi_ratio=multi/max(1,detected)
    avg_score=sum(scores)/max(1,len(scores))
    avg_area=sum(area_ratios)/max(1,len(area_ratios))
    avg_motion=sum(motions)/max(1,len(motions))
    risk=0
    reasons=[]
    if coverage<0.80: risk+=3; reasons.append('visage souvent non détecté')
    elif coverage<0.95: risk+=1; reasons.append('détection parfois instable')
    if multi_ratio>0.20: risk+=2; reasons.append('plusieurs visages fréquents')
    if avg_score<0.60: risk+=2; reasons.append('score de détection faible')
    elif avg_score<float(VIDEO_FACE_SWAP_EASY_DET_SCORE): risk+=1; reasons.append('score de détection moyen')
    if avg_motion>0.08: risk+=2; reasons.append('mouvements importants')
    elif avg_motion>0.04: risk+=1; reasons.append('mouvements modérés')
    if avg_area<0.018: risk+=2; reasons.append('visage petit dans l’image')
    recommendation='facefusion-ultra' if risk>=4 else 'builtin-ultra'
    level='difficile' if risk>=4 else ('moyenne' if risk>=2 else 'facile')
    summary=(
        f'Analyse vidéo: difficulté {level} | couverture visage={coverage*100:.0f}% | '
        f'plusieurs visages={multi_ratio*100:.0f}% | score moyen={avg_score:.2f} | '
        f'mouvement moyen={avg_motion:.3f} | surface visage={avg_area*100:.2f}%\n'
        f'Recommandation: {recommendation}'
    )
    if reasons:
        summary += '\nFacteurs: ' + ', '.join(reasons)
    result=(summary,recommendation)
    if key is not None:
        VIDEO_ANALYSIS_CACHE[key]=result
    return result

def cleanup_video_cache(max_age_hours=None,keep_recent=None):
    max_age=float(max_age_hours if max_age_hours is not None else VIDEO_FACE_SWAP_CACHE_MAX_AGE_HOURS)
    keep=int(keep_recent if keep_recent is not None else VIDEO_FACE_SWAP_CACHE_KEEP_RECENT)
    active={str(j.get('id')) for j in list_jobs(limit=500) if j.get('status') in ('queued','running','paused')}
    now=time.time()
    removed=0
    for root in (VIDEO_LOCAL_JOB_ROOT,VIDEO_CHECKPOINT_ROOT):
        dirs=[p for p in Path(root).iterdir() if p.is_dir()] if os.path.isdir(root) else []
        dirs.sort(key=lambda p:p.stat().st_mtime,reverse=True)
        for idx,p in enumerate(dirs):
            if idx<keep or p.name in active:
                continue
            age_h=(now-p.stat().st_mtime)/3600.0
            if age_h>=max_age:
                shutil.rmtree(p,ignore_errors=True)
                removed+=1
    return f'Nettoyage terminé: {removed} dossier(s) vidéo supprimé(s).'

def video_selftest_report():
    tests=[]
    try:
        tests.append(('IoU identique',abs(_bbox_iou([0,0,10,10],[0,0,10,10])-1.0)<1e-6))
        tests.append(('IoU séparé',_bbox_iou([0,0,10,10],[20,20,30,30])==0.0))
        tests.append(('Dossier local vidéo',os.path.isdir(VIDEO_LOCAL_JOB_ROOT)))
        tests.append(('Dossier checkpoints persistants',os.path.isdir(VIDEO_CHECKPOINT_ROOT)))
        tests.append(('Checksum checkpoints',callable(_checkpoint_is_valid)))
        tests.append(('FFmpeg ou installation lazy',bool(shutil.which('ffmpeg')) or True))
        tests.append(('FaceFusion config',bool(VIDEO_FACEFUSION_VERSION)))
    except Exception:
        tests.append(('Self-test interne',False))
    ok=sum(1 for _,v in tests if v)
    lines=[f'{name}: '+('OK' if value else 'ERREUR') for name,value in tests]
    return f'Self-test vidéo: {ok}/{len(tests)} OK\n'+'\n'.join(lines)

def _resolve_video_backend(target_video,requested_backend='auto',job_id=None):
    requested=str(requested_backend or 'auto').lower()
    if requested in ('facefusion','facefusion-ultra','facefusion ultra'):
        return 'facefusion-ultra'
    if requested in ('builtin','builtin-ultra','builtin ultra'):
        return 'builtin-ultra'
    if not VIDEO_FACEFUSION_ENABLED:
        return 'builtin-ultra'
    try:
        summary,recommended=analyze_video_difficulty(target_video)
        j=load_job(job_id) if job_id else None
        if j:
            j['analysis']=summary
            j['info']='Analyse automatique terminée.\n'+summary
            save_job(j)
        if recommended=='facefusion-ultra':
            if _facefusion_ready() or VIDEO_FACEFUSION_AUTO_INSTALL_ON_HARD:
                return 'facefusion-ultra'
        return 'builtin-ultra'
    except Exception as e:
        j=load_job(job_id) if job_id else None
        if j:
            j['info']=f'Analyse automatique indisponible: {type(e).__name__}: {e}\nFallback builtin-ultra.'
            save_job(j)
        return 'builtin-ultra'

def _choose_mode(video_path):
    if not VIDEO_FACE_SWAP_AUTO_MODE:
        return 'ultra'
    duration=_ffprobe_duration(video_path)
    return 'fast' if duration>0 and duration<8.0 else 'ultra'

def _ensure_video_disk_headroom(width,height,chunk_frames):
    free=shutil.disk_usage(ROOT).free
    floor=int(VIDEO_FACE_SWAP_MIN_FREE_DISK_GB)*1024**3
    # marge conservatrice pour les PNG du chunk + encodage temporaire.
    estimated=max(512*1024**2,int(max(1,width)*max(1,height)*3*max(1,chunk_frames)*1.35))
    required=max(floor,estimated)
    if free<required:
        raise RuntimeError(
            f'Espace disque local insuffisant: {free/1024**3:.1f} Go libres; '
            f'environ {required/1024**3:.1f} Go requis pour ce chunk.'
        )

def _encode_checkpoint_chunks(job_id,fps,target_video,out,keep_audio,crf,preset,preview_seconds=0,max_frame=0):
    checkpoint_dir=_video_checkpoint_dir(job_id)
    archives=sorted(
        [p for p in Path(checkpoint_dir).glob('chunk_*.tar') if _checkpoint_chunk_bounds(p) and _checkpoint_is_valid(p)],
        key=lambda p:_checkpoint_chunk_bounds(p)[0]
    )
    if not archives:
        raise RuntimeError('Aucun checkpoint vidéo disponible pour l’encodage final.')
    encode_root=os.path.join(_video_job_dir(job_id),'encode_chunks')
    shutil.rmtree(encode_root,ignore_errors=True)
    os.makedirs(encode_root,exist_ok=True)
    segment_paths=[]
    try:
        selected=[]
        limit=max(0,int(max_frame or 0))
        for archive in archives:
            start,end=_checkpoint_chunk_bounds(archive)
            if limit and start>limit:
                break
            selected.append((archive,start,min(end,limit) if limit else end))
        if not selected:
            raise RuntimeError('Aucun checkpoint dans la plage demandée.')
        for seg_idx,(archive,start,end) in enumerate(selected,1):
            seg_frames=os.path.join(encode_root,f'frames_{seg_idx:04d}')
            os.makedirs(seg_frames,exist_ok=True)
            with tarfile.open(archive,'r') as tf:
                tf.extractall(seg_frames)
            count=max(1,end-start+1)
            seg_out=os.path.join(encode_root,f'segment_{seg_idx:04d}.mp4')
            cmd=[
                'ffmpeg','-y','-framerate',f'{fps:.6f}',
                '-start_number',str(start),
                '-i',os.path.join(seg_frames,'%08d.png'),
                '-frames:v',str(count),
                '-c:v','libx264','-preset',preset,'-crf',str(crf),
                '-pix_fmt','yuv420p',seg_out
            ]
            p=subprocess.run(cmd,capture_output=True,text=True)
            if p.returncode!=0:
                raise RuntimeError('Encodage chunk échoué:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-3000:])
            segment_paths.append(seg_out)
            shutil.rmtree(seg_frames,ignore_errors=True)
        concat_file=os.path.join(encode_root,'segments.txt')
        with open(concat_file,'w',encoding='utf-8') as fh:
            for seg in segment_paths:
                escaped=seg.replace("'","'\\''")
                fh.write(f"file '{escaped}'\n")
        video_only=os.path.join(encode_root,'video_only.mp4')
        p=subprocess.run(
            ['ffmpeg','-y','-f','concat','-safe','0','-i',concat_file,'-c','copy',video_only],
            capture_output=True,text=True
        )
        if p.returncode!=0:
            raise RuntimeError('Concaténation vidéo échouée:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-3000:])
        if keep_audio and int(preview_seconds or 0)<=0:
            mux_copy=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
                      '-c:v','copy','-c:a','copy','-shortest',out]
            p=subprocess.run(mux_copy,capture_output=True,text=True)
            if p.returncode!=0:
                mux_aac=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
                         '-c:v','copy','-c:a','aac','-b:a','192k','-shortest',out]
                p=subprocess.run(mux_aac,capture_output=True,text=True)
                if p.returncode!=0:
                    shutil.copy2(video_only,out)
        else:
            shutil.copy2(video_only,out)
    finally:
        shutil.rmtree(encode_root,ignore_errors=True)

def _encode_frames(frames_out,fps,target_video,out,keep_audio,crf,preset,preview_seconds=0):
    video_only=os.path.join(os.path.dirname(frames_out),'video_only.mp4')
    enc=['ffmpeg','-y','-framerate',f'{fps:.6f}','-i',os.path.join(frames_out,'%08d.png'),
         '-c:v','libx264','-preset',preset,'-crf',str(crf),'-pix_fmt','yuv420p',video_only]
    p=subprocess.run(enc,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError('Encodage vidéo échoué:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-4000:])
    if keep_audio and int(preview_seconds or 0)<=0:
        mux_copy=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
                  '-c:v','copy','-c:a','copy','-shortest',out]
        p=subprocess.run(mux_copy,capture_output=True,text=True)
        if p.returncode!=0:
            mux_aac=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
                     '-c:v','copy','-c:a','aac','-b:a','192k','-shortest',out]
            p=subprocess.run(mux_aac,capture_output=True,text=True)
            if p.returncode!=0:
                shutil.copy2(video_only,out)
    else:
        shutil.copy2(video_only,out)

def video_face_swap(source_image,target_video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1,preview_seconds=0,job_id=None,backend='auto'):
    import cv2, math
    ensure_ffmpeg()
    frame_stride=max(1,int(frame_stride or 1))
    detect_every=max(1,int(detect_every or 1))
    max_frames=max(0,int(max_frames or 0))
    preview_seconds=max(0,int(preview_seconds or 0))
    crf=max(10,min(30,int(crf or 17)))
    preset=str(preset or 'slow')

    resolved_backend=_resolve_video_backend(target_video,backend,job_id)
    if resolved_backend=='facefusion-ultra':
        return run_facefusion_ultra(source_image,target_video,keep_audio,preview_seconds,job_id)

    init_faceswap()
    mode=_choose_mode(target_video)
    if mode=='ultra':
        detect_every=1
        crf=min(crf,int(VIDEO_FACE_SWAP_HARD_MODE_CRF))

    source=cv2.imread(source_image)
    if source is None:
        raise RuntimeError('Image source illisible')
    source_faces=FS_APP.get(source)
    source_face=_largest(source_faces)

    work=_video_job_dir(job_id or ('video_'+uuid.uuid4().hex[:8]))
    frames_out=os.path.join(work,'frames_out')
    os.makedirs(frames_out,exist_ok=True)

    source_fps=_ffprobe_fps(target_video)
    output_fps=max(1.0,source_fps/float(frame_stride))
    width,height=_ffprobe_size(target_video)
    checkpoint_every=max(1,int(VIDEO_FACE_SWAP_CHECKPOINT_EVERY or 120))
    _ensure_video_disk_headroom(width,height,checkpoint_every)

    cap=cv2.VideoCapture(str(target_video))
    if not cap.isOpened():
        raise RuntimeError('Impossible d’ouvrir la vidéo cible pour le traitement streaming.')
    source_frame_count=max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    if source_frame_count>0:
        total=max(1,int(math.ceil(source_frame_count/float(frame_stride))))
    else:
        duration=_ffprobe_duration(target_video)
        total=max(1,int(math.ceil(duration*source_fps/float(frame_stride)))) if duration>0 else 1
    if preview_seconds>0:
        total=min(total,max(1,int(round(output_fps*preview_seconds))))
    if max_frames>0:
        total=min(total,max_frames)

    persisted_end=min(total,_latest_checkpoint_end(job_id))
    local_indices=[]
    for pth in Path(frames_out).glob('*.png'):
        try:
            idx=int(pth.stem)
            if idx<=total:
                local_indices.append(idx)
        except Exception:
            pass
    local_end=max(local_indices) if local_indices else 0
    resume_end=max(persisted_end,local_end)
    restored_checkpoint_end=persisted_end

    # Reprendre directement près de la première frame non calculée.
    source_start=max(0,resume_end*frame_stride)
    if source_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES,source_start)

    selected_idx=resume_end
    source_idx=source_start
    prev=None
    stable_after_occlusion=0
    processed=resume_end
    swapped=0
    skipped=0
    resumed=resume_end
    started=time.time()
    progress_every=max(1,int(VIDEO_FACE_SWAP_PROGRESS_EVERY or 8))
    last_checkpoint=persisted_end

    if resume_end:
        _update_job_progress(job_id,resume_end,total,'reprise',f'{resume_end} frame(s) déjà calculée(s)')

    try:
        while selected_idx<total:
            ok,frame=cap.read()
            if not ok or frame is None:
                break
            current_source_idx=source_idx
            source_idx+=1
            if current_source_idx % frame_stride != 0:
                continue

            idx=selected_idx+1
            selected_idx=idx

            if _is_pause_requested(job_id):
                _checkpoint_unsaved_frames(job_id,frames_out)
            _wait_if_paused(job_id)
            if _is_cancel_requested(job_id):
                _checkpoint_unsaved_frames(job_id,frames_out)
                raise JobCancelled('Annulation demandée par l’utilisateur.')

            out_frame=os.path.join(frames_out,f'{idx:08d}.png')
            if os.path.exists(out_frame) and os.path.getsize(out_frame)>1000:
                processed=max(processed,idx)
                resumed+=1
                continue

            original=frame.copy()
            try:
                faces=FS_APP.get(frame)
            except Exception:
                faces=[]

            face,sig=_select_face(faces,prev) if faces else (None,None)
            bad=(face is None or sig is None)
            if not bad and VIDEO_FACE_SWAP_SKIP_ON_OCCLUSION:
                bad=_unstable(sig,prev)

            if bad:
                stable_after_occlusion=0
                result=original
                skipped+=1
            else:
                stable_after_occlusion+=1
                required=max(1,int(VIDEO_FACE_SWAP_RECOVERY_STABLE_FRAMES or 1))
                if prev is not None and stable_after_occlusion<required:
                    result=original
                    skipped+=1
                else:
                    try:
                        swapped_frame=FS_SWAPPER.get(frame.copy(),face,source_face,paste_back=True)
                        result=_post_blend_face(original,swapped_frame,sig)
                        prev=sig
                        swapped+=1
                    except Exception:
                        result=original
                        skipped+=1
                        stable_after_occlusion=0

            cv2.imwrite(out_frame,result)
            processed=max(processed,idx)

            if VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS and (idx-last_checkpoint)>=checkpoint_every:
                _checkpoint_frames(job_id,frames_out,last_checkpoint+1,idx)
                last_checkpoint=idx

            if idx%progress_every==0 or idx==total:
                elapsed=max(0.001,time.time()-started)
                newly_done=max(1,idx-resume_end)
                speed=newly_done/elapsed
                eta=max(0,int((total-idx)/max(speed,1e-6)))
                _update_job_progress(
                    job_id,idx,total,'traitement vidéo',
                    f'ETA ~ {eta}s | backend={resolved_backend} | mode={mode} | swap={swapped} | protégées={skipped}'
                )
    finally:
        cap.release()

    if selected_idx<total:
        total=selected_idx
    if total<=0:
        raise RuntimeError('Aucune frame vidéo n’a pu être traitée.')

    if VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS and total>last_checkpoint:
        _checkpoint_frames(job_id,frames_out,last_checkpoint+1,total)
        last_checkpoint=total

    out=_new_output_video('faceswap_video_preview' if preview_seconds>0 else 'faceswap_video')
    if VIDEO_FACE_SWAP_PERSIST_CHECKPOINTS:
        _encode_checkpoint_chunks(
            job_id,output_fps,target_video,out,keep_audio,crf,preset,
            preview_seconds,max_frame=total
        )
    else:
        _encode_frames(frames_out,output_fps,target_video,out,keep_audio,crf,preset,preview_seconds)

    if VIDEO_FACE_SWAP_CLEANUP_TEMP and preview_seconds<=0:
        shutil.rmtree(frames_out,ignore_errors=True)
        _clear_persistent_video_checkpoints(job_id)

    info=(
        f'Face swap vidéo terminé | backend={resolved_backend} | mode={mode} | {width}x{height} | '
        f'fps_source={source_fps:.3f} | fps_sortie={output_fps:.3f} | frames={total} | '
        f'swap_nouveaux={swapped} | protégées_nouvelles={skipped} | reprise={resume_end} | '
        f'checkpoint_restauré={restored_checkpoint_end} | stride={frame_stride} | '
        f'detect_every={detect_every} | CRF={crf} | preset={preset} | preview={preview_seconds}s\n{out}'
    )
    return out,info


def submit_faceswap_video_job(src,video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1,preview_seconds=0,backend='auto',preflight_seconds=0):
    if not src or not video:
        raise ValueError('Ajoute un visage source et une vidéo cible')
    temp=uuid.uuid4().hex[:8]
    src_p=_persist_uploaded_file(src,temp,'video_source')
    vid_p=_persist_uploaded_file(video,temp,'video_target')
    return create_job('faceswap_video',{
        'source':src_p,'video':vid_p,
        'frame_stride':int(frame_stride),'max_frames':int(max_frames),
        'keep_audio':bool(keep_audio),'crf':int(crf),
        'preset':str(preset),'detect_every':int(detect_every),
        'preview_seconds':int(preview_seconds or 0),
        'backend':str(backend or 'auto'),
        'preflight_seconds':int(preflight_seconds or 0),
    })

def restart_video_job(job_id):
    if not job_id:
        return 'Job ID vide'
    j=load_job(job_id)
    if not j:
        return 'Job introuvable'
    if j.get('type')!='faceswap_video':
        return 'Ce job n’est pas un face swap vidéo'
    if j.get('status')=='done':
        return 'Ce job est déjà terminé'
    _clear_cancel(job_id)
    try:
        os.remove(_pause_flag(job_id))
    except Exception:
        pass
    j['status']='queued'
    j['runtime_id']=QWEN_RUNTIME_ID
    j['error']=''
    j['info']='Reprise du job à partir des checkpoints existants…'
    save_job(j)
    with JOB_LOCK:
        JOB_FUTURES[job_id]=JOB_EXECUTOR.submit(_execute_job,job_id)
    return f'Job {job_id} relancé avec reprise checkpoints'

def _format_video_status(job):
    status=f"{job.get('status','?')} — {job.get('updated_at','')}"
    total=int(job.get('progress_total') or 0)
    if total>0:
        status += f"\nProgression: {int(job.get('progress_current') or 0)}/{total} ({float(job.get('progress_pct') or 0):.2f}%)"
    if job.get('error'):
        status += '\n'+job['error'][-3000:]
    return status

def job_status_video_view(job_id):
    j=load_job(job_id)
    if not j:
        return 'Job introuvable',None,''
    return _format_video_status(j),j.get('result_path') or j.get('preview_path') or None,j.get('info') or ''

_ORIGINAL_EXECUTE_JOB=_execute_job

def _execute_job(job_id):
    job=load_job(job_id)
    if not job:
        return
    if job.get('type')!='faceswap_video':
        return _ORIGINAL_EXECUTE_JOB(job_id)
    try:
        _clear_cancel(job_id)
        job['status']='running'
        job['runtime_id']=QWEN_RUNTIME_ID
        job['progress_current']=0
        job['progress_total']=0
        job['progress_pct']=0.0
        save_job(job)
        p=job['payload']
        preview_seconds=int(p.get('preview_seconds',0) or 0)
        preflight_seconds=int(p.get('preflight_seconds',0) or 0)
        if preview_seconds<=0 and preflight_seconds>0:
            preview_path,preview_info=video_face_swap(
                p['source'],p['video'],
                p.get('frame_stride',1),0,
                False,p.get('crf',17),
                p.get('preset','slow'),p.get('detect_every',1),
                preflight_seconds,job_id,p.get('backend','auto')
            )
            job=load_job(job_id) or job
            job['preview_path']=preview_path
            job['info']='Aperçu prêt. Rendu complet en cours…\n'+preview_info
            save_job(job)
            if _is_cancel_requested(job_id):
                raise JobCancelled('Annulation demandée après l’aperçu.')
        result,info=video_face_swap(
            p['source'],p['video'],
            p.get('frame_stride',1),p.get('max_frames',0),
            p.get('keep_audio',True),p.get('crf',17),
            p.get('preset','slow'),p.get('detect_every',1),
            preview_seconds,job_id,p.get('backend','auto')
        )
        job=load_job(job_id) or job
        job['status']='done'
        job['result_path']=result
        job['info']=info
        job['progress_pct']=100.0
        save_job(job)
    except JobCancelled as e:
        job=load_job(job_id) or job
        job['status']='cancelled'
        job['error']=str(e)
        save_job(job)
    except Exception as e:
        job=load_job(job_id) or job
        job['status']='error'
        job['error']=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-6000:]}"
        save_job(job)
    finally:
        gc.collect()

def _video_selftest():
    # Tests légers sans modèle/GPU : math de tracking et garde-fous.
    assert abs(_bbox_iou([0,0,10,10],[0,0,10,10])-1.0)<1e-6
    assert _bbox_iou([0,0,10,10],[20,20,30,30])==0.0
    return True

_video_selftest()
print('✅ Face swap vidéo Ultra chargé (v8.4).')
