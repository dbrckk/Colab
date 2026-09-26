# v8.4 — Face swap vidéo Ultra (tracking + bouche + checkpoints + progression)
import os, subprocess, shutil, uuid, time, json
from pathlib import Path
from fractions import Fraction

VIDEO_FACE_ROOT = os.path.join(ROOT, 'video_faceswap')
VIDEO_JOB_ROOT = os.path.join(JOB_ROOT, 'video_artifacts')
VIDEO_CANCEL_ROOT = os.path.join(JOB_ROOT, 'cancel_flags')
os.makedirs(VIDEO_FACE_ROOT, exist_ok=True)
os.makedirs(VIDEO_JOB_ROOT, exist_ok=True)
os.makedirs(VIDEO_CANCEL_ROOT, exist_ok=True)

class JobCancelled(Exception):
    pass

def _new_output_video(prefix='faceswap_video'):
    ts=time.strftime('%Y%m%d-%H%M%S')
    return os.path.join(OUTPUT_ROOT, f'{ts}_{prefix}_{uuid.uuid4().hex[:6]}.mp4')

def _video_job_dir(job_id):
    p=os.path.join(VIDEO_JOB_ROOT,str(job_id))
    os.makedirs(p,exist_ok=True)
    return p

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
        ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate',
         '-of','default=nokey=1:noprint_wrappers=1',video_path],
        capture_output=True,text=True
    )
    raw=(p.stdout or '').strip().splitlines()
    val=raw[0].strip() if raw else '25/1'
    try:
        return max(1.0,float(Fraction(val)))
    except Exception:
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

def _choose_mode(video_path):
    if not VIDEO_FACE_SWAP_AUTO_MODE:
        return 'ultra'
    duration=_ffprobe_duration(video_path)
    return 'fast' if duration>0 and duration<8.0 else 'ultra'

def _encode_frames(frames_out,fps,target_video,out,keep_audio,crf,preset,preview_seconds=0):
    video_only=os.path.join(os.path.dirname(frames_out),'video_only.mp4')
    enc=['ffmpeg','-y','-framerate',f'{fps:.6f}','-i',os.path.join(frames_out,'%08d.png'),
         '-c:v','libx264','-preset',preset,'-crf',str(crf),'-pix_fmt','yuv420p',video_only]
    p=subprocess.run(enc,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError('Encodage vidéo échoué:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-4000:])
    if keep_audio and int(preview_seconds or 0)<=0:
        mux=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
             '-c:v','copy','-c:a','copy','-shortest',out]
        p=subprocess.run(mux,capture_output=True,text=True)
        if p.returncode!=0:
            shutil.copy2(video_only,out)
    else:
        shutil.copy2(video_only,out)

def video_face_swap(source_image,target_video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1,preview_seconds=0,job_id=None):
    import cv2
    init_faceswap()
    ensure_ffmpeg()
    frame_stride=max(1,int(frame_stride or 1))
    detect_every=max(1,int(detect_every or 1))
    max_frames=max(0,int(max_frames or 0))
    preview_seconds=max(0,int(preview_seconds or 0))
    crf=max(10,min(30,int(crf or 17)))
    preset=str(preset or 'slow')
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
    frames_in=os.path.join(work,'frames_in')
    frames_out=os.path.join(work,'frames_out')
    os.makedirs(frames_in,exist_ok=True)
    os.makedirs(frames_out,exist_ok=True)

    fps=_ffprobe_fps(target_video)
    width,height=_ffprobe_size(target_video)

    if not list(Path(frames_in).glob('*.png')):
        extract=['ffmpeg','-y','-i',target_video,'-vsync','0']
        if frame_stride>1:
            extract += ['-vf',f'select=not(mod(n\\,{frame_stride}))']
        extract += [os.path.join(frames_in,'%08d.png')]
        p=subprocess.run(extract,capture_output=True,text=True)
        if p.returncode!=0:
            raise RuntimeError('Extraction vidéo échouée:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-4000:])

    files=sorted(Path(frames_in).glob('*.png'))
    if preview_seconds>0:
        files=files[:max(1,int(round(fps*preview_seconds)))]
    if max_frames>0:
        files=files[:max_frames]
    if not files:
        raise RuntimeError('Aucune frame extraite')

    total=len(files)
    prev=None
    stable_after_occlusion=0
    processed=swapped=skipped=resumed=0
    started=time.time()
    progress_every=max(1,int(VIDEO_FACE_SWAP_PROGRESS_EVERY or 8))

    for idx,path in enumerate(files,1):
        if _is_cancel_requested(job_id):
            raise JobCancelled('Annulation demandée par l’utilisateur.')

        out_frame=os.path.join(frames_out,f'{idx:08d}.png')
        if VIDEO_FACE_SWAP_AUTO_RESUME and os.path.exists(out_frame) and os.path.getsize(out_frame)>1000:
            resumed+=1
            processed+=1
            if idx%progress_every==0 or idx==total:
                elapsed=max(0.001,time.time()-started)
                speed=idx/elapsed
                eta=max(0,int((total-idx)/max(speed,1e-6)))
                _update_job_progress(job_id,idx,total,'reprise',f'ETA ~ {eta}s')
            continue

        frame=cv2.imread(str(path))
        if frame is None:
            skipped+=1
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
        processed+=1

        if idx%progress_every==0 or idx==total:
            elapsed=max(0.001,time.time()-started)
            speed=idx/elapsed
            eta=max(0,int((total-idx)/max(speed,1e-6)))
            _update_job_progress(
                job_id,idx,total,'traitement vidéo',
                f'ETA ~ {eta}s | mode={mode} | swap={swapped} | protégées={skipped}'
            )

    out=_new_output_video('faceswap_video_preview' if preview_seconds>0 else 'faceswap_video')
    _encode_frames(frames_out,fps,target_video,out,keep_audio,crf,preset,preview_seconds)

    if VIDEO_FACE_SWAP_CLEANUP_TEMP and preview_seconds<=0:
        shutil.rmtree(frames_in,ignore_errors=True)
        shutil.rmtree(frames_out,ignore_errors=True)

    info=(
        f'Face swap vidéo terminé | mode={mode} | {width}x{height} | fps={fps:.3f} | '
        f'frames={processed} | swap={swapped} | protégées={skipped} | reprises={resumed} | '
        f'stride={frame_stride} | detect_every={detect_every} | CRF={crf} | preset={preset} | '
        f'preview={preview_seconds}s\n{out}'
    )
    return out,info

def submit_faceswap_video_job(src,video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1,preview_seconds=0):
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
    })

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
    return _format_video_status(j),j.get('result_path') or None,j.get('info') or ''

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
        job['progress_current']=0
        job['progress_total']=0
        job['progress_pct']=0.0
        save_job(job)
        p=job['payload']
        result,info=video_face_swap(
            p['source'],p['video'],
            p.get('frame_stride',1),p.get('max_frames',0),
            p.get('keep_audio',True),p.get('crf',17),
            p.get('preset','slow'),p.get('detect_every',1),
            p.get('preview_seconds',0),job_id
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
