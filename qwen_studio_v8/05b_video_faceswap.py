# v8.2 — Face swap vidéo robuste (lazy, Gradio only)
import os, subprocess, shutil, uuid
from pathlib import Path
from fractions import Fraction

VIDEO_FACE_ROOT = os.path.join(ROOT, 'video_faceswap')
os.makedirs(VIDEO_FACE_ROOT, exist_ok=True)

def _new_output_video(prefix='faceswap_video'):
    ts=time.strftime('%Y%m%d-%H%M%S')
    return os.path.join(OUTPUT_ROOT, f'{ts}_{prefix}_{uuid.uuid4().hex[:6]}.mp4')

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
         '-of','default=nokey=1:noprint_wrappers=1', video_path],
        capture_output=True, text=True
    )
    raw=(p.stdout or '').strip().splitlines()
    val=raw[0].strip() if raw else '25/1'
    try:
        return max(1.0, float(Fraction(val)))
    except Exception:
        return 25.0

def _ffprobe_size(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames',
         '-of','default=nokey=1:noprint_wrappers=1', video_path],
        capture_output=True, text=True
    )
    vals=[x.strip() for x in (p.stdout or '').splitlines() if x.strip()]
    try:
        width=int(vals[0]); height=int(vals[1])
    except Exception:
        width=height=0
    nb=vals[2] if len(vals)>=3 else '?'
    return width, height, nb

def _face_bbox(face):
    x1,y1,x2,y2=[float(v) for v in face.bbox[:4]]
    return [x1,y1,x2,y2]

def _face_area(face):
    x1,y1,x2,y2=_face_bbox(face)
    return max(1.0,(x2-x1)*(y2-y1))

def _face_center(face):
    x1,y1,x2,y2=_face_bbox(face)
    return ((x1+x2)/2.0,(y1+y2)/2.0)

def _bbox_iou(a,b):
    ax1,ay1,ax2,ay2=a
    bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1)
    ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1)
    inter=iw*ih
    area1=max(1.0,(ax2-ax1)*(ay2-ay1))
    area2=max(1.0,(bx2-bx1)*(by2-by1))
    return inter/max(1.0,area1+area2-inter)

def _cosine_sim(a,b):
    import numpy as np
    if a is None or b is None:
        return 0.0
    a=np.asarray(a,dtype='float32').reshape(-1)
    b=np.asarray(b,dtype='float32').reshape(-1)
    an=float(np.linalg.norm(a)); bn=float(np.linalg.norm(b))
    if an<=1e-8 or bn<=1e-8:
        return 0.0
    return float(np.dot(a,b)/(an*bn))

def _track_signature(face):
    return {
        'bbox':_face_bbox(face),
        'area':_face_area(face),
        'center':_face_center(face),
        'embedding':getattr(face,'normed_embedding',None),
        'det_score':float(getattr(face,'det_score',0.0) or 0.0),
        'kps':getattr(face,'kps',None),
    }

def _select_face(faces, prev_sig=None):
    if not faces:
        return None,None
    if prev_sig is None or not VIDEO_FACE_SWAP_STRICT_TRACKING:
        f=_largest(faces)
        return f,_track_signature(f)
    best=None; best_score=-1e9
    for face in faces:
        sig=_track_signature(face)
        det=sig['det_score']
        iou=_bbox_iou(sig['bbox'],prev_sig['bbox'])
        emb=_cosine_sim(sig['embedding'],prev_sig['embedding'])
        cx,cy=sig['center']; px,py=prev_sig['center']
        drift=((cx-px)**2+(cy-py)**2)**0.5
        score=(det*2.0)+(iou*3.5)+(emb*3.0)-(drift*0.002)
        if score>best_score:
            best_score=score
            best=(face,sig)
    return best

def _unstable(sig, prev_sig=None):
    if sig['det_score'] < float(VIDEO_FACE_SWAP_MIN_DET_SCORE):
        return True
    if prev_sig is None:
        return False
    ratio=sig['area']/max(1.0,prev_sig['area'])
    if ratio < float(VIDEO_FACE_SWAP_AREA_RATIO_MIN) or ratio > float(VIDEO_FACE_SWAP_AREA_RATIO_MAX):
        return True
    iou=_bbox_iou(sig['bbox'],prev_sig['bbox'])
    emb=_cosine_sim(sig['embedding'],prev_sig['embedding'])
    if iou < float(VIDEO_FACE_SWAP_MIN_IOU) and emb < float(VIDEO_FACE_SWAP_MIN_EMBED_SIM):
        return True
    return False

def _post_blend_face(original, swapped, sig):
    import numpy as np, cv2
    if not VIDEO_FACE_SWAP_USE_SEAMLESS_BLEND:
        return swapped
    kps=sig.get('kps')
    if kps is None:
        return swapped
    try:
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
        return cv2.seamlessClone(swapped,original,mask,center,cv2.NORMAL_CLONE)
    except Exception:
        return swapped

def video_face_swap(source_image,target_video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1):
    import cv2
    init_faceswap()
    ensure_ffmpeg()
    frame_stride=max(1,int(frame_stride or 1))
    detect_every=max(1,int(detect_every or 1))
    max_frames=max(0,int(max_frames or 0))
    crf=int(crf or 17)
    preset=str(preset or 'slow')

    source=cv2.imread(source_image)
    if source is None:
        raise RuntimeError('Image source illisible')
    source_face=_largest(FS_APP.get(source))

    work=os.path.join(VIDEO_FACE_ROOT,'job_'+uuid.uuid4().hex[:8])
    in_frames=os.path.join(work,'frames_in')
    out_frames=os.path.join(work,'frames_out')
    os.makedirs(in_frames,exist_ok=True)
    os.makedirs(out_frames,exist_ok=True)

    fps=_ffprobe_fps(target_video)
    width,height,_=_ffprobe_size(target_video)

    extract=['ffmpeg','-y','-i',target_video,'-vsync','0']
    if frame_stride>1:
        extract += ['-vf',f'select=not(mod(n\\,{frame_stride}))']
    extract += [os.path.join(in_frames,'%08d.png')]
    p=subprocess.run(extract,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError('Extraction frames échouée:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-4000:])

    frames=sorted(Path(in_frames).glob('*.png'))
    if not frames:
        raise RuntimeError('Aucune frame extraite')
    if max_frames>0:
        frames=frames[:max_frames]

    prev_sig=None
    processed=swapped_count=skipped_count=0
    for idx,frame_path in enumerate(frames,start=1):
        frame=cv2.imread(str(frame_path))
        if frame is None:
            continue
        original=frame.copy()
        try:
            faces=FS_APP.get(frame)
        except Exception:
            faces=[]
        face,sig=_select_face(faces,prev_sig) if faces else (None,None)

        bad=(face is None or sig is None)
        if not bad and VIDEO_FACE_SWAP_SKIP_ON_OCCLUSION:
            bad=_unstable(sig,prev_sig)

        if bad:
            result=original
            skipped_count+=1
        else:
            try:
                swapped=FS_SWAPPER.get(frame.copy(),face,source_face,paste_back=True)
                result=_post_blend_face(original,swapped,sig)
                prev_sig=sig
                swapped_count+=1
            except Exception:
                result=original
                skipped_count+=1

        cv2.imwrite(os.path.join(out_frames,f'{idx:08d}.png'),result)
        processed+=1

    out=_new_output_video('faceswap_video')
    video_only=os.path.join(work,'video_only.mp4')
    enc=['ffmpeg','-y','-framerate',f'{fps:.6f}','-i',os.path.join(out_frames,'%08d.png'),
         '-c:v','libx264','-preset',preset,'-crf',str(crf),'-pix_fmt','yuv420p',video_only]
    p=subprocess.run(enc,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError('Encodage vidéo échoué:\n'+((p.stdout or '')+'\n'+(p.stderr or ''))[-4000:])

    if keep_audio:
        mux=['ffmpeg','-y','-i',video_only,'-i',target_video,'-map','0:v:0','-map','1:a?',
             '-c:v','copy','-c:a','copy','-shortest',out]
        p=subprocess.run(mux,capture_output=True,text=True)
        if p.returncode!=0:
            shutil.copy2(video_only,out)
    else:
        shutil.copy2(video_only,out)

    info=(
        f'Video face swap terminé | {width}x{height} | fps={fps:.3f} | '
        f'frames={processed} | swappées={swapped_count} | protégées={skipped_count} | '
        f'crf={crf} | preset={preset} | strict_tracking={VIDEO_FACE_SWAP_STRICT_TRACKING}\n{out}'
    )
    return out,info

def submit_faceswap_video_job(src,video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset='slow',detect_every=1):
    if not src or not video:
        raise ValueError('Ajoute image source et vidéo cible')
    temp_id=uuid.uuid4().hex[:8]
    src_p=_persist_uploaded_file(src,temp_id,'video_source')
    vid_p=_persist_uploaded_file(video,temp_id,'video_target')
    return create_job('faceswap_video',{
        'source':src_p,'video':vid_p,'frame_stride':int(frame_stride),'max_frames':int(max_frames),
        'keep_audio':bool(keep_audio),'crf':int(crf),'preset':str(preset),'detect_every':int(detect_every),
    })

def job_status_video_view(job_id):
    return job_status_view(job_id)

# Redéfinit le worker pour prendre en charge les vidéos.
def _execute_job(job_id):
    job=load_job(job_id)
    if not job:
        return
    try:
        job['status']='running'
        save_job(job)
        p=job['payload']
        if job['type']=='generate':
            result,info=run_sd(p['prompt'],p['aspect'],p['steps'],p['cfg'],p['seed'],None)
        elif job['type']=='edit':
            result,info=run_sd(p['prompt'],p['aspect'],p['steps'],p['cfg'],p['seed'],p['reference_image'])
        elif job['type']=='faceswap':
            result,info=face_swap(p['source'],p['target'])
        elif job['type']=='faceswap_video':
            result,info=video_face_swap(
                p['source'],p['video'],p.get('frame_stride',1),p.get('max_frames',0),
                p.get('keep_audio',True),p.get('crf',17),p.get('preset','slow'),p.get('detect_every',1)
            )
        else:
            raise RuntimeError(f"Type de job inconnu: {job['type']}")
        job=load_job(job_id) or job
        job['status']='done'
        job['result_path']=result
        job['info']=info
        save_job(job)
    except Exception as e:
        job=load_job(job_id) or job
        job['status']='error'
        job['error']=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-6000:]}"
        save_job(job)
    finally:
        gc.collect()

print('✅ Face swap vidéo robuste chargé (v8.2).')
