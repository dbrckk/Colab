# v8.2 — Face swap vidéo robuste
import os, subprocess, shutil, uuid, time
from pathlib import Path

VIDEO_FACE_ROOT = os.path.join(ROOT, "video_faceswap")
os.makedirs(VIDEO_FACE_ROOT, exist_ok=True)

def ensure_ffmpeg():
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    subprocess.run(
        "apt-get -qq update && apt-get -qq install -y --no-install-recommends ffmpeg",
        shell=True, check=True, timeout=600
    )

def _video_new_output(prefix="faceswap_video"):
    ts=time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(OUTPUT_ROOT, f"{ts}_{prefix}_{uuid.uuid4().hex[:6]}.mp4")

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
    a=np.asarray(a,dtype="float32").reshape(-1)
    b=np.asarray(b,dtype="float32").reshape(-1)
    na=float(np.linalg.norm(a)); nb=float(np.linalg.norm(b))
    if na<1e-8 or nb<1e-8:
        return 0.0
    return float(np.dot(a,b)/(na*nb))

def _sig(face):
    return {
        "bbox":_face_bbox(face),
        "area":_face_area(face),
        "center":_face_center(face),
        "embedding":getattr(face,"normed_embedding",None),
        "det_score":float(getattr(face,"det_score",0.0) or 0.0),
    }

def _select_face(faces, prev=None):
    if not faces:
        return None,None
    if prev is None or not VIDEO_FACE_SWAP_STRICT_TRACKING:
        f=_largest(faces)
        return f,_sig(f)
    best=None; best_score=-1e9
    px,py=prev["center"]
    for f in faces:
        s=_sig(f)
        cx,cy=s["center"]
        drift=((cx-px)**2+(cy-py)**2)**0.5
        iou=_bbox_iou(s["bbox"],prev["bbox"])
        emb=_cosine(s["embedding"],prev["embedding"])
        score=s["det_score"]*2.0+iou*3.5+emb*3.0-drift*0.002
        if score>best_score:
            best_score=score
            best=(f,s)
    return best if best else (None,None)

def _unstable(sig,prev):
    if sig is None:
        return True
    if sig["det_score"] < float(VIDEO_FACE_SWAP_MIN_DET_SCORE):
        return True
    if prev is None:
        return False
    ratio=sig["area"]/max(1.0,prev["area"])
    if ratio<float(VIDEO_FACE_SWAP_AREA_RATIO_MIN) or ratio>float(VIDEO_FACE_SWAP_AREA_RATIO_MAX):
        return True
    iou=_bbox_iou(sig["bbox"],prev["bbox"])
    emb=_cosine(sig["embedding"],prev["embedding"])
    if iou<float(VIDEO_FACE_SWAP_MIN_IOU) and emb<float(VIDEO_FACE_SWAP_MIN_EMBED_SIM):
        return True
    return False

def _ffprobe_fps(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate",
         "-of","default=nokey=1:noprint_wrappers=1",video_path],
        capture_output=True,text=True
    )
    raw=(p.stdout or "").strip().splitlines()
    val=raw[0].strip() if raw else "25/1"
    try:
        num,den=val.split("/",1)
        return max(1.0,float(num)/max(float(den),1e-8))
    except Exception:
        try:return max(1.0,float(val))
        except Exception:return 25.0

def _ffprobe_size(video_path):
    ensure_ffmpeg()
    p=subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height",
         "-of","csv=s=x:p=0",video_path],
        capture_output=True,text=True
    )
    try:
        w,h=(p.stdout or "").strip().split("x",1)
        return int(w),int(h)
    except Exception:
        return 0,0

def video_face_swap(source_image,target_video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset="slow",detect_every=1):
    import cv2
    init_faceswap()
    ensure_ffmpeg()

    frame_stride=max(1,int(frame_stride or 1))
    detect_every=max(1,int(detect_every or 1))
    max_frames=max(0,int(max_frames or 0))
    crf=max(10,min(30,int(crf or 17)))
    preset=str(preset or "slow")

    source=cv2.imread(source_image)
    if source is None:
        raise RuntimeError("Image source illisible")
    source_face=_largest(FS_APP.get(source))

    work=os.path.join(VIDEO_FACE_ROOT,"job_"+uuid.uuid4().hex[:8])
    frames_in=os.path.join(work,"in")
    frames_out=os.path.join(work,"out")
    os.makedirs(frames_in,exist_ok=True)
    os.makedirs(frames_out,exist_ok=True)

    fps=_ffprobe_fps(target_video)
    width,height=_ffprobe_size(target_video)

    extract=["ffmpeg","-y","-i",target_video,"-vsync","0"]
    if frame_stride>1:
        extract += ["-vf",f"select=not(mod(n\\,{frame_stride}))"]
    extract += [os.path.join(frames_in,"%08d.png")]
    p=subprocess.run(extract,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError("Extraction vidéo échouée:\n"+((p.stdout or "")+"\n"+(p.stderr or ""))[-4000:])

    files=sorted(Path(frames_in).glob("*.png"))
    if max_frames>0:
        files=files[:max_frames]
    if not files:
        raise RuntimeError("Aucune frame extraite")

    prev=None
    processed=swapped=skipped=0

    for idx,path in enumerate(files,1):
        frame=cv2.imread(str(path))
        if frame is None:
            continue

        # Détection à chaque frame par défaut. Cela coûte plus de temps mais réduit les glitches
        # lors de parole, rotation, main/objet devant le visage et mouvements rapides.
        try:
            faces=FS_APP.get(frame)
        except Exception:
            faces=[]

        face,sig=_select_face(faces,prev)
        bad=(face is None or sig is None)
        if not bad and VIDEO_FACE_SWAP_SKIP_ON_OCCLUSION:
            bad=_unstable(sig,prev)

        if bad:
            # Préserver la frame originale est préférable à un swap faux/corrompu.
            result=frame
            skipped+=1
        else:
            try:
                result=FS_SWAPPER.get(frame.copy(),face,source_face,paste_back=True)
                prev=sig
                swapped+=1
            except Exception:
                result=frame
                skipped+=1

        cv2.imwrite(os.path.join(frames_out,f"{idx:08d}.png"),result)
        processed+=1

    out=_video_new_output()
    video_only=os.path.join(work,"video_only.mp4")
    enc=[
        "ffmpeg","-y","-framerate",f"{fps:.6f}","-i",os.path.join(frames_out,"%08d.png"),
        "-c:v","libx264","-preset",preset,"-crf",str(crf),"-pix_fmt","yuv420p",video_only
    ]
    p=subprocess.run(enc,capture_output=True,text=True)
    if p.returncode!=0:
        raise RuntimeError("Encodage vidéo échoué:\n"+((p.stdout or "")+"\n"+(p.stderr or ""))[-4000:])

    if keep_audio:
        mux=[
            "ffmpeg","-y","-i",video_only,"-i",target_video,
            "-map","0:v:0","-map","1:a?","-c:v","copy","-c:a","copy","-shortest",out
        ]
        p=subprocess.run(mux,capture_output=True,text=True)
        if p.returncode!=0:
            shutil.copy2(video_only,out)
    else:
        shutil.copy2(video_only,out)

    info=(
        f"Face swap vidéo terminé | {width}x{height} | fps={fps:.3f} | "
        f"frames={processed} | swap={swapped} | préservées={skipped} | "
        f"stride={frame_stride} | CRF={crf} | preset={preset}\n{out}"
    )
    return out,info

def submit_faceswap_video_job(src,video,frame_stride=1,max_frames=0,keep_audio=True,crf=17,preset="slow",detect_every=1):
    if not src or not video:
        raise ValueError("Ajoute un visage source et une vidéo cible")
    temp=uuid.uuid4().hex[:8]
    src_p=_persist_uploaded_file(src,temp,"video_source")
    vid_p=_persist_uploaded_file(video,temp,"video_target")
    return create_job("faceswap_video",{
        "source":src_p,"video":vid_p,
        "frame_stride":int(frame_stride),"max_frames":int(max_frames),
        "keep_audio":bool(keep_audio),"crf":int(crf),
        "preset":str(preset),"detect_every":int(detect_every)
    })

def job_status_video_view(job_id):
    j=load_job(job_id)
    if not j:
        return "Job introuvable",None,""
    status=f"{j.get('status','?')} — {j.get('updated_at','')}"
    if j.get("error"):
        status += "\n"+j["error"][-3000:]
    return status,j.get("result_path") or None,j.get("info") or ""

_ORIGINAL_EXECUTE_JOB=_execute_job

def _execute_job(job_id):
    job=load_job(job_id)
    if not job:
        return
    if job.get("type")!="faceswap_video":
        return _ORIGINAL_EXECUTE_JOB(job_id)
    try:
        job["status"]="running"
        save_job(job)
        p=job["payload"]
        result,info=video_face_swap(
            p["source"],p["video"],
            p.get("frame_stride",1),p.get("max_frames",0),
            p.get("keep_audio",True),p.get("crf",17),
            p.get("preset","slow"),p.get("detect_every",1)
        )
        job=load_job(job_id) or job
        job["status"]="done"
        job["result_path"]=result
        job["info"]=info
        save_job(job)
    except Exception as e:
        job=load_job(job_id) or job
        job["status"]="error"
        job["error"]=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-6000:]}"
        save_job(job)
    finally:
        gc.collect()
