from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def find_input_root() -> Path:
    roots = list(Path("/kaggle/input").glob("*/job_config.json"))
    if not roots:
        raise FileNotFoundError("job_config.json introuvable dans /kaggle/input.")
    return roots[0].parent

INPUT = find_input_root()
CONFIG = json.loads((INPUT / "job_config.json").read_text(encoding="utf-8"))

def pip_install(*packages: str):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", *packages],
        check=True,
        timeout=1800,
    )

def write_result(status: str, **extra):
    payload = {"job_id": CONFIG.get("job_id"), "status": status, **extra}
    (OUT / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return path
    urllib.request.urlretrieve(url, path)
    return path

def ensure_crane() -> Path:
    crane = WORK / "crane"
    if crane.exists():
        crane.chmod(0o755)
        return crane
    archive = WORK / "go-containerregistry.tar.gz"
    download(
        "https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz",
        archive,
    )
    with tarfile.open(archive, "r:gz") as tf:
        member = next(m for m in tf.getmembers() if Path(m.name).name == "crane")
        member.name = "crane"
        tf.extract(member, WORK)
    crane.chmod(0o755)
    return crane

def sdcli_works(path: Path) -> bool:
    try:
        path.chmod(0o755)
        env = os.environ.copy()
        bindir = str(path.parent)
        env["LD_LIBRARY_PATH"] = bindir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        p = subprocess.run(
            [str(path), "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=20,
        )
        return p.returncode in (0, 1)
    except Exception:
        return False

def compile_sdcli() -> Path:
    if shutil.which("cmake") is None or shutil.which("ninja") is None:
        subprocess.run(
            ["apt-get", "-qq", "update"],
            check=True,
            timeout=600,
        )
        subprocess.run(
            ["apt-get", "-qq", "install", "-y", "--no-install-recommends", "cmake", "ninja-build"],
            check=True,
            timeout=900,
        )
    src = WORK / "stable-diffusion.cpp"
    if not src.exists():
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--recurse-submodules",
                "--shallow-submodules", "https://github.com/leejet/stable-diffusion.cpp",
                str(src),
            ],
            check=True,
            timeout=900,
        )
    subprocess.run(
        [
            "cmake", "-S", ".", "-B", "build", "-G", "Ninja",
            "-DSD_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
            "-DSD_WEBP=OFF", "-DSD_WEBM=OFF",
        ],
        cwd=src,
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["cmake", "--build", "build", "--target", "sd-cli", "-j", "4"],
        cwd=src,
        check=True,
        timeout=2400,
    )
    path = src / "build" / "bin" / "sd-cli"
    if not sdcli_works(path):
        raise RuntimeError("Le sd-cli compilé ne démarre pas.")
    return path

def ensure_sdcli() -> Path:
    runtime = WORK / "sdcpp-runtime"
    if runtime.exists():
        for candidate in runtime.rglob("sd-cli"):
            if candidate.is_file() and sdcli_works(candidate):
                return candidate
    try:
        crane = ensure_crane()
        runtime.mkdir(parents=True, exist_ok=True)
        p1 = subprocess.Popen(
            [
                str(crane), "export", "--platform", "linux/amd64",
                "ghcr.io/leejet/stable-diffusion.cpp:master-cuda", "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p2 = subprocess.Popen(
            ["tar", "-xf", "-", "-C", str(runtime), "sd.cpp/bin"],
            stdin=p1.stdout,
        )
        if p1.stdout:
            p1.stdout.close()
        rc2 = p2.wait(timeout=1200)
        rc1 = p1.wait(timeout=60)
        if rc1 == 0 and rc2 == 0:
            for candidate in runtime.rglob("sd-cli"):
                if candidate.is_file() and sdcli_works(candidate):
                    return candidate
    except Exception as exc:
        print("Prebuilt sd-cli indisponible, compilation locale:", exc, flush=True)
    return compile_sdcli()

_HF_READY = False

def hf_file(repo_id: str, filename: str) -> Path:
    global _HF_READY
    if not _HF_READY:
        pip_install("huggingface_hub")
        _HF_READY = True
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))

def load_qwen_models(include_vision: bool = False) -> dict[str, Path]:
    specs = {
        "heretic": (
            "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF",
            "qwen3vl_8b_heretic-Q4_K_M.gguf",
        ),
        "dit": (
            "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
            "qwen-image-2.1-UC-Q4_K_M.gguf",
        ),
        "vae": (
            "abenzerps/Qwen-Image-2.1-Uncensored-GGUF",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        ),
    }
    if include_vision:
        specs["mmproj"] = (
            "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF",
            "mmproj-qwen3vl_8b_heretic-f16.gguf",
        )

    def fetch(item):
        key, (repo_id, filename) = item
        return key, hf_file(repo_id, filename)

    with ThreadPoolExecutor(max_workers=min(4, len(specs))) as pool:
        return dict(pool.map(fetch, specs.items()))

def run_image():
    sdcli = ensure_sdcli()
    models = load_qwen_models(include_vision=False)
    heretic = models["heretic"]
    dit = models["dit"]
    vae = models["vae"]

    aspects = {
        "1:1": (1024, 1024),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
    }
    width, height = aspects.get(CONFIG.get("aspect", "1:1"), (1024, 1024))
    seed = int(CONFIG.get("seed", -1))
    if seed < 0:
        seed = random.randint(1, 2_000_000_000)

    out = OUT / "image.png"
    cmd = [
        str(sdcli),
        "--diffusion-model", str(dit),
        "--vae", str(vae),
        "--llm", str(heretic),
        "-p", CONFIG.get("prompt", ""),
        "--negative-prompt", CONFIG.get("negative_prompt", ""),
        "--cfg-scale", str(float(CONFIG.get("cfg", 1.0))),
        "--sampling-method", "euler",
        "--steps", str(int(CONFIG.get("steps", 25))),
        "--seed", str(seed),
        "-W", str(width),
        "-H", str(height),
        "--offload-to-cpu",
        "--diffusion-fa",
        "-o", str(out),
    ]
    env = os.environ.copy()
    bindir = str(sdcli.parent)
    env["LD_LIBRARY_PATH"] = bindir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10800)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(((p.stdout or "") + "\n" + (p.stderr or ""))[-7000:])
    write_result("done", task="image", seed=seed, files=[out.name])

def run_image_edit():
    sdcli = ensure_sdcli()
    source_name = CONFIG.get("source_image")
    if not source_name:
        raise ValueError("image_edit requiert source_image.")
    source = INPUT / source_name
    if not source.exists():
        raise FileNotFoundError(f"Image source absente: {source_name}")

    models = load_qwen_models(include_vision=True)
    heretic = models["heretic"]
    mmproj = models["mmproj"]
    dit = models["dit"]
    vae = models["vae"]

    aspects = {
        "1:1": (1024, 1024),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
    }
    width, height = aspects.get(CONFIG.get("aspect", "1:1"), (1024, 1024))
    seed = int(CONFIG.get("seed", -1))
    if seed < 0:
        seed = random.randint(1, 2_000_000_000)

    out = OUT / "edited_image.png"
    cmd = [
        str(sdcli),
        "--diffusion-model", str(dit),
        "--vae", str(vae),
        "--llm", str(heretic),
        "--llm_vision", str(mmproj),
        "-r", str(source),
        "-p", CONFIG.get("prompt", ""),
        "--negative-prompt", CONFIG.get("negative_prompt", ""),
        "--cfg-scale", str(float(CONFIG.get("cfg", 1.0))),
        "--sampling-method", "euler",
        "--steps", str(int(CONFIG.get("steps", 25))),
        "--seed", str(seed),
        "-W", str(width),
        "-H", str(height),
        "--offload-to-cpu",
        "--diffusion-fa",
        "-o", str(out),
    ]
    env = os.environ.copy()
    bindir = str(sdcli.parent)
    env["LD_LIBRARY_PATH"] = bindir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10800)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(((p.stdout or "") + "\n" + (p.stderr or ""))[-7000:])
    write_result("done", task="image_edit", seed=seed, files=[out.name])

def run_video_faceswap():
    source_name = CONFIG.get("source_image")
    target_name = CONFIG.get("target_video")
    if not source_name or not target_name:
        raise ValueError("video_faceswap requiert source_image et target_video.")

    source = INPUT / source_name
    target = INPUT / target_name
    if not source.exists() or not target.exists():
        raise FileNotFoundError("Entrée face swap absente du dataset Kaggle.")

    ff = WORK / "facefusion"
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--branch", "3.9.0",
            "https://github.com/facefusion/facefusion.git", str(ff),
        ],
        check=True,
        timeout=900,
    )
    subprocess.run(
        [sys.executable, "install.py", "cuda@12", "--skip-conda"],
        cwd=ff,
        check=True,
        timeout=2400,
    )

    out = OUT / "faceswap.mp4"
    cmd = [
        sys.executable, "facefusion.py", "headless-run",
        "--workflow-strategy", "disk",
        "--processors", "face_swapper", "expression_restorer", "face_enhancer",
        "--face-mask-types", "occlusion", "region",
        "--face-enhancer-model", "gfpgan_1.4",
        "--face-enhancer-blend", "60",
        "--expression-restorer-model", "live_portrait",
        "--expression-restorer-factor", "80",
        "--output-video-preset", "slow",
        "--output-video-quality", "95",
        "--output-audio-quality", "95",
        "--execution-providers", "cuda",
        "-s", str(source),
        "-t", str(target),
        "-o", str(out),
    ]
    p = subprocess.run(cmd, cwd=ff, capture_output=True, text=True, timeout=10800)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(((p.stdout or "") + "\n" + (p.stderr or ""))[-7000:])
    write_result("done", task="video_faceswap", files=[out.name])

def main():
    task = CONFIG.get("task", "image")
    try:
        if task == "image":
            run_image()
        elif task == "image_edit":
            run_image_edit()
        elif task == "video_faceswap":
            run_video_faceswap()
        else:
            raise ValueError(f"Tâche inconnue: {task}")
    except Exception as exc:
        write_result("error", error=f"{type(exc).__name__}: {exc}")
        raise

if __name__ == "__main__":
    main()
