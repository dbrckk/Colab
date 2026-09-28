from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "qwen_studio_v8"

GROUPS = [
    ["01_setup.py"],
    ["02_models.py"],
    ["03_runtime.py"],
    ["04_faceswap.py"],
    ["05_backend.part00", "05_backend.part01", "05_backend.part02"],
    ["05b_video_faceswap.py"],
    ["06_ui.part00", "06_ui.part01", "06_ui.part02", "06_ui.part03"],
]

def read_group(names):
    return "".join((MOD / name).read_text(encoding="utf-8") for name in names)

for group in GROUPS:
    source = read_group(group)
    compile(source, "+".join(group), "exec")
    ast.parse(source)
    print("OK syntax:", " + ".join(group))

video = (MOD / "05b_video_faceswap.py").read_text(encoding="utf-8")
required_video = [
    "def video_face_swap(",
    "def analyze_video_difficulty(",
    "def pause_job(",
    "def resume_job(",
    "def restart_video_job(",
    "def cleanup_video_cache(",
    "def run_facefusion_ultra(",
    "expression_restorer",
    "face_enhancer",
    "face-mask-types",
    "def _encode_checkpoint_chunks(",
    "def _preflight_protected_ratio(",
    "def _resolve_video_backend(",
    "def _target_anchor_from_video(",
    "def _smooth_face_geometry(",
    "def _contiguous_checkpoint_segments(",
    "talking_likely",
    "auto_upgraded_backend",
    "preflight_protected_ratio",
]
for token in required_video:
    assert token in video, f"Missing video capability: {token}"

setup = (MOD / "01_setup.py").read_text(encoding="utf-8")
required_config = [
    "VIDEO_FACE_SWAP_USE_SEAMLESS_BLEND",
    "VIDEO_FACE_SWAP_MASK_DILATE",
    "VIDEO_FACE_SWAP_MASK_BLUR",
    "VIDEO_FACE_SWAP_PREFLIGHT_SECONDS",
    "VIDEO_FACE_SWAP_PAUSE_ENABLED",
    "VIDEO_FACE_SWAP_AUTO_REFRESH",
    "VIDEO_FACE_SWAP_STREAM_INPUT",
    "VIDEO_FACE_SWAP_PREFLIGHT_MAX_PROTECTED_RATIO",
    "VIDEO_FACE_SWAP_AUTO_QUALITY_FIRST",
    "VIDEO_FACE_SWAP_TARGET_FACE_POSITION",
    "VIDEO_FACE_SWAP_MIN_TARGET_SIM",
    "VIDEO_FACEFUSION_VIDEO_QUALITY",
]
for token in required_config:
    assert token in setup, f"Missing setup option: {token}"

ui = read_group(["06_ui.part00", "06_ui.part01", "06_ui.part02", "06_ui.part03"])
for token in [
    "Face swap vidéo Ultra v8.6",
    "_pause_video_job",
    "_resume_video_job",
    "_restart_video_job",
    "_analyze_video",
    "gr.Timer",
    "_apply_video_preset",
    "Qualité maximale",
    "Faible disque",
    "vfs_target_face_position",
]:
    assert token in ui, f"Missing UI feature: {token}"


class GradioComponentValidator(ast.NodeVisitor):
    """Static checks for constructor mistakes that Python syntax alone cannot catch."""
    def __init__(self):
        self.errors = []

    @staticmethod
    def _number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            return -float(node.operand.value)
        return None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "gr":
            component = node.func.attr
            if component == "Slider":
                minimum = self._number(node.args[0]) if len(node.args) >= 1 else None
                maximum = self._number(node.args[1]) if len(node.args) >= 2 else None
                value = None
                for kw in node.keywords:
                    if kw.arg == "minimum":
                        minimum = self._number(kw.value)
                    elif kw.arg == "maximum":
                        maximum = self._number(kw.value)
                    elif kw.arg == "value":
                        value = self._number(kw.value)
                if minimum is not None and maximum is not None and minimum >= maximum:
                    self.errors.append(
                        f"gr.Slider line {getattr(node, 'lineno', '?')}: minimum={minimum} must be < maximum={maximum}"
                    )
                if minimum is not None and maximum is not None and value is not None and not (minimum <= value <= maximum):
                    self.errors.append(
                        f"gr.Slider line {getattr(node, 'lineno', '?')}: value={value} outside [{minimum}, {maximum}]"
                    )
        self.generic_visit(node)

ui_ast = ast.parse(ui)
ui_validator = GradioComponentValidator()
ui_validator.visit(ui_ast)
assert not ui_validator.errors, "\n".join(ui_validator.errors)

nb_path = ROOT / "Qwen_Image_2_1_Heretic_GGUF_Gradio_v8_PremiumUX.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
assert nb.get("nbformat") == 4
code = "\n".join(
    "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
    for cell in nb.get("cells", [])
    if cell.get("cell_type") == "code"
)
assert "05b_video_faceswap.py" in code
assert "05_video_faceswap.py" not in code
assert code.count("05b_video_faceswap.py") == 1
assert "v8.6" in nb_path.read_text(encoding="utf-8")
assert "QWEN_STUDIO_SOURCE_REF" in code
assert "api.github.com/repos/{REPO}/commits/main" in code
assert "compile(source, local_path, \"exec\")" in code

print("Qwen Studio validation passed.")


backend = read_group(["05_backend.part00", "05_backend.part01", "05_backend.part02"])
assert "QWEN_RUNTIME_ID" in backend
assert "JOB_BASE = DRIVE_ROOT if USE_DRIVE else ROOT" in backend
assert "j.get('runtime_id') == QWEN_RUNTIME_ID" in backend

assert "_encode_checkpoint_chunks(" in video
assert "cv2.VideoCapture" in video
assert "_restore_video_checkpoints(job_id,frames_out)" not in video or "return _latest_checkpoint_end(job_id)" in video


# VIDEO_* configuration references must all be defined by 01_setup.py.
setup_tree = ast.parse(setup)
defined_video = set()
for node in ast.walk(setup_tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("VIDEO_"):
                defined_video.add(target.id)

combined_video_ui_tree = ast.parse(video + "\n" + ui)

locally_defined_video = set()
for node in ast.walk(combined_video_ui_tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.startswith("VIDEO_"):
                locally_defined_video.add(target.id)

referenced_video = {
    node.id
    for node in ast.walk(combined_video_ui_tree)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id.startswith("VIDEO_")
}
missing_video = sorted(referenced_video - defined_video - locally_defined_video)
assert not missing_video, "Undefined VIDEO_* configuration names: " + ", ".join(missing_video)
print(
    f"VIDEO_* references passed "
    f"({len(referenced_video)} referenced, {len(locally_defined_video)} internal)."
)
