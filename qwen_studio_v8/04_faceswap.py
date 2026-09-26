# 6) Face swap — v7.6 : téléchargement différé au premier usage
import os

FACE_ROOT = os.path.join(MODEL_ROOT, 'faceswap')
os.makedirs(FACE_ROOT, exist_ok=True)
INSWAPPER = os.path.join(FACE_ROOT, 'inswapper_128.onnx')

print('✅ Face swap configuré en mode différé.')
if os.path.exists(INSWAPPER) and os.path.getsize(INSWAPPER) > 10_000_000:
    print('Modèle inswapper déjà présent:', INSWAPPER)
else:
    print('Le modèle inswapper sera téléchargé au premier Face Swap.')
