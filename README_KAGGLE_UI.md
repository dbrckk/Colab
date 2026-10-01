# Qwen Kaggle Studio

## Lancement recommandé sur smartphone

Ouvre `Qwen_Kaggle_Studio_Controller.ipynb` dans Google Colab et exécute son unique cellule. Le contrôleur monte Google Drive, lance Gradio et utilise Kaggle uniquement pour les tâches GPU. Les résultats sont conservés dans `MyDrive/QwenKaggleStudio/`.


Cette interface transforme le repo en **contrôleur de jobs Kaggle**.

## Flux

1. Lance :
   ```bash
   python launch_kaggle_ui.py
   ```
2. Ouvre l'onglet **Paramètres Kaggle** et renseigne :
   - `KAGGLE_USERNAME`
   - `KAGGLE_API_TOKEN` (**recommandé**)
   
   L'ancien `KAGGLE_KEY` reste accepté en fallback.
3. Dans **Générer**, choisis :
   - `image` : prompt → Qwen Image 2.1 / Heretic GGUF sur un GPU Kaggle.
   - `video_faceswap` : visage source + vidéo cible → FaceFusion Ultra sur Kaggle.
4. L'UI :
   - crée un job local ;
   - crée un dataset Kaggle privé contenant la config et les entrées ;
   - crée/lance un kernel Kaggle privé GPU ;
   - surveille son état ;
   - télécharge les outputs ;
   - les stocke dans `storage/kaggle_media/<job_id>/` ;
   - les indexe dans SQLite ;
   - les affiche dans l'UI.
5. Par défaut, le kernel et le dataset temporaires sont supprimés après récupération.

## Pourquoi un dataset temporaire ?

Kaggle attend les gros fichiers d'entrée (notamment les vidéos) comme source de données. Le contrôleur crée donc un **dataset privé par job** au lieu d'essayer d'injecter une vidéo directement dans le code du kernel.

## Stockage local

- Base : `storage/kaggle_jobs.sqlite3`
- Médias : `storage/kaggle_media/<job_id>/`
- Secrets : `.env.local` si tu choisis de les sauvegarder

Le fichier `.env.local` et le dossier `storage/` sont ignorés par Git.

## Variables utiles

```env
KAGGLE_USERNAME=...
KAGGLE_API_TOKEN=...
# ou, ancienne méthode :
# KAGGLE_KEY=...
KAGGLE_ACCELERATOR=NvidiaTeslaT4
KAGGLE_POLL_SECONDS=20
KAGGLE_DELETE_REMOTE_KERNEL=true
QWEN_KAGGLE_SHARE=true
```

## Dépendances

Le launcher installe automatiquement Gradio et le CLI Kaggle s'ils manquent. Tu peux aussi installer manuellement :

```bash
pip install -r requirements-kaggle-ui.txt
```

## Comportement de la file

Les jobs Kaggle sont volontairement exécutés **un par un**. Cela évite les collisions de session GPU et rend l'utilisation plus fiable depuis un smartphone.

## Tâches actuelles

### Image

Le worker Kaggle utilise :
- Qwen-Image 2.1 GGUF ;
- le text encoder Heretic Q4_K_M ;
- stable-diffusion.cpp CUDA ;
- fallback vers compilation locale si le runtime précompilé n'est pas récupérable.

### Face swap vidéo

Le worker Kaggle utilise FaceFusion Ultra avec :
- face swapper ;
- expression restorer ;
- face enhancer ;
- masques occlusion + region ;
- encodage vidéo haute qualité.


## Sécurité de l'interface

Quand `QWEN_KAGGLE_SHARE=true`, l'URL Gradio publique est protégée par un utilisateur/mot de passe. Si tu ne définis pas `QWEN_UI_PASSWORD`, un mot de passe aléatoire est généré et affiché dans la cellule de lancement. Cela évite qu'une personne ayant récupéré l'URL consomme ton quota Kaggle.
