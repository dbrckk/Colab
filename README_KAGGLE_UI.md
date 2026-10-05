# Qwen Kaggle Studio

## Lancement recommandé sur smartphone

Ouvre `Qwen_Kaggle_Studio_Controller.ipynb` dans Google Colab et exécute son unique cellule. Le notebook lit automatiquement les **Secrets Colab** `KAGGLE_USERNAME` et `KAGGLE_API_TOKEN` s'ils existent. Le contrôleur monte Google Drive, lance Gradio et utilise Kaggle uniquement pour les tâches GPU. Les résultats sont conservés dans `MyDrive/QwenKaggleStudio/`.


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
   - `image_edit` : image source + instruction → édition Qwen Image 2.1 avec le mmproj Heretic.
   - `video_faceswap` : visage source + vidéo cible → FaceFusion Ultra sur Kaggle.
4. L'UI :
   - crée un job local ;
   - crée un dataset Kaggle privé uniquement lorsqu'un fichier source doit être transféré ;
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

Le launcher installe automatiquement les versions compatibles de Gradio et du CLI Kaggle s'ils manquent ou si une version majeure non testée est présente. La release stable utilise les plages :

- `gradio>=6.0,<7.0`
- `kaggle>=2.2.3,<3.0`

Tu peux aussi installer manuellement :

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


## Optimisations de démarrage

- Les téléchargements Qwen (DiT, VAE, Heretic et mmproj si nécessaire) sont lancés en parallèle sur le worker Kaggle.
- Le runtime `stable-diffusion.cpp` tente d'abord un binaire CUDA précompilé.
- Si le runtime précompilé ne fonctionne pas, le worker installe automatiquement les outils de build manquants puis compile `sd-cli`.


## Génération par lot optimisée

La génération par lot ne crée plus un kernel Kaggle par prompt. Un lot de jusqu'à 20 prompts devient **un seul job Kaggle** :

- Qwen et le runtime sont chargés une seule fois ;
- les images sont générées successivement dans la même session GPU ;
- les résultats sont récupérés ensemble dans la bibliothèque ;
- avec une seed fixe, les seeds sont incrémentées pour chaque image.

## Génération texte → image accélérée

Un job `image` ou `image_batch` sans fichier source **ne crée plus de dataset Kaggle temporaire**. La configuration est injectée directement dans le notebook privé généré. Les datasets privés temporaires ne sont créés que lorsque des fichiers doivent réellement être transférés, par exemple pour `image_edit` ou `video_faceswap`.


## Reprise après coupure ou redémarrage

Le contrôleur conserve l'état des jobs dans SQLite. En cas de coupure Colab, réseau ou Kaggle :

- un kernel déjà confirmé est repris sans recalcul lorsque c'est possible ;
- les outputs terminés peuvent être récupérés sans relancer le GPU ;
- une expiration des identifiants place le job en attente d'authentification au lieu de perdre son état ;
- une soumission interrompue autour de `kernels push` est d'abord vérifiée par son identifiant exact ;
- si Kaggle confirme que le kernel n'existe pas, le job peut être rejoué proprement ;
- si l'état distant est ambigu, aucun second push n'est lancé automatiquement afin d'éviter un double calcul.

Le bouton de récupération peut également relancer cette vérification sans effectuer de nouveau push tant que l'existence distante n'est pas tranchée.

## Validation de la release

La branche `main` est protégée fonctionnellement par le workflow **Validate Qwen Studio**, qui vérifie notamment :

- compilation de tous les modules Python ;
- modules notebook/runtime ;
- smoke tests Gradio ;
- contrôleur Kaggle et reprise après incident ;
- authentification et persistance des secrets ;
- staging dataset, kernel, outputs et manifestes SHA-256 ;
- CLI Kaggle réel et ses options attendues ;
- UI du contrôleur et launcher.
