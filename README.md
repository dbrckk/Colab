# Qwen Studio — Heretic GGUF v8

Interface Gradio mobile-first pour Qwen-Image 2.1 avec le text encoder Heretic GGUF.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dbrckk/Colab/blob/main/Qwen_Image_2_1_Heretic_GGUF_Gradio_v8_PremiumUX.ipynb)

## Points clés

- Qwen-Image 2.1
- Heretic Q4_K_M
- génération et édition d'image
- face swap image différé au premier usage
- face swap vidéo HQ robuste avec suivi d'identité et protection anti-artefacts
- jobs persistants
- cache Google Drive
- interface Gradio responsive
- images et vidéos affichées uniquement dans Gradio


## v8.6 — Video Face Swap Ultra

- suivi strict du même visage
- protections anti-occlusion et anti-glitch
- préservation partielle de la bouche pendant la parole
- aperçu rapide avant rendu complet
- progression et ETA par frames
- annulation de job
- reprise depuis les frames déjà traitées
- nettoyage automatique des fichiers temporaires
- mode automatique rapide / ultra
- conservation audio au rendu final


### Backend Ultra optionnel

Le mode vidéo propose maintenant :
- `auto`
- `builtin-ultra`
- `facefusion-ultra`

FaceFusion Ultra est installé dans un environnement Python séparé au premier usage afin de ne pas casser les dépendances du notebook principal. Il active le face swapper, le masque d'occlusion/région, l'expression restorer et le face enhancer.


### Améliorations v8.6

- lecture **streaming de la vidéo source** : les frames d'entrée ne sont plus toutes extraites sur disque ;
- checkpoints vidéo en chunks persistants sur Google Drive ;
- création des archives sur le disque local rapide avant copie vers Drive ;
- checksum SHA-256 et manifeste pour détecter un checkpoint corrompu ;
- reprise après perte du runtime à partir des chunks persistants ;
- jobs et fichiers uploadés persistants sur Drive ;
- identifiant de runtime pour éviter de marquer à tort un job actif comme interrompu lors d'une réexécution de cellule ;
- FPS ajusté lorsque `frame stride > 1` pour conserver la durée ;
- fallback audio AAC si le codec original ne peut pas être remuxé ;
- cache persistant lazy du runtime FaceFusion après le premier usage réussi ;
- validation syntaxique/intégration via GitHub Actions.
