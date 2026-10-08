# Qwen Kaggle Studio — site Render

## URL
- Site : https://qwen-kaggle-studio.onrender.com/
- Tableau de bord : https://dashboard.render.com/web/srv-db41tf7lot8c73cdu3p0
- Branche de déploiement : `deploy/qwen-render-free`
- Plan : Render **Free**, région Francfort
- Version Colab stable séparée : `stable/qwen-kaggle-v1`

## Accès
Le site exige le nom d'utilisateur `QWEN_UI_USER` et le mot de passe
`QWEN_UI_PASSWORD` définis comme variables d'environnement dans Render.
**Ne jamais mettre le mot de passe ou les secrets Kaggle dans GitHub.**
Le lanceur ne journalise pas le mot de passe en mode hébergé.

## Activer Kaggle une fois pour toutes
Dans Render → Qwen Kaggle Studio → Environment, définir comme secrets :
- `KAGGLE_USERNAME`
- `KAGGLE_API_TOKEN` (ou `KAGGLE_KEY` pour l'ancien mode)

Cela évite de devoir renseigner les secrets dans Gradio après chaque mise en veille.
Les identifiants n'ont pas été transférés depuis Colab.

## Limite de l'offre gratuite
L'URL HTTPS reste identique, mais le serveur peut s'endormir après 15 minutes
sans trafic. Le prochain visiteur le réveille. Render Free ne possède pas de
disque persistant : SQLite, fichiers médias et secrets sauvegardés localement
disparaissent après une mise en veille, un redémarrage ou un déploiement.

**Télécharger les résultats rapidement.** Pour une conservation garantie :
utiliser un volume Render (service payant) ou intégrer un stockage externe et
une base de données persistante. Ne pas annoncer le stockage local de Render
Free comme durable.

Les jobs Kaggle peuvent aussi être interrompus côté contrôleur lorsque Render
met le service en veille ; les jobs privés sur Kaggle restent soumis aux
règles, quotas et délais de Kaggle.

## Déploiement
Les déploiements automatiques sont désactivés pour empêcher qu'une modification
de `main` remplace le site. Déployer volontairement les commits de
`deploy/qwen-render-free` depuis le tableau de bord Render.

Commande build : `pip install -r requirements-kaggle-ui.txt`

Commande start : `python launch_kaggle_ui.py`

Le programme écoute le port défini par Render et exige un mot de passe quand
`PORT` est présent. Il ne crée alors aucun tunnel Gradio public secondaire.
