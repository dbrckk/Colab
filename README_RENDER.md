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

## Sans stockage externe activé (Render Free)
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


## Activer un stockage externe Supabase privé

L'URL Render ne change pas. Pour garder la base SQLite, les médias et les
sources après un redémarrage, il faut créer un projet Supabase dédié à Qwen
Kaggle Studio (ne pas réutiliser les projets d'autres applications).

1. Créer un projet Supabase, de préférence dans une région européenne.
2. Dans Supabase > Project Settings > API Keys, récupérer l'URL du projet
   et la clé serveur secrète. Ne jamais les publier dans GitHub ni dans
   les champs publics de l'interface.
3. Dans Render > Qwen Kaggle Studio > Environment, ajouter comme **secrets** :
   - QWEN_SUPABASE_URL = https://PROJECT-REF.supabase.co
   - QWEN_SUPABASE_SERVICE_ROLE_KEY = la clé serveur du projet
   - QWEN_SUPABASE_BUCKET = qwen-studio-private (facultatif)
4. Redéployer volontairement le service Render.
5. L'interface doit alors indiquer « Stockage externe privé actif (Supabase) ».

Au démarrage, le backend crée son bucket privé si nécessaire, refuse un bucket
public et restaure SQLite avant le lancement des tâches. Chaque transaction
SQLite est sauvegardée en snapshot compressé. Les médias et sources sont
transférés en blocs de 8 Mio, vérifiés par SHA-256 ; le manifeste n'est
publié qu'après transfert complet. Les médias sont restaurés à la demande.

**Sécurité :** la clé serveur Supabase contourne les règles RLS et ne doit
jamais être envoyée à un navigateur. Le programme l'utilise uniquement côté
serveur. En cas de panne du stockage, l'application bloque explicitement les
mutations au lieu d'affirmer à tort que le travail est sauvegardé.

**Contraintes :** cette méthode suppose un seul contrôleur en écriture.
N'utilise pas plusieurs réplicas Render sur le même bucket. Le stockage gratuit
Supabase est limité à environ 1 Go ; chaque objet est limité à 50 Mo et les
projets Free peuvent être suspendus en cas d'inactivité prolongée. Le
découpage évite la limite par objet, pas le quota total. L'URL Render reste
fixe même lorsque le service gratuit est endormi, mais l'uptime 24/7 n'est
pas garanti.

La sauvegarde externe n'est **pas active** tant que les deux secrets Supabase
ne sont pas configurés. Les anciennes données déjà perdues lors d'un
redémarrage ne peuvent pas être reconstituées automatiquement.
