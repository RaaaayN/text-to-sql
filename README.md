# Texte vers SQL

## Ce que fait le système — et ce qu'il ne fait pas

Cet assistant mono-tour sélectionne les tables et colonnes SQLite pertinentes, demande une
requête structurée à Gemini, vérifie sa syntaxe avec `sqlglot`, puis l'exécute sur une connexion
en lecture seule. Une erreur SQLite peut déclencher au plus deux réparations. Les requêtes sont
plafonnées en lignes et en temps; une question hors schéma ou un SQL dangereux produit un refus
explicite. Le service expose le coût, les tokens, la latence et les refus au format Prometheus.

Le périmètre est volontairement étroit : SQLite/BIRD, une question à la fois, aucun fine-tuning,
agent, interface web, graphique, recherche vectorielle ou second fournisseur de modèle. Le mode
`fake` par défaut sert uniquement au test hors ligne de la base de démonstration.

## Lancer

Prérequis : Python 3.11+ et [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
uv run text-to-sql init-demo
uv run text-to-sql serve
```

Dans un autre terminal :

```bash
curl -s http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"How many products are there?"}'
curl -s http://127.0.0.1:8000/metrics
```

Le parcours exact du README, sans serveur ni réseau, est vérifié par :

```bash
uv run python scripts/readme_smoke.py
```

Pour un seul appel Gemini réellement mesuré avant une évaluation coûteuse :

```bash
TEXT_TO_SQL_LLM_PROVIDER=gemini \
TEXT_TO_SQL_MODEL_NAME=gemini-3.1-flash-lite \
TEXT_TO_SQL_API_KEY=... \
uv run --extra gemini text-to-sql smoke
```

La sortie JSON contient les tokens fournisseur, les tokens de réflexion facturables, la latence
du modèle, la latence totale et le coût estimé au tarif standard publié.

Pour Gemini, installer l'extra puis définir les variables documentées dans `.env.example` :

```bash
uv sync --frozen --extra dev --extra gemini
TEXT_TO_SQL_LLM_PROVIDER=gemini TEXT_TO_SQL_API_KEY=... uv run text-to-sql serve
```

Docker exécute le même service : `docker build -t text-to-sql .` puis
`docker run --rm -p 8000:8000 text-to-sql`.

## Résultats

Le harnais compare les jeux de résultats avec conservation des doublons. L'ordre des lignes ne
compte que lorsque la référence contient `ORDER BY`; l'ordre des colonnes est normalisé. Les
intervalles d'exactitude et les écarts d'ablation sont obtenus par bootstrap apparié à graine
fixe. Les quatre configurations prévues sont : appel direct, résolution seule, réparation seule,
et système complet.

La passe finale a été exécutée sur les 500 exemples SQLite de BIRD Mini-Dev avec
`gemini-3.1-flash-lite`, `temperature=0` et 10 000 rééchantillonnages bootstrap :

| Configuration | Exactitude d'exécution (IC95) | Écart vs direct (IC95) | Coût moyen | Latence p95 |
|---|---:|---:|---:|---:|
| Appel direct | 56,0 % [51,6 ; 60,4] | référence | 0,000960 $ | 1,37 s |
| Résolution seule | 42,6 % [38,4 ; 47,0] | −13,4 pts [−17,0 ; −9,8] | 0,000604 $ | 1,32 s |
| Réparation seule | 56,8 % [52,4 ; 61,2] | +0,8 pt [+0,2 ; +1,6] | 0,000971 $ | 1,55 s |
| Système complet | 43,6 % [39,2 ; 48,0] | −12,4 pts [−16,0 ; −8,8] | 0,000625 $ | 1,39 s |

La résolution réduit le coût d'environ 35 %, mais dégrade nettement l'exactitude. Le système
complet ne satisfait donc pas le seuil produit d'une amélioration d'au moins 10 points. La boucle
de réparation sauve 1,0 % des cas dans la configuration complète; son gain isolé reste faible.
Le [rapport versionné](artifacts/evaluation-report.json) contient les tokens, refus, tentatives,
coûts, latences et intervalles complets. La passe a consommé 2 027 appels,
5 512 594 tokens d'entrée et 134 812 tokens de
sortie/réflexion, soit **1,5804 $** au [tarif standard Gemini publié](https://ai.google.dev/gemini-api/docs/pricing).
Le manifeste immuable se crée avec :

```bash
uv run text-to-sql freeze-split mini_dev.json artifacts/mini-dev-manifest.json
```

Une passe de développement peut ensuite exercer le runner complet; la passe finale ajoute des
garde-fous contre un petit échantillon, le faux fournisseur et l'écrasement du rapport :

```bash
uv run text-to-sql evaluate mini_dev.json databases artifacts/dev-run
TEXT_TO_SQL_LLM_PROVIDER=gemini TEXT_TO_SQL_API_KEY=... \
  uv run text-to-sql evaluate mini_dev.json databases artifacts/final-run --final
```

## Limites connues

BIRD ne représente pas un entrepôt d'entreprise et l'exactitude d'exécution peut déclarer égales
deux requêtes sémantiquement différentes sur une base donnée. La résolution est lexicale : elle
ne couvre ni les synonymes absents des valeurs d'exemple, ni les schémas testés à mille tables.
Le comparateur normalise l'ordre des colonnes, choix adapté au benchmark mais trop permissif si
leur identité métier importe. La résolution lexicale actuelle est le principal échec mesuré :
elle omet trop souvent des tables ou colonnes utiles. Ces chiffres ne justifient pas encore le
bullet CV cible; une correction doit être réglée sur un jeu de développement séparé avant toute
nouvelle passe Mini-Dev. Les métriques Prometheus ne remplacent pas le rapport d'évaluation.
