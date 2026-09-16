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

Pour Gemini, installer l'extra puis définir les variables documentées dans `.env.example` :

```bash
uv sync --frozen --extra dev --extra gemini
TEXT_TO_SQL_LLM_PROVIDER=gemini TEXT_TO_SQL_GEMINI_API_KEY=... uv run text-to-sql serve
```

Docker exécute le même service : `docker build -t text-to-sql .` puis
`docker run --rm -p 8000:8000 text-to-sql`.

## Résultats

Le harnais compare les jeux de résultats avec conservation des doublons. L'ordre des lignes ne
compte que lorsque la référence contient `ORDER BY`; l'ordre des colonnes est normalisé. Les
intervalles d'exactitude et les écarts d'ablation sont obtenus par bootstrap apparié à graine
fixe. Les quatre configurations prévues sont : appel direct, résolution seule, réparation seule,
et système complet.

Le [rapport versionné](artifacts/evaluation-report.json) indique actuellement `not_run` : aucun
chiffre BIRD n'est inventé avant la passe finale sur au moins 300 exemples. Une fois le jeu BIRD
obtenu, son manifeste immuable se crée avec :

```bash
uv run text-to-sql freeze-split mini_dev.json artifacts/mini-dev-manifest.json
```

Une passe de développement peut ensuite exercer le runner complet; la passe finale ajoute des
garde-fous contre un petit échantillon, le faux fournisseur et l'écrasement du rapport :

```bash
uv run text-to-sql evaluate mini_dev.json databases artifacts/dev-run
TEXT_TO_SQL_LLM_PROVIDER=gemini TEXT_TO_SQL_GEMINI_API_KEY=... \
  uv run text-to-sql evaluate mini_dev.json databases artifacts/final-run --final
```

## Limites connues

BIRD ne représente pas un entrepôt d'entreprise et l'exactitude d'exécution peut déclarer égales
deux requêtes sémantiquement différentes sur une base donnée. La résolution est lexicale : elle
ne couvre ni les synonymes absents des valeurs d'exemple, ni les schémas testés à mille tables.
Le comparateur normalise l'ordre des colonnes, choix adapté au benchmark mais trop permissif si
leur identité métier importe. Enfin, ce dépôt ne publiera ni score, ni coût, ni latence p95 avant
une passe Gemini gelée et complète; les métriques Prometheus du service ne remplacent pas ce
rapport d'évaluation.
