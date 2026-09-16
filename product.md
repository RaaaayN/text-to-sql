# Fiche produit

Objectif : produire un assistant de requêtage en langage naturel dont la génération SQL est
vérifiée par exécution, et mesurer honnêtement son exactitude, son coût et sa latence sur BIRD.

Le système comprend six briques : introspection multi-tables, résolution lexicale du schéma avec
fermeture par clés étrangères, génération structurée par Gemini, validation et exécution SQLite
en lecture seule, boucle de réparation bornée, puis évaluation par comparaison des résultats.

Critères de publication : au moins 300 exemples gelés, intervalle bootstrap à 95 % d'une largeur
maximale de 10 points, ablation utile d'au moins 10 points, coût moyen et latence p95 mesurés,
rapport versionné et parcours README vérifié en CI. Les résultats ne sont pas encore exécutés et
ne doivent pas être remplacés par des valeurs estimées.

Hors périmètre : interface web, dialogue multi-tour, fine-tuning, comparaison de fournisseurs,
graphiques, dialectes multiples, recherche vectorielle et framework d'agents.
