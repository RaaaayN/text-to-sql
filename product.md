# Fiche produit

Objectif : produire un assistant de requêtage en langage naturel dont la génération SQL est
vérifiée par exécution, et mesurer honnêtement son exactitude, son coût et sa latence sur BIRD.

Le système comprend six briques : introspection multi-tables, résolution lexicale du schéma avec
fermeture par clés étrangères, génération structurée par Gemini, validation et exécution SQLite
en lecture seule, boucle de réparation bornée, puis évaluation par comparaison des résultats.

Critères de publication : au moins 300 exemples gelés, intervalle bootstrap à 95 % d'une largeur
maximale de 10 points, ablation utile d'au moins 10 points, coût moyen et latence p95 mesurés,
rapport versionné et parcours README vérifié en CI.

La passe finale du 17 septembre 2026 sur 500 exemples BIRD Mini-Dev donne 56,0 %
[51,6 ; 60,4] pour l'appel direct et 43,6 % [39,2 ; 48,0] pour le système complet. La
résolution de schéma réduit le coût moyen de 0,000960 $ à 0,000625 $ par question, mais fait
perdre 12,4 points d'exactitude. La réparation seule apporte +0,8 point [0,2 ; 1,6]. Le seuil
d'acceptation d'une ablation positive d'au moins 10 points n'est donc pas atteint.

Hors périmètre : interface web, dialogue multi-tour, fine-tuning, comparaison de fournisseurs,
graphiques, dialectes multiples, recherche vectorielle et framework d'agents.
