# Interface de test — Test du retrieval

Petit serveur Flask qui branche la maquette Stitch sur le vrai `hybrid_search()` du projet.
Aucune donnée statique : chaque recherche interroge réellement l'index 3-juridictions
(ou le corpus de secours si `index_data/` n'a pas été construit — visible dans le badge
« index: réel » / « index: fallback-demo » en haut de page).

## Installation

```bash
pip install -r ui/requirements.txt
```

(Nécessite que `src/retrieval.py` et son index — voir `python src/ingest.py` — soient
disponibles ; sinon l'interface tourne quand même en mode dégradé, étiqueté comme tel.)

## Lancer

```bash
python ui/app.py
```

Puis ouvrir http://127.0.0.1:5050

## Ce que ça expose

- **Recherche en direct** : requête + filtre juridiction (Toutes/EU/US/UK) + mode
  (baseline/hybride/complet) + Top-K → appelle `POST /api/search`, qui appelle
  directement `retrieval.hybrid_search(...)`.
- **Ré-évaluation à la demande** : le bouton « Relancer l'éval » appelle
  `GET /api/eval?mode=...`, qui exécute le jeu de 12 questions gold de
  `tests/eval_retrieval.py` (scoring tolérant) pour le mode actuellement sélectionné.
- Chaque carte affiche la **méthode réelle** utilisée (`bm25+dense+rrf+cross-encoder`
  ou `(fallback-demo)`) et le **contexte parent** complet (bloc juridique entier,
  pas seulement le chunk enfant retrouvé).

## Limites connues

- Le premier appel (recherche ou éval) est lent (~20s) : chargement des modèles
  sentence-transformers/cross-encoder en mémoire. Les appels suivants sont rapides
  (~2s), le modèle reste chargé tant que le serveur tourne (singleton `_IndexRetriever`).
- Pas d'authentification — outil de test interne, ne pas exposer publiquement tel quel.
