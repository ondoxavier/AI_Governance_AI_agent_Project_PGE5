# Données

Placez ici les documents de référence utilisés par l'agent.

Formats pris en charge par le socle actuel :

- `.md`
- `.txt`

Au lancement, `src/retrieval.py` lit les fichiers de ce dossier. Si aucun document métier n'est présent, l'agent utilise un petit corpus de démonstration intégré sur la gouvernance de l'IA et l'AI Act.

## Organisation actuelle — 3 juridictions

```text
data/
  ai_act_corpus/              # UE — règlement AI Act, annexes, guides (16 PDF)
  gdpr_corpus/                # UE — RGPD, articles, avis CNIL/EDPB (16 PDF)
  us_ai_regulation_corpus/    # US — décrets exécutifs 14110/14179/14365,
                               #      NIST AI RMF 1.0 + Playbook + Generative AI Profile,
                               #      Blueprint AI Bill of Rights, Colorado SB24-205 (6+2 PDF)
  uk_ai_regulation_corpus/    # UK — livre blanc pro-innovation (2023) + réponse
                               #      gouvernementale (2024), non contraignants (2 PDF)
  ai_act_reference.md         # résumé synthétique utilisé par le socle offline actuel
```

Chaque sous-dossier a son propre `README.md` détaillant la liste des documents, leur source officielle, leur date et leur **statut** (obligatoire / volontaire / projet / recommandation) — cette dernière information est utilisée par `src/reasoning.py` pour éviter toute généralisation abusive entre juridictions.

⚠️ **Écart connu** : les PDF ci-dessus ne sont pas encore lus par `src/retrieval.py` (qui ne traite que `.md`/`.txt` pour l'instant). L'extraction PDF → texte + chunking parent-enfant sur ce corpus réel est requise avant de pouvoir prétendre à un vrai retrieval hybride (BM25 + dense + RRF + reranking) sur les 3 juridictions.
