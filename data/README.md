# Données

Placez ici les documents de référence utilisés par l'agent.

Formats pris en charge par le socle actuel :

- `.md`
- `.txt`

Au lancement, `src/retrieval.py` lit les fichiers de ce dossier. Si aucun document métier n'est présent, l'agent utilise un petit corpus de démonstration intégré sur la gouvernance de l'IA et l'AI Act.

Organisation recommandée :

```text
data/
  ai_act/
    obligations.md
    risk_levels.md
  internal_policy/
    model_governance.md
```
