# Agent IA de gouvernance et conformité AI Act

Agent de recherche et d'analyse destiné à aider une équipe conformité à évaluer rapidement le niveau de risque d'un cas d'usage IA au regard de l'AI Act européen.

Le projet est structuré pour le devoir PGE5 : recherche hybride, garde-fous, raisonnement structuré, serveur MCP, agent critique, observabilité et tests de sécurité.

## Installation

```bash
git clone votre-depot
cd votre-depot
cp .env.example .env
pip install -r requirements.txt
python src/agent.py
```

Sous Windows PowerShell :

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/agent.py
```

Si l'exécution de scripts PowerShell est bloquée, utilisez directement le Python de l'environnement :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\agent.py
```

## Exécution

Exemple avec la question par défaut :

```bash
python src/agent.py
```

Exemple avec une question personnalisée :

```bash
python src/agent.py "Une banque utilise un modèle IA pour présélectionner des candidats à un crédit. Quel est le niveau de risque AI Act ?"
```

## Tests

```bash
python -m pytest tests/test_security.py
```

## Architecture

```text
Utilisateur
   |
   v
src/agent.py
   |-- guardrails.py  : filtrage L1, contrôle d'action L4, TokenBudget
   |-- retrieval.py   : BM25 + embeddings locaux + RRF + reranking
   |-- reasoning.py   : synthèse PREUVES / ANALYSE / CONCLUSION / CONFIANCE + self-consistency k=3
   |-- mcp_server.py  : outils MCP exposés
   |
   v
data/              : documents de référence du corpus
docs/              : consignes et documentation d'architecture
tests/             : tests d'injection
```

Le projet fonctionne hors ligne avec un petit corpus local. En production, les mêmes interfaces peuvent être branchées sur un LLM, Langfuse et un serveur MCP complet via les variables d'environnement.

## Outils MCP prévus

- `hybrid_search` : recherche documentaire hybride dans le corpus.
- `classify_ai_act_risk` : classification du niveau de risque AI Act.
- `security_screen` : analyse d'une entrée utilisateur par les garde-fous.

## Variables d'environnement

Voir `.env.example`. Les clés sont optionnelles pour l'exécution locale de démonstration, mais nécessaires pour une version connectée à un LLM et à Langfuse.
