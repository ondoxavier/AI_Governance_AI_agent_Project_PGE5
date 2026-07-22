# Agent IA de gouvernance et conformité IA (UE / US / UK)

Agent de recherche et d'analyse destiné à aider une équipe conformité à évaluer rapidement le niveau de risque d'un cas d'usage IA, et à comparer le traitement réglementaire de ce cas d'usage entre trois juridictions : **Union européenne** (AI Act), **États-Unis** (cadre fédéral volontaire + loi d'État), **Royaume-Uni** (principes non contraignants + régulateurs sectoriels).

Le projet est structuré pour le devoir PGE5 : recherche hybride, garde-fous, raisonnement structuré, serveur MCP, agent critique, observabilité et tests de sécurité.

## Juridictions couvertes

| Juridiction | Nature | Classification | Corpus |
|---|---|---|---|
| **Union européenne** (principale) | Réglementation horizontale **obligatoire** | Niveaux de risque (interdit / haut / limité / minimal) | `data/ai_act_corpus/`, `data/gdpr_corpus/` |
| **États-Unis** | Cadre principalement **volontaire** et sectoriel, sauf exception d'État | Gestion des risques (Govern/Map/Measure/Manage) | `data/us_ai_regulation_corpus/` |
| **Royaume-Uni** | Approche par **principes non contraignants** + régulateurs existants | Contextuelle et sectorielle | `data/uk_ai_regulation_corpus/` |

Détail des documents et de leur statut (obligatoire / volontaire / recommandation) dans le `README.md` de chaque sous-dossier de `data/`.

> **Règle de conception impérative** : l'agent ne doit jamais présenter une comparaison entre juridictions comme une vérité juridique universelle. Chaque affirmation produite doit préciser sa source, sa date et son statut (obligatoire, volontaire, projet ou recommandation) — voir `src/reasoning.py`.

> **État actuel** : le pipeline d'ingestion (`src/ingest.py`) extrait les PDF des 3 juridictions et construit l'index hybride (~3 000 chunks parent-enfant). `src/retrieval.py` interroge cet index (BM25 + dense + RRF + cross-encoder) avec filtre par juridiction, et bascule automatiquement sur un petit corpus de démonstration si l'index n'a pas encore été construit.

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

## Construire l'index RAG (une fois après le clone)

```bash
python src/ingest.py
```

Ce script : (1) extrait le texte des PDF de `data/*/` vers `processed_txt/` (avec cache — relance rapide), (2) découpe selon la structure juridique (Article/Chapitre/Considérant) en chunks parent-enfant, (3) calcule les embeddings (`all-MiniLM-L6-v2`) et écrit l'index dans `index_data/` (git-ignoré, régénérable).

Sans cet index, l'agent fonctionne quand même en **mode dégradé** (petit corpus de démonstration intégré, méthode marquée `fallback-demo` dans les résultats) — utile pour vérifier l'installation, insuffisant pour de vraies réponses.

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

## Evaluation et observabilite

Evaluation locale sur 10 questions UE / US / UK :

```bash
python src/evaluate.py
```

Le script ecrit les resultats dans :

```text
evaluation/latest_results.json
```

Chaque execution de l'agent exporte aussi une trace locale compatible avec une structure Langfuse :

```text
observability/latest_trace.jsonl
```

La trace contient au minimum les spans `agent`, `guardrails.l1`, `tool.hybrid_search`,
`llm.synthesis.self_consistency_k3` et `agent.critic`, avec `AGENT_VERSION` et duree.

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
