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

La trace contient au minimum les spans `agent.run`, `guardrail.l1`,
`guardrail.l4`, `retrieval.search`, `reasoning.self_consistency`,
`reasoning.synthesis.*` et `critic.review`, avec `AGENT_VERSION` et duree.

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

## MCP Server

Le serveur MCP expose les outils du projet a un client compatible MCP tout en
gardant les fonctions importables directement par les tests Python.

Lancer le serveur :

```bash
python src/mcp_server.py
```

Ouvrir l'Inspector lorsque le SDK et `uv` sont disponibles :

```bash
uv run mcp dev src/mcp_server.py
```

Verifier les commandes disponibles selon l'environnement :

```bash
mcp --help
uv run mcp --help
```

Outils disponibles :

| Outil | Objectif | Sortie |
|---|---|---|
| `hybrid_search` | Recherche hybride dans le corpus avec filtre optionnel `EU`, `US`, `UK` ou `all`. | Passages avec `title`, `text`, `source`, `jurisdiction`, `status`, `score`. |
| `classify_ai_act_risk` | Premiere qualification AI Act fondee sur les passages EU recuperes. | Niveau de risque probable, sources et disclaimer. |
| `security_screen` | Passage L1 sur une entree avant execution d'outil ou raisonnement. | `ok`, texte normalise ou erreur controlee. |
| `compare_jurisdiction` | Compare un meme sujet via trois recherches separees `EU`, `US`, `UK`. | Blocs separes par juridiction, resume des statuts, warnings partiels. |

Exemple local sans client MCP :

```bash
python -c "import sys; sys.path.insert(0, 'src'); from mcp_server import compare_jurisdiction; print(compare_jurisdiction('AI credit decisions'))"
```

Mode offline : si `index_data/` n'existe pas ou si une dependance ML manque, le
retrieval bascule sur le corpus de demonstration integre. Les outils retournent
des erreurs structurees au lieu d'une stack trace brute.

Disclaimer affiche par les outils d'analyse :

```text
Cette analyse est generee par IA et doit etre validee par un juriste avant toute decision.
```

Observabilite : si `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` et
eventuellement `LANGFUSE_HOST` sont definis, les spans sont envoyes a Langfuse.
Sinon, le fallback local imprime les spans `[span:start]` / `[span:end]`.

## Interface web

RegulaAI ajoute une interface React responsive au-dessus du code Python
existant. La CLI et le serveur MCP restent disponibles.

```bash
python -m uvicorn src.web_api:app --reload --port 8000

cd frontend
npm install
npm run dev
```

Vérifications :

```bash
python -m pytest -q
cd frontend
npm run lint
npm run typecheck
npm run test -- --run
npm run build
```

Vite relaie `/api` vers `http://127.0.0.1:8000`. Aucune clé n’est exposée au
navigateur. Voir `docs/frontend-architecture.md`.

### Deploiement Vercel

Le depot contient un point d'entree ASGI dans `api/index.py` et un
`vercel.json` qui construit l'interface Vite, publie `frontend/dist`, achemine
`/api/*` vers FastAPI et renvoie les autres routes vers la SPA.

1. Importer ce depot dans Vercel en conservant la racine du depot comme
   `Root Directory`.
2. Conserver la commande de build et le dossier de sortie de `vercel.json`.
3. Ajouter `DEEPINFRA_API_KEY` dans les variables d'environnement Vercel.
4. Ajouter facultativement les variables `LANGFUSE_*` et `AGENT_VERSION`.
5. Deployer, puis verifier `/api/v1/health`.

Les analyses web sont executees dans la requete HTTP, puis conservees dans la
session du navigateur. Elles ne dependent donc pas d'un thread d'arriere-plan
ou de la memoire d'une instance serverless. Les dependances ML lourdes ne sont
pas installees sur Vercel : le retrieval web utilise le mode fallback leger.
Pour construire l'index vectoriel localement :

```bash
python -m pip install -r requirements-ml.txt
python src/ingest.py
```

## Variables d'environnement

Voir `.env.example`. Les clés sont optionnelles pour l'exécution locale de démonstration, mais nécessaires pour une version connectée à un LLM et à Langfuse.
