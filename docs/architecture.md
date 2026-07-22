# Architecture du projet

```text
src/agent.py
  |
  |-- observability.py
  |     Spans locaux au format JSONL
  |     Noms compatibles Langfuse
  |     Version agent via AGENT_VERSION
  |
  |-- Guardrails L1
  |     Normalisation Unicode
  |     Détection injection
  |     TokenBudget
  |
  |-- Retrieval
  |     Chargement data/*.md et data/*.txt
  |     Découpage parent-enfant
  |     BM25
  |     Embedding local par hachage
  |     Fusion RRF
  |     Reranking lexical
  |
  |-- Reasoning
  |     Prompt PREUVES / ANALYSE / CONCLUSION / CONFIANCE
  |     Self-consistency k = 3
  |     Agent critique
  |
  |-- MCP Server
        hybrid_search
        classify_ai_act_risk
        security_screen

src/evaluate.py
  |
  |-- 10 questions UE / US / UK
  |-- baseline dense top-1
  |-- final BM25 + dense + RRF + rerank
  |-- latence, cout local, appels outils, TokenBudget
  |-- export evaluation/latest_results.json
```

## Composants

`agent.py` orchestre l'exécution : contrôle de l'entrée, récupération documentaire, synthèse, critique et affichage.

`observability.py` centralise les spans. En mode local, les traces sont exportées dans `observability/latest_trace.jsonl`. Les noms de spans reprennent la structure attendue dans Langfuse : agent, garde-fous, outil de recherche, synthèse et critique.

`retrieval.py` implémente la recherche hybride demandée. Le BM25 privilégie la correspondance lexicale, l'embedding local fournit un signal dense déterministe et RRF fusionne les classements. Le reranking final réordonne les documents avant la synthèse.

`guardrails.py` contient le filtre L1 et le contrôle L4. L1 bloque les injections évidentes avant action. L4 autorise ou refuse chaque outil selon une matrice de risque.

`reasoning.py` produit une réponse structurée et applique une self-consistency à trois variantes déterministes.

`mcp_server.py` expose les outils attendus. Si la librairie MCP est installée, le serveur peut être lancé ; sinon les fonctions restent importables et testables localement.

`evaluate.py` produit les mesures utilisées dans le rapport. Le mode baseline correspond à une recherche dense top-1 sans RRF ni reranking. La version finale utilise le pipeline hybride et mesure aussi la latence, les appels d'outils et le déclenchement du `TokenBudget`.
