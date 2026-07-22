# Architecture du projet

```text
src/agent.py
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
```

## Composants

`agent.py` orchestre l'exécution : contrôle de l'entrée, récupération documentaire, synthèse, critique et affichage.

`retrieval.py` implémente la recherche hybride demandée. Le BM25 privilégie la correspondance lexicale, l'embedding local fournit un signal dense déterministe et RRF fusionne les classements. Le reranking final réordonne les documents avant la synthèse.

`guardrails.py` contient le filtre L1 et le contrôle L4. L1 bloque les injections évidentes avant action. L4 autorise ou refuse chaque outil selon une matrice de risque.

`reasoning.py` produit une réponse structurée et applique une self-consistency à trois variantes déterministes.

`mcp_server.py` expose les outils attendus. Si la librairie MCP est installée, le serveur peut être lancé ; sinon les fonctions restent importables et testables localement.
