# Architecture du projet

```text
User / CLI / MCP client
  |
  v
src/agent.py
  |
  |-- Observability
  |     Langfuse si configure
  |     fallback local sinon
  |
  |-- Guardrail L1
  |     validation type/longueur
  |     normalisation Unicode via guardrails.py
  |     detection prompt injection
  |
  |-- Guardrail L4
  |     autorisation explicite des actions outillees
  |     refus des actions inconnues ou sensibles
  |
  |-- Retrieval / MCP Tool
  |     hybrid_search(query, top_k, data_dir, jurisdiction, mode)
  |     baseline dense-only
  |     BM25 + dense + RRF + reranking
  |     fallback corpus local si index_data/ absent
  |
  |-- Reasoning
  |     self_consistency(question, contexts, k=3)
  |     format PREUVES / ANALYSE / CONCLUSION / CONFIANCE
  |
  |-- Critic
  |     critic_review(...)
  |     verdict visible APPROVE / REVISE
  |     une revision maximum si budget disponible
  |
  v
AgentResponse
  answer
  conclusion
  confidence
  critic_verdict
  sources
  warnings
  trace_id
  latency_ms

src/evaluate.py
  |
  |-- 10 questions UE / US / UK
  |-- baseline dense top-1
  |-- final BM25 + dense + RRF + rerank
  |-- latence, cout local, appels outils, TokenBudget
  |-- export evaluation/latest_results.json
```

## Serveur MCP

`src/mcp_server.py` expose quatre outils. Les fonctions restent importables sans
serveur distant afin que les tests automatises ne dependent pas du reseau.

```text
MCP client
  |
  v
FastMCP("ai-governance-agent")
  |
  |-- hybrid_search
  |     recherche hybride avec filtre optionnel EU / US / UK / all
  |
  |-- classify_ai_act_risk
  |     recupere des sources EU puis appelle reasoning.classify_ai_act_risk
  |
  |-- security_screen
  |     applique L1 avant execution d'outils ou raisonnement
  |
  |-- compare_jurisdiction
        appelle hybrid_search trois fois :
        1. jurisdiction="EU"
        2. jurisdiction="US"
        3. jurisdiction="UK"
        conserve source, jurisdiction et status dans chaque bloc
```

Le choix de ne pas paralleliser `compare_jurisdiction` est volontaire : le
retrieval peut charger des modeles ou caches partages (`sentence-transformers`,
cross-encoder). Les trois appels sequentiels gardent un ordre deterministe.

## Composants

`agent.py` orchestre l'execution : validation de l'entree, L1, L4, recherche
documentaire, self-consistency, critique, budget de tokens, rendu final et
observabilite. Il retourne une `AgentResponse` testable et la CLI imprime une
version lisible pour le professeur.

`observability.py` fournit un tracer commun. Si les variables Langfuse sont
presentes et que le package est installe, les spans sont envoyes a Langfuse.
Sinon, les memes noms de spans sont imprimes localement et conserves en memoire.
L'absence de Langfuse ne casse jamais l'import ou l'execution.

`retrieval.py` implemente la recherche hybride demandee. En mode production,
l'index genere par `src/ingest.py` fournit les chunks parent-enfant avec
metadata juridique. En mode fallback, les fichiers Markdown de `data/` et des
README de corpus permettent de lancer l'agent depuis un clone vierge.

`guardrails.py` contient le filtre L1, la matrice L4 et `TokenBudget`. Les tests
de securite verifient les prompt injections, les actions inconnues/sensibles et
les depassements de budget.

`reasoning.py` produit la synthese structuree et l'agent critique deterministe.

`evaluate.py` produit les mesures utilisees dans le rapport. Le mode baseline
correspond a une recherche dense top-1 sans RRF ni reranking. La version finale
utilise le pipeline hybride et mesure aussi la latence, les appels d'outils et
le declenchement du `TokenBudget`.

Chaque reponse d'analyse reglementaire contient le disclaimer :

```text
Cette analyse est generee par IA et doit etre validee par un juriste avant toute decision.
```
