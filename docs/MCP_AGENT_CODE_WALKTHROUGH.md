# Guide oral - Partie B MCP & Agent

## A. Resume de mon travail

J'ai finalise l'orchestration de l'agent et le serveur MCP. L'agent applique L1
avant toute recherche, autorise l'action de retrieval via L4, consomme un
TokenBudget, appelle le retrieval hybride, lance la self-consistency du module
reasoning, passe par le critique, puis retourne une reponse structuree avec
sources, confiance, verdict, warnings, trace_id et latence. L'observabilite
utilise Langfuse quand les cles sont presentes et garde un fallback local sinon.

Cote MCP, les trois outils existants sont conserves et un quatrieme outil,
`compare_jurisdiction`, compare un meme sujet en lancant trois recherches
distinctes pour EU, US et UK. Les resultats restent separes et conservent
`source`, `jurisdiction` et `status`, pour eviter de melanger droit obligatoire,
cadres volontaires et recommandations.

## B. Explication de `mcp_server.py`

### `hybrid_search`

Objectif : exposer le retrieval hybride du projet a un client MCP.

Entree : `query`, `top_k`, `jurisdiction`.

Sortie : dictionnaire JSON-serialisable avec `ok`, `query`, `jurisdiction`,
`results`, `warnings`, `disclaimer`, `latency_ms`.

Dependances : `guardrails.l1_filter`, `guardrails.authorize_action`,
`retrieval.hybrid_search`, `observability.create_tracer`.

Gestion des erreurs : query vide, `top_k` invalide, juridiction inconnue,
erreur de retrieval. L'outil retourne une erreur controlee, pas une stack trace.

Securite : L1 filtre l'entree, L4 autorise `hybrid_search`, les secrets ne sont
pas traces.

Performances : `top_k` est borne a 10 et les extraits longs sont tronques dans
la sortie MCP.

### `classify_ai_act_risk`

Objectif : faire une premiere classification AI Act a partir de preuves EU.

Entree : description ou question utilisateur.

Sortie : `risk`, `evidence_count`, `sources`, `disclaimer`, `latency_ms`.

Dependances : retrieval EU puis `reasoning.classify_ai_act_risk`.

Gestion des erreurs : entree invalide, blocage L1, erreur retrieval ou
classification.

Securite : l'outil rappelle le disclaimer de validation humaine. Il ne remplace
pas un juriste.

Performances : top_k fixe par defaut pour limiter le volume.

### `security_screen`

Objectif : fournir un outil MCP de verification L1 avant execution.

Entree : texte utilisateur ou texte externe.

Sortie : `ok`, texte normalise ou erreur controlee.

Dependances : `l1_filter` et `authorize_action`.

Gestion des erreurs : type invalide ou pattern d'injection detecte.

Securite : utile pour tester les prompt injections avant d'appeler un LLM ou un
outil.

Performances : operation locale, sans reseau.

### `compare_jurisdiction`

Objectif : comparer un sujet de gouvernance IA entre EU, US et UK.

Entree : `topic`, `top_k`, `data_dir`.

Sortie : dictionnaire avec un bloc `EU`, un bloc `US`, un bloc `UK`, chacun
contenant `status_summary` et `results`.

Dependances : trois appels separes a `retrieval.hybrid_search`.

Gestion des erreurs : si une juridiction echoue, son bloc contient `error`, mais
les autres juridictions restent disponibles.

Securite : L1 valide le topic, L4 autorise l'outil, le disclaimer juridique est
toujours present.

Performances : appels sequentiels pour eviter de supposer que les caches ML sont
thread-safe. Le volume est limite par `top_k`.

## C. Explication de `agent.py`

Pipeline :

```text
question
  -> validation type / longueur
  -> L1 l1_filter
  -> TokenBudget sur la question
  -> L4 authorize_action("hybrid_search")
  -> retrieval.hybrid_search
  -> TokenBudget sur les contextes
  -> self_consistency(question, contexts, k=3)
  -> critic_review
  -> revision unique si REVISE et budget suffisant
  -> AgentResponse + disclaimer + traces
```

L1 : bloque les prompt injections avant tout retrieval ou reasoning.

Retrieval : appele via la signature gelee `hybrid_search(query, top_k, data_dir,
jurisdiction)`.

Self-consistency : l'agent ne recree pas cette logique. Il appelle la fonction
publique `self_consistency(question, contexts, k=3)`.

Critique : le verdict brut du module reasoning est converti en `APPROVE` ou
`REVISE`, afin d'etre visible dans la sortie et testable.

L4 : autorise explicitement `hybrid_search`. Les outils inconnus restent refuses
par `guardrails.py`.

TokenBudget : consomme la question, les contextes et la reponse finale. En cas
de depassement attendu, l'agent retourne une reponse degradee comprehensible.

Fallback : sans index, `retrieval.py` utilise son corpus integre. Sans Langfuse,
`observability.py` imprime des spans locaux. Sans LLM, le reasoning deterministe
existant continue de fonctionner.

Observabilite : les spans principaux sont `agent.run`, `guardrail.l1`,
`guardrail.l4`, `retrieval.search`, `reasoning.self_consistency`,
`reasoning.synthesis.1`, `reasoning.synthesis.2`, `reasoning.synthesis.3` et
`critic.review`.

## D. Questions probables du professeur

1. Pourquoi utiliser MCP ?
   Pour rendre les outils utilisables par n'importe quel client compatible MCP,
   pas seulement par notre script Python.

2. Quelle difference entre un outil MCP et une fonction Python classique ?
   MCP ajoute un contrat outille : nom, schema d'entree, description et
   execution via serveur.

3. Pourquoi separer EU, US et UK ?
   Parce que les statuts juridiques different : reglement obligatoire,
   framework volontaire ou recommandation.

4. Pourquoi ne pas utiliser `jurisdiction="all"` dans `compare_jurisdiction` ?
   Pour eviter de fusionner les cadres et rendre les trois recherches visibles.

5. Comment evitez-vous le melange des statuts juridiques ?
   Chaque resultat conserve `status` jusqu'a la sortie finale.

6. Comment l'agent se protege-t-il contre une prompt injection ?
   L1 normalise l'entree et bloque les patterns connus avant toute action.

7. A quoi sert L1 ?
   A filtrer l'entree et les textes suspects avant qu'ils atteignent le modele.

8. A quoi sert L4 ?
   A autoriser ou refuser une action outillee selon une matrice de risque.

9. Pourquoi utiliser un TokenBudget ?
   Pour eviter une explosion de cout ou une boucle qui consomme trop de tokens.

10. Pourquoi self-consistency utilise-t-elle trois appels ?
    Pour comparer plusieurs syntheses et reduire les conclusions fragiles.

11. Comment la confiance est-elle calculee ?
    Dans le composant reasoning. L'agent ne fait qu'extraire une valeur
    numerique quand elle est disponible.

12. Pourquoi un critique est-il necessaire ?
    Il verifie la presence de preuves et signale les conclusions incomplètes.

13. Pourquoi limiter la revision a une seule tentative ?
    Pour eviter une boucle infinie si le critique retourne toujours `REVISE`.

14. Que se passe-t-il sans cle LLM ?
    Le reasoning deterministe actuel continue de produire une synthese.

15. Que se passe-t-il sans Langfuse ?
    `observability.py` utilise le fallback local et l'agent continue.

16. Pourquoi conserver LocalTracer-compatible fallback ?
    Pour que le correcteur puisse lancer le repo sans compte externe.

17. Comment mesurez-vous la latence ?
    Avec `time.perf_counter()` autour de l'agent, des outils, du retrieval et du
    critique.

18. Pourquoi utiliser `time.perf_counter()` ?
    C'est une horloge monotone adaptee aux durees courtes.

19. Comment testez-vous MCP sans reseau ?
    Les fonctions MCP sont importees directement et `run_hybrid_search` est
    monkeypatche dans les tests.

20. Pourquoi l'agent ne constitue-t-il pas un conseil juridique definitif ?
    Il structure des sources et affiche un disclaimer de validation humaine.

21. Comment empechez-vous une boucle infinie d'outils ?
    Pas de boucle d'outils autonome et une seule revision critique maximum.

22. Comment l'outil reagit-il lorsqu'une juridiction echoue ?
    Il met l'erreur dans le bloc de cette juridiction et garde les autres.

23. Pourquoi les fonctions publiques sont-elles gelees ?
    Pour permettre aux autres membres de travailler sans casser les imports.

24. Quels sont les principaux compromis de performance ?
    Appels EU/US/UK sequentiels, `top_k` borne, snippets tronques en sortie MCP.

25. Quelle difference entre obligatoire, volontaire et recommandation ?
    Obligatoire cree une contrainte juridique, volontaire guide les pratiques,
    recommandation oriente sans imposer directement.

## E. Lecture commentee du code

### Tracing

```python
active_tracer = tracer or create_tracer("agent.run", {...})
with active_tracer.span("retrieval.search", {...}):
    contexts = hybrid_search(...)
```

Decision : le tracer est injecte ou cree localement. Cela rend les tests faciles
et evite de bloquer l'agent si Langfuse n'est pas configure.

### L4 avant retrieval

```python
with active_tracer.span("guardrail.l4", {"tool_name": "hybrid_search"}):
    authorize_action("hybrid_search")
```

Decision : meme une action peu risquee passe par L4 pour que le controle soit
auditable.

### Revision limitee

```python
if critic_code == "REVISE" and budget.remaining > 50 and contexts:
    revised = self_consistency(..., k=self_consistency_k)
```

Decision : une seule revision evite les boucles infinies et garde le cout sous
controle.

### Comparaison separee

```python
for jurisdiction in ("EU", "US", "UK"):
    results = run_hybrid_search(..., jurisdiction=jurisdiction)
```

Decision : l'ordre est deterministe et les statuts ne sont jamais fusionnes
entre juridictions.

### Erreurs partielles

```python
except Exception as exc:
    jurisdiction_blocks[jurisdiction] = _empty_jurisdiction_block(error=message)
```

Decision : une panne US ne doit pas supprimer les preuves EU et UK deja
disponibles.
