# Architecture de l’interface RegulaAI

```text
React → /api/v1 → FastAPI → run_agent / MCP_TOOL_REGISTRY
                           ↘ evaluation JSON / trace JSONL
```

FastAPI est une couche d’adaptation : la CLI, `run_agent`, MCP, le retrieval,
le raisonnement et les garde-fous restent les sources de vérité. L’agent reste
synchrone ; un `ThreadPoolExecutor` limité à deux workers l’isole des requêtes
HTTP. Les jobs UUID expirent après une heure. Redis/Celery serait nécessaire en
production distribuée.

Le tracer web enveloppe le tracer existant et transmet les vrais débuts et fins
de spans au SSE. Aucune progression n’est simulée. Pydantic valide les entrées,
Zod les réponses importantes, et l’invocation est limitée à
`MCP_TOOL_REGISTRY`. Les chemins de fichiers sont constants et les métadonnées
sensibles filtrées.

Les routes couvrent l’analyse, la comparaison, les preuves, MCP, les traces,
l’évaluation, l’architecture et les préférences locales. Le shell propose skip
link, labels, `aria-live`, focus visible, réduction des animations et
disclaimer permanent, y compris à l’impression. Le fallback et les données
partielles sont affichés explicitement.
