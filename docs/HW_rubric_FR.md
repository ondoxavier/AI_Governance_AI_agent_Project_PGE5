# 📊 Grille d’évaluation — Projet de devoir

> **Disponible dès maintenant — à lire avant de commencer.**
> Total : 100 points. Aucun point attribué par les pairs. Aucune présentation orale. La note repose entièrement sur le dépôt et le rapport soumis.

---

## Répartition des points

| Composante | Points |
|------------|--------|
| Implémentation technique | 50 |
| Évaluation et mesures | 20 |
| Qualité du rapport | 20 |
| Transparence sur l’utilisation de l’IA | 10 |
| **Total** | **100** |

---

## Implémentation technique — 50 points

### A. Pipeline de recherche documentaire — `15 points`

| Score | Description |
|-------|-------------|
| 13–15 | Recherche hybride — BM25 + dense + RRF — implémentée et fonctionnelle. Reranking par cross-encoder appliqué. Découpage parent-enfant utilisé. RAGAS montre une amélioration mesurable par rapport à une recherche de base. |
| 9–12 | Recherche hybride implémentée. Reranking présent. Quelques problèmes mineurs, par exemple : la fusion RRF fonctionne, mais le découpage parent-enfant n’est pas implémenté. |
| 5–8 | Une des trois techniques — recherche hybride, reranking ou parent-enfant — est implémentée. Une recherche basique est utilisée pour les autres éléments. |
| 1–4 | Uniquement une similarité cosinus top-k basique. Aucune technique avancée de recherche issue du Bloc 1. |
| 0 | Aucun pipeline de recherche documentaire. L’agent répond uniquement à partir des connaissances internes du modèle. |

### B. Serveur MCP — `10 points`

| Score | Description |
|-------|-------------|
| 9–10 | Serveur MCP comportant au moins trois outils. Tous les outils disposent de docstrings complètes — Quand l’utiliser / Quand ne pas l’utiliser / Valeur retournée / Exemple. Tous les outils gèrent correctement les erreurs, sans exception non interceptée. Les tests avec MCP Inspector réussissent. |
| 6–8 | Serveur MCP comportant au moins deux outils. Des docstrings sont présentes, mais incomplètes. La plupart des outils disposent d’une gestion des erreurs. |
| 3–5 | Un serveur MCP existe, mais les outils ont des descriptions minimales ou une gestion des erreurs peu fiable. |
| 1–2 | Uniquement un squelette de serveur MCP : les outils ne fonctionnent pas réellement. |
| 0 | Aucun serveur MCP. Les outils sont uniquement définis comme de simples fonctions Python. |

### C. Pile de sécurité — `10 points`

| Score | Description |
|-------|-------------|
| 9–10 | Filtre d’entrée L1 comportant des motifs d’injection et une normalisation Unicode. Contrôle d’action L4 avec une `ACTION_RISK_MATRIX` couvrant tous les outils. `TokenBudget` intégré. Les cinq tests d’injection réussissent. Les tests sont inclus dans `tests/test_security.py`. |
| 6–8 | Les couches L1 et L4 sont présentes. Quatre tests d’injection sur cinq réussissent. `TokenBudget` est présent. |
| 3–5 | Une seule des couches L1 ou L4 est implémentée. Moins de quatre tests d’injection réussissent. |
| 1–2 | La sécurité est mentionnée dans le rapport, mais son implémentation dans le code est minimale. |
| 0 | Aucune implémentation de sécurité. Les tests d’injection n’ont pas été exécutés. |

### D. Stratégie de raisonnement — `10 points`

| Score | Description |
|-------|-------------|
| 9–10 | CoT few-shot avec le format PREUVES / ANALYSE / CONCLUSION / CONFIANCE dans le prompt système de synthèse. Self-Consistency avec k ≥ 3 lors de l’étape finale de synthèse. Étiquetage du niveau de confiance utilisé dans l’ensemble du processus. |
| 6–8 | CoT zero-shot utilisé. Étiquetage du niveau de confiance présent. Self-Consistency implémentée, mais avec k = 1 ou non connectée à l’étape de synthèse. |
| 3–5 | La consigne « raisonner étape par étape » est ajoutée. Aucun étiquetage du niveau de confiance. Aucune Self-Consistency. |
| 1–2 | Aucun CoT. Utilisation uniquement de prompts directs. |

### E. Observabilité — `5 points`

| Score | Description |
|-------|-------------|
| 5 | Des traces Langfuse sont visibles pour toutes les exécutions. L’agent, les appels LLM et les appels d’outils disposent chacun de leur propre span. La version de l’agent est journalisée. Au moins une alerte de surveillance est décrite. |
| 3–4 | Des traces Langfuse sont visibles. La plupart des spans sont présents. La version de l’agent n’est pas journalisée. |
| 1–2 | Langfuse est connecté, mais l’instrumentation est minimale — uniquement une trace globale, sans spans pour les outils. |
| 0 | Aucune observabilité. |

---

## Évaluation et mesures — 20 points

### F. Baseline RAGAS et amélioration — `12 points`

| Score | Description |
|-------|-------------|
| 11–12 | RAGAS exécuté sur au moins dix questions. Les quatre métriques sont rapportées : `context_recall`, `context_precision`, `faithfulness` et `answer_relevancy`. La baseline avant les améliorations du Bloc 1 est documentée. Les scores finaux montrent une amélioration mesurable. Chaque amélioration est reliée à la technique qui l’a provoquée. |
| 8–10 | RAGAS exécuté sur au moins cinq questions. Trois ou quatre métriques sont rapportées. Une baseline est présente. Les améliorations sont documentées, mais elles ne sont pas toutes expliquées. |
| 5–7 | RAGAS est exécuté, mais sur moins de cinq questions. La baseline est absente ou incomplète. Certaines métriques sont rapportées. |
| 2–4 | RAGAS est mentionné, mais exécuté une seule fois. Aucune comparaison avec une baseline. |
| 0–1 | RAGAS n’est pas exécuté. Aucune évaluation quantitative de la qualité de la recherche documentaire. |

### G. Mesure du coût et de la latence — `8 points`

| Score | Description |
|-------|-------------|
| 7–8 | Le coût moyen par exécution en dollars américains est calculé sur au moins dix exécutions. La latence moyenne en secondes est indiquée. La répartition des appels d’outils est rapportée — nombre d’appels pour chaque outil. `TokenBudget` est déclenché au moins une fois pendant les tests et cet événement est documenté. |
| 4–6 | Le coût et la latence sont rapportés. Moins de dix exécutions ont été réalisées. La répartition des appels d’outils n’est pas indiquée. |
| 1–3 | Seul le coût ou seule la latence est indiqué. Les tests sont limités. |
| 0 | Aucune mesure du coût ou de la latence. |

---

## Qualité du rapport — 20 points

### H. Présentation du problème et architecture — `8 points`

| Score | Description |
|-------|-------------|
| 7–8 | La présentation du problème désigne un utilisateur précis et un scénario précis. Le diagramme d’architecture correspond exactement au code exécuté. Une décision de conception est expliquée avec son compromis. |
| 5–6 | La présentation du problème est précise. Le diagramme correspond globalement au code. Les décisions de conception sont décrites, mais pas justifiées. |
| 3–4 | Présentation vague du problème — par exemple : « aider les gens à apprendre des choses sur X ». Le diagramme est incomplet ou obsolète. |
| 1–2 | Aucune véritable présentation du problème. Aucun diagramme d’architecture. |

### I. Évaluation au regard de l’AI Act européen — `6 points`

| Score | Description |
|-------|-------------|
| 6 | Le niveau de risque est identifié avec une justification précise faisant référence aux critères de l’AI Act européen. Une obligation est déduite de ce niveau et son implémentation est décrite — par exemple, une information claire de l’utilisateur dans l’interface. |
| 4–5 | Le niveau de risque est identifié. Une justification est présente, mais reste générale. Une obligation est mentionnée, mais non implémentée. |
| 2–3 | Le niveau de risque est identifié sans justification. Aucune obligation n’est décrite. |
| 0–1 | L’AI Act européen n’est pas traité ou l’analyse est manifestement incorrecte — par exemple, le rapport affirme qu’aucune réglementation ne s’applique. |

### J. Limites et prochaines étapes — `6 points`

| Score | Description |
|-------|-------------|
| 5–6 | Au moins deux limites précises sont identifiées, avec les conditions dans lesquelles elles apparaîtraient. La section « prochaines étapes » est techniquement concrète et nomme une technique précise, au lieu de simplement indiquer « améliorer l’agent ». |
| 3–4 | Les limites sont listées, mais restent générales — par exemple : « l’agent pourrait être plus précis ». |
| 1–2 | Une seule phrase est consacrée aux limites. Aucun détail précis. |
| 0 | Aucune section sur les limites. |

---

## Transparence sur l’utilisation de l’IA — 10 points

### K. Tableau de déclaration et maîtrise du code — `10 points`

| Score | Description |
|-------|-------------|
| 9–10 | Le tableau d’utilisation de l’IA est rempli de manière honnête et précise. Le rapport explique ce qui a été écrit, généré ou modifié. Chaque fonction de la base de code peut être expliquée par le groupe — cela pourra être vérifié par des questions complémentaires lorsque l’enseignant a un doute. |
| 6–8 | Le tableau est rempli avec un certain niveau de précision. Une ou deux fonctions peuvent être difficiles à expliquer lors d’un questionnement. |
| 3–5 | Le tableau est présent, mais reste vague — par exemple : « nous avons utilisé l’IA pour certaines parties ». Plusieurs fonctions ne sont pas totalement comprises. |
| 1–2 | Déclaration minimale ou absente. L’utilisation de l’IA semble importante, mais n’est pas reconnue. |
| 0 | Aucune déclaration. Ou bien les questions complémentaires montrent que le groupe ne comprend aucune partie du code. |

---

## Qualité du dépôt — condition préalable de réussite

Avant l’application de la grille d’évaluation, le dépôt doit satisfaire les conditions suivantes :

- [ ] Le dépôt est public et accessible.
- [ ] `pip install -r requirements.txt` se termine sans erreur.
- [ ] `python src/agent.py` s’exécute et produit une sortie conforme au README.
- [ ] `python -m pytest tests/test_security.py` s’exécute sans erreur d’importation.

**Lorsque le dépôt ne satisfait pas cette condition préalable, le score maximal de la partie Implémentation technique est limité à 10/50 — correspondant à la partie non exécutable. Le rapport peut néanmoins être évalué normalement.**

---

## Formule de calcul de la note finale

```text
implementation_technique = A + B + C + D + E    (maximum 50)
evaluation_mesures       = F + G                (maximum 20)
qualite_rapport          = H + I + J            (maximum 20)
transparence_ia          = K                    (maximum 10)

note_finale = implementation_technique + evaluation_mesures + qualite_rapport + transparence_ia
```

Maximum : 100 points.

---

## Interprétation de la note

| Score | Interprétation |
|-------|----------------|
| 85–100 | Exceptionnel — système de niveau production, toutes les techniques sont intégrées, rapport honnête et précis |
| 70–84 | Solide — système fonctionnel, bonne évaluation, quelques lacunes mineures |
| 55–69 | Satisfaisant — le système fonctionne, les techniques de base sont présentes, mais l’évaluation manque de profondeur |
| 40–54 | Faible — implémentation partielle, évaluation manquante, rapport superficiel |
| Moins de 40 | Insuffisant — le système ne fonctionne pas, certaines techniques ne sont pas implémentées ou la déclaration d’utilisation de l’IA est insuffisante |
