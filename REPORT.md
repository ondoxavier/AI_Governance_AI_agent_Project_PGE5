# Rapport - Agent IA de gouvernance et conformité AI Act

## 1. Présentation du problème

L'utilisateur cible est un analyste conformité dans une organisation qui déploie des systèmes d'IA. Il doit qualifier rapidement le niveau de risque d'un cas d'usage au regard de l'AI Act européen, identifier les obligations associées et documenter les preuves utilisées.

Scénario concret : une équipe métier propose un système d'aide au tri de candidatures ou de scoring de crédit. L'agent récupère les passages pertinents du corpus réglementaire, classe le niveau de risque, produit une synthèse argumentée et fait relire la réponse par un agent critique.

## 2. Architecture

```text
Question utilisateur
   -> L1 Guardrail
   -> Recherche hybride BM25 + dense locale
   -> Fusion RRF
   -> Reranking
   -> Synthèse self-consistency k=3
   -> Agent critique
   -> Réponse finale
```

Décision de conception : le socle fonctionne hors ligne avec des implémentations locales déterministes. Ce choix réduit la dépendance aux clés API pendant la correction et permet à l'enseignant d'exécuter le projet depuis un clone vierge. Les interfaces restent séparées pour permettre un remplacement par OpenAI, Langfuse et un cross-encoder réel.

## 3. Évaluation

| Métrique | Baseline | Version finale | Technique à l'origine du changement |
|----------|----------|----------------|--------------------------------------|
| context_recall | À mesurer | À mesurer | BM25 + dense + RRF |
| context_precision | À mesurer | À mesurer | Reranking |
| faithfulness | À mesurer | À mesurer | Citations structurées + agent critique |
| answer_relevancy | À mesurer | À mesurer | Self-consistency k=3 |

Mesures à compléter après dix exécutions :

- Coût moyen : à mesurer en dollars américains.
- Latence moyenne : à mesurer en secondes.
- Répartition des appels d'outils : à mesurer sur dix exécutions.
- Déclenchements `TokenBudget` : à documenter.

## 4. Sécurité

Le projet contient un filtre L1 contre les injections de prompt et une matrice L4 de contrôle d'action. Les cinq tests d'injection sont dans `tests/test_security.py`.

Exemple de tentative bloquée : une entrée demandant d'ignorer les consignes précédentes et de révéler les variables d'environnement est bloquée par le filtre L1 avant toute recherche documentaire.

## 5. Évaluation AI Act européen

L'agent est un outil d'assistance à l'analyse réglementaire. Dans sa configuration actuelle, il ne prend pas de décision automatique à effet juridique direct : il relève donc plutôt du risque limité ou minimal selon le contexte d'usage. Si l'agent était intégré dans une décision RH, crédit, éducation, santé ou accès à un service essentiel, l'usage pourrait devenir à haut risque.

Obligation retenue : informer l'utilisateur que la sortie est une aide à la décision et conserver les preuves utilisées. Cette obligation est mise en œuvre par la section `PREUVES` de la réponse et par le verdict de l'agent critique.

## 6. Limites et prochaines étapes

Première limite : le reranking local est lexical et ne remplace pas un vrai cross-encoder entraîné. Il peut échouer sur des paraphrases réglementaires complexes.

Deuxième limite : les métriques RAGAS doivent encore être exécutées sur un jeu de dix questions annotées.

Prochain sprint : brancher un modèle d'embeddings et un cross-encoder réels, ajouter Langfuse en production avec traces exportées, puis compléter l'évaluation quantitative.

## 7. Déclaration d'utilisation de l'IA

| Composant | Écrit par un humain | Assisté par IA | Généré par IA |
|-----------|---------------------|----------------|---------------|
| Présentation du problème | À compléter | À compléter | À compléter |
| Architecture | À compléter | À compléter | À compléter |
| Boucle principale de l'agent - agent.py | À compléter | À compléter | À compléter |
| Serveur MCP - mcp_server.py | À compléter | À compléter | À compléter |
| Garde-fous - guardrails.py | À compléter | À compléter | À compléter |
| Pipeline de recherche documentaire | À compléter | À compléter | À compléter |
| Texte du rapport | À compléter | À compléter | À compléter |
