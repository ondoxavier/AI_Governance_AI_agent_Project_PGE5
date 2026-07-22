# 📝 Consignes du devoir — Agent IA de production

> **Groupes de 2 à 3 personnes · Dépôt GitHub + rapport · À rendre le jeudi 23 juillet à 23 h 59**
> **Tout devoir remis en retard recevra la note de 0. Aucune exception.**

---

## Le travail demandé

Construisez un **agent IA de production complet** qui répond à une problématique réelle. Votre agent doit intégrer l’ensemble des éléments étudiés dans ce cours, non pas sous la forme d’une simple démonstration, mais comme un système fonctionnel.

La différence par rapport au premier livrable est la suivante : vous disposez désormais d’un système de recherche documentaire de niveau production — recherche hybride et reranking —, d’un serveur MCP fonctionnel, d’une pile de garde-fous testée contre les injections et d’une stratégie de raisonnement. L’agent réalisé pour ce devoir doit exploiter tous ces éléments.

---

## Ce que vous devez construire

Un agent de recherche ou d’analyse qui :

- utilise une **recherche hybride** — BM25 + recherche dense + RRF — pour la récupération d’informations ;
- applique un **reranking par cross-encoder** avant l’assemblage du contexte ;
- expose ses outils au moyen d’un **serveur MCP personnalisé**, comprenant au moins trois outils ;
- implémente un **filtrage des entrées de niveau L1** et un **contrôle des actions de niveau L4** ;
- utilise un **raisonnement CoT avec quelques exemples — few-shot CoT** selon le format PREUVES / ANALYSE / CONCLUSION / CONFIANCE ;
- applique une **Self-Consistency avec k = 3** lors de l’étape finale de synthèse ;
- est **instrumenté avec Langfuse** : chaque appel au LLM et chaque appel d’outil doit disposer de son propre span ;
- comprend un **second rôle d’agent**, au minimum un agent critique chargé de vérifier la réponse avant qu’elle ne soit retournée.

---

## Liste des sujets

Choisissez un sujet. La règle est : premier sujet enregistré, premier servi. Lorsque votre sujet est déjà pris, choisissez-en un autre. Tous les sujets proposés offrent suffisamment de matière.

| Nº | Sujet | Exemple d’angle d’analyse |
|---|-------|---------------------------|
| 1 | 🌊 Déplacements climatiques | Prédiction des risques par région · analyse des corridors de relocalisation · recherche sur l’allocation de l’aide |
| 2 | 🦠 Préparation aux pandémies | Détection précoce de signaux dans les actualités et les prépublications · génération de rapports de situation |
| 3 | 🍎 Sécurité alimentaire | Alerte précoce sur les mauvaises récoltes · analyse des perturbations de la chaîne d’approvisionnement |
| 4 | 🏙️ Migration urbaine | Analyse des facteurs d’attraction et de répulsion · évaluation de la capacité d’accueil des villes |
| 5 | ♻️ Surveillance environnementale | Suivi de la déforestation · synthèse des événements de pollution |
| 6 | 💊 Résistance aux médicaments | Suivi des tendances de la résistance aux antimicrobiens — AMR · recherche sur les protocoles thérapeutiques |
| 7 | 📰 Détection de la désinformation | Traçage de l’origine d’un récit · vérification croisée d’affirmations entre plusieurs sources |
| 8 | ⚖️ Gouvernance de l’IA | Étude de conformité à l’AI Act européen · comparaison des réglementations entre juridictions |

---

## Structure obligatoire du dépôt

```text
votre-depot/
├── README.md              # instructions d’installation + description de l’architecture
├── REPORT.md              # rapport écrit — voir la structure ci-dessous
├── requirements.txt       # dépendances avec versions fixées, installation propre depuis zéro
├── .env.example           # toutes les clés requises, sans aucune valeur
├── src/
│   ├── agent.py           # boucle principale de l’agent
│   ├── mcp_server.py      # serveur MCP — au moins 3 outils
│   ├── retrieval.py       # recherche hybride + reranking
│   ├── guardrails.py      # filtre L1 + contrôle L4 + TokenBudget
│   └── reasoning.py       # CoT few-shot + self-consistency
├── tests/
│   └── test_security.py   # 5 tests d’injection — ils doivent tous réussir
├── docs/
│   └── architecture.md    # diagramme d’architecture + description des composants
└── data/
    └── README.md          # décrit les données à placer ici et la manière de les charger
```

**L’enseignant clonera votre dépôt et exécutera votre projet.** Lorsque le projet ne fonctionne pas à partir d’un clone vierge en suivant les instructions du README, il est considéré comme non fonctionnel.

```bash
# Votre README doit permettre d’exécuter exactement cette séquence :
git clone votre-depot
cd votre-depot
cp .env.example .env    # l’étudiant renseigne ses propres clés
pip install -r requirements.txt
python src/agent.py     # l’agent s’exécute et produit une sortie
```

---

## Structure du fichier REPORT.md

Le rapport doit comporter au maximum quatre pages. Aucun remplissage inutile. Chaque section doit être propre à votre projet.

### 1. Présentation du problème — ½ page

Qui est l’utilisateur ? Que fait votre agent qu’un chatbot ou un moteur de recherche classique ne peut pas faire ? Présentez un scénario concret dans lequel votre agent produit un résultat utile qui demanderait autrement plusieurs heures de travail.

### 2. Architecture — ½ page + diagramme

Décrivez chaque composant et son rôle. Votre diagramme doit correspondre exactement au code réellement exécuté. Expliquez une décision de conception qui n’était pas évidente : pourquoi avez-vous fait ce choix ?

### 3. Évaluation — 1 page

Présentez votre tableau RAGAS en comparant la baseline — avant les améliorations du Bloc 1 — et la version finale — après toutes les améliorations. Pour chaque métrique ayant progressé, indiquez la technique responsable de cette progression. Pour chaque métrique n’ayant pas progressé, expliquez pourquoi.

| Métrique | Baseline | Version finale | Technique à l’origine du changement |
|----------|----------|----------------|--------------------------------------|
| context_recall | | | |
| context_precision | | | |
| faithfulness | | | |
| answer_relevancy | | | |

Indiquez également :

- le coût moyen d’une exécution en dollars américains ;
- la latence moyenne en secondes ;
- la répartition des appels d’outils, c’est-à-dire le nombre d’appels de chaque outil au cours de dix exécutions de test.

### 4. Sécurité — ½ page

Présentez les résultats de vos cinq tests d’injection, avant et après l’ajout des protections L1 et L4. Décrivez une tentative réelle d’injection bloquée par votre système et précisez exactement quelle couche l’a détectée.

### 5. Évaluation au regard de l’AI Act européen — ½ page

À quel niveau de risque appartient votre agent : interdit, élevé, limité ou minimal ? Justifiez votre réponse en vous appuyant sur les critères précis de l’AI Act européen. Quelle obligation découle de ce niveau de risque et comment l’avez-vous mise en œuvre ?

### 6. Limites et prochaines étapes — ½ page

Quel élément serait le premier à échouer dans un environnement de production ? Qu’ajouteriez-vous lors du prochain sprint ? Soyez précis : « améliorer l’agent » ne constitue pas une limite suffisamment détaillée.

### 7. Déclaration d’utilisation de l’IA

| Composant | Écrit par un humain | Assisté par IA | Généré par IA |
|-----------|---------------------|----------------|---------------|
| Présentation du problème | | | |
| Architecture | | | |
| Boucle principale de l’agent — agent.py | | | |
| Serveur MCP — mcp_server.py | | | |
| Garde-fous — guardrails.py | | | |
| Pipeline de recherche documentaire | | | |
| Texte du rapport | | | |

Soyez honnêtes. Il pourra vous être demandé d’expliquer n’importe quelle fonction de votre base de code.

---

## Soumission

1. Poussez l’intégralité du projet sur la branche `main` avant la date limite.
2. Envoyez par e-mail l’URL du dépôt à l’enseignant avec l’objet : `[PGE5 HW] Groupe N — Nom du sujet`.
3. L’enseignant clonera et exécutera le dépôt après la date limite.

**Date limite : exactement quatre jours à compter d’aujourd’hui, à 23 h 59.** Les commits effectués après cette échéance ne seront pas pris en compte. L’enseignant clonera le dépôt à 23 h 59.

---

## Caractéristiques d’un excellent rendu

Un excellent rendu comporte :

- une problématique précise associée à un utilisateur réel — et non une simple formulation telle que « l’agent répond à des questions sur X » ;
- des scores RAGAS améliorés par rapport à la baseline, avec la technique responsable clairement identifiée ;
- cinq tests d’injection sur cinq réussis ;
- une trace Langfuse contenant au moins cinq spans visibles — agent + deux appels LLM + deux appels d’outils ;
- un agent critique produisant un verdict visible sur la réponse ;
- un rapport dans lequel chaque affirmation est appuyée par une valeur numérique ou une référence au code.

Un rendu faible comporte :

- une présentation vague du problème — par exemple : « aider les gens à comprendre le changement climatique » ;
- l’absence de baseline RAGAS, empêchant toute comparaison ;
- des tests de sécurité non exécutés ou partiellement réussis ;
- un rapport décrivant l’agent de façon générale, sans éléments spécifiques ;
- un code qui ne fonctionne pas à partir d’un clone vierge.
