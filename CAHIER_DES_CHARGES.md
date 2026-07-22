# Cahier des charges — Agent de gouvernance et conformité IA (UE / US / UK)
**Groupe de 4 · Deadline : jeudi 23 juillet 2026, 23:59**

> Sujet 8 — AI governance : *EU AI Act compliance research · regulation comparison across jurisdictions*
> Repo : `AI_Governance_AI_agent_Project_PGE5` · Brief : `docs/HW_brief_FR.md` · Rubric : `docs/HW_rubric_FR.md`

---

## 0. Règle d'or du sprint

Avec le temps restant, **on n'optimise pas la qualité, on optimise les points du rubric**. Priorité absolue n°1 : que le repo passe la porte pass/fail (`pip install` + `python src/agent.py` + `pytest tests/test_security.py` depuis un clone vierge — sinon plafond à 10/50 sur l'implémentation technique). Tout le reste vient après, dans l'ordre de la section 9.

---

## 1. Pitch produit

**Utilisateur :** un·e analyste conformité dans une organisation qui déploie des systèmes d'IA sur plusieurs marchés. Il/elle doit qualifier rapidement le niveau de risque d'un cas d'usage IA au regard de l'AI Act européen, **et** savoir comment ce même cas d'usage est traité aux États-Unis et au Royaume-Uni.

**Scénario concret (à mettre tel quel dans REPORT.md §1) :**
> « Une banque opérant en Europe et aux US utilise un modèle IA pour présélectionner des candidats à un crédit. L'analyste conformité doit déterminer : (1) le niveau de risque AI Act et les obligations associées (Art. 6 + Annexe III → Art. 9–15), (2) ce que ce cas déclenche côté US (NIST RMF volontaire, mais loi Colorado SB24-205 contraignante sur les décisions de crédit) et côté UK (principes non contraignants appliqués par la FCA). Recherche manuelle dans 3 cadres réglementaires ≈ plusieurs heures ; avec l'agent ≈ quelques minutes, chaque affirmation citée avec sa source, sa date et son statut. »

**Ce qu'un chatbot générique ne fait pas :** ancrage sur les textes exacts (pas d'hallucination de numéros d'article), grille de décision structurée par juridiction, et surtout **la discipline statut** (voir §2).

---

## 2. Règle de conception impérative — source / date / statut

> **L'agent ne présente jamais une comparaison entre juridictions comme une vérité juridique universelle.** Chaque affirmation précise sa source, sa date et son statut : **obligatoire / volontaire / projet / recommandation**.

Cette règle est portée par la donnée elle-même : chaque chunk de l'index a des champs `jurisdiction` et `status` (remplis par `src/ingest.py`), et la synthèse (`src/reasoning.py`) doit interdire de fusionner des statuts différents dans une phrase généraliste. Exemples de statuts en corpus :
- AI Act, RGPD → `obligatoire`
- NIST AI RMF / Playbook / GenAI Profile → `volontaire`
- Executive Orders US → `obligatoire (federal, revocable)` — le triptyque EO 14110 → 14179 → 14365 illustre l'instabilité par décret
- Colorado SB24-205 → `obligatoire (etat)`
- White paper UK + réponse gouvernementale → `recommandation`

---

## 3. Les trois juridictions

| Juridiction | Nature | Classification | Corpus (`data/`) |
|---|---|---|---|
| **UE** (principale) | Réglementation horizontale **obligatoire** | Niveaux de risque : interdit / haut / limité / minimal | `ai_act_corpus/` (16 PDF) + `gdpr_corpus/` (17 PDF) |
| **États-Unis** | Cadre principalement **volontaire** et sectoriel + exceptions contraignantes (décrets fédéraux, loi d'État Colorado) | Gestion des risques : Govern / Map / Measure / Manage | `us_ai_regulation_corpus/` (8 PDF) |
| **Royaume-Uni** | **Principes non contraignants** appliqués par les régulateurs sectoriels existants (ICO, FCA, MHRA, Ofcom) | Contextuelle et sectorielle — 5 principes cross-sectoriels | `uk_ai_regulation_corpus/` (2 PDF) |

Détail des documents, sources officielles et statuts : `README.md` de chaque sous-dossier de `data/`.

**Périmètre fonctionnel** : l'UE reste la juridiction de référence (classification de risque = cœur de l'agent). US et UK servent à l'angle comparaison — l'outil de recherche accepte un filtre `jurisdiction` (EU/US/UK/all), déjà implémenté dans `retrieval.py`.

---

## 4. Architecture — structure du repo (imposée par le brief)

```
AI_Governance_AI_agent_Project_PGE5/
├── README.md                  ✅ (3 juridictions documentées)
├── CAHIER_DES_CHARGES.md      ✅ (ce fichier)
├── REPORT.md                  ⚠️ squelette — chiffres à compléter (RAGAS, coût, latence)
├── requirements.txt           ✅
├── .env.example               ✅
├── src/
│   ├── agent.py               ⚠️ boucle L1→retrieval→synthèse→critique OK, mais tracer local (pas Langfuse) et pas de LLM
│   ├── ingest.py              ✅ PDF→txt→chunks parent-enfant→embeddings (2 968 chunks : EU 1 492, US 968, UK 508)
│   ├── retrieval.py           ✅ BM25 + dense + RRF + cross-encoder, filtre juridiction, fallback offline
│   ├── guardrails.py          ✅ L1 + L4 (ACTION_RISK_MATRIX) + TokenBudget — 9/9 tests
│   ├── reasoning.py           ❌ format PREUVES/ANALYSE/CONCLUSION/CONFIANCE OK, mais classification par mots-clés — PAS de LLM, self-consistency k=3 factice
│   └── mcp_server.py          ⚠️ 3 tools avec docstrings complets, mais à valider avec MCP inspector + ajouter compare_jurisdiction
├── tests/
│   └── test_security.py       ✅ 9 tests passent (5 injections + L4 + budget)
├── docs/
│   └── architecture.md        ⚠️ à mettre à jour pour refléter ingest.py + les 3 juridictions
└── data/                      ✅ 43 PDF, 4 corpus, README par corpus avec statuts
```

---

## 5. Spécification des composants — reste à faire

### 5.1 Retrieval — ✅ FAIT (15 pts visés)
Hybrid BM25 + dense (MiniLM) + RRF + cross-encoder, chunking parent-enfant juridique, filtre juridiction. Garder un `mode baseline` (cosine seul) pour le tableau RAGAS avant/après — **à ajouter** : un flag `mode="baseline"` dans `hybrid_search` (~15 lignes).

⚠️ Limitation connue (pour REPORT.md §6) : corpus ~95% anglais, MiniLM non multilingue → les requêtes françaises sous-performent. Options : `paraphrase-multilingual-MiniLM-L12-v2` ou traduction de la requête avant retrieval.

### 5.2 Reasoning — ❌ PRIORITÉ 1 (10 pts)
Brancher un **vrai LLM** (clé Anthropic/OpenAI dans `.env`) dans `reasoning.py` :
- Prompt de synthèse few-shot (2-3 exemples) au format PREUVES / ANALYSE / CONCLUSION / CONFIANCE
- Le prompt exige : chaque preuve citée avec `[source · juridiction · statut]` (les champs sont déjà dans `SearchResult.document`)
- **Self-Consistency k=3 réel** : 3 appels LLM indépendants sur la synthèse, vote majoritaire sur la CONCLUSION, désaccord → CONFIANCE basse
- Conserver le mode déterministe actuel comme fallback sans clé API (même pattern que retrieval)

### 5.3 Agent critique — ⚠️ à renforcer avec le LLM
`critic_review` existe mais est trivial. Version LLM : vérifier que chaque article cité existe dans le contexte récupéré (anti-hallucination), que la conclusion suit la grille de la juridiction concernée, que les statuts ne sont pas mélangés. Verdict APPROVE / REVISE visible dans la sortie (exigé par « strong submission »).

### 5.4 MCP — ⚠️ à finaliser (10 pts)
- Valider les 3 tools existants avec `mcp inspector` (le package `mcp` est dans requirements)
- **Ajouter un 4ᵉ tool `compare_jurisdiction(topic)`** : lance `hybrid_search` sur EU, US et UK séparément et retourne les 3 blocs de résultats étiquetés — c'est l'outil qui matérialise l'angle comparaison du sujet
- Docstrings déjà au format exigé (Use when / Do NOT use / Returns / Example) — garder ce format pour le nouveau tool

### 5.5 Guardrails — ✅ FAIT, enrichissement optionnel
Enrichir `INJECTION_PATTERNS` avec les patterns du Lab B2 (`role_injection` type DAN, `fictional_framing`, homoglyphes via NFKC — déjà présent). Seulement si temps.

### 5.6 Observabilité — ❌ (5 pts)
Remplacer/doubler `LocalTracer` par **Langfuse réel** (`@observe()` ou spans manuels) : trace agent + spans LLM (synthèse ×3, critique) + spans tools. Compte gratuit cloud.langfuse.com, clés dans `.env`. Logger `AGENT_VERSION`. Minimum 5 spans visibles.

---

## 6. Plan d'évaluation (20 pts)

### RAGAS (12 pts) — ≥10 questions, 4 métriques
Questions construites depuis le corpus, réparties sur les 3 juridictions (~6 EU, ~2 US, ~2 UK) :
1. Quelles pratiques d'IA sont interdites selon l'Article 5 de l'AI Act ?
2. Un outil de tri de CV est-il à haut risque selon l'AI Act ? (Annexe III)
3. Quelles obligations pour les fournisseurs de systèmes à haut risque ? (Art. 9–15)
4. Que doit contenir la documentation technique (Art. 11) ?
5. Qu'exige la supervision humaine (Art. 14) ?
6. Quand une AIPD est-elle obligatoire sous le RGPD (Art. 35) ?
7. Quelles sont les 4 fonctions du NIST AI RMF ?
8. Que régule la loi Colorado SB24-205 et qui est concerné ?
9. Quels sont les 5 principes UK et sont-ils contraignants ?
10. Comment l'approche UK diffère-t-elle de l'AI Act sur la classification des risques ?

**Baseline** = retrieval en mode cosine seul · **Final** = pipeline complet. Remplir le tableau REPORT.md §3 avec les 4 métriques et lier chaque amélioration à sa technique (attendu : context_precision ↑ grâce au reranking, context_recall ↑ grâce à hybrid+RRF).

### Coût / latence (8 pts)
10 runs sur les questions ci-dessus : coût moyen USD (via usage LLM), latence moyenne, distribution des appels d'outils. Provoquer 1 dépassement de TokenBudget documenté (cap volontairement bas sur un run).

---

## 7. Auto-évaluation EU AI Act (REPORT.md §5) — pré-analyse

**Niveau de risque de NOTRE agent : risque limité (Article 50)** :
- Pas dans l'Annexe III : il *renseigne* sur les catégories à haut risque, il n'exécute aucune décision (recrutement, crédit...) lui-même
- Aide à la décision : la sortie doit être validée par un humain — à écrire dans le prompt système ET afficher en sortie : *« Cette analyse est générée par IA et doit être validée par un juriste avant toute décision. »* (= l'obligation de transparence Art. 50, implémentée)
- Bonus rapport : notre propre règle source/date/statut est un garde-fou de conformité par conception — à mentionner

---

## 8. Répartition du travail (4 personnes)

| Rôle | Fichiers | Tâches |
|---|---|---|
| **A — Retrieval** | `retrieval.py`, `data/`, `ingest.py` | ✅ corpus + index faits · reste : flag `mode="baseline"` pour RAGAS, éventuel passage multilingue |
| **B — MCP & agent** | `mcp_server.py`, `agent.py` | Valider les 3 tools (inspector), ajouter `compare_jurisdiction`, brancher le tracer Langfuse dans la boucle |
| **C — Guardrails & reasoning** | `guardrails.py`, `reasoning.py` | ✅ guardrails faits · reste : LLM réel dans la synthèse (few-shot CoT + self-consistency k=3 + critique LLM) — **priorité 1 du projet** |
| **D — Eval, observabilité, rapport** | RAGAS, `REPORT.md`, `README.md`, Langfuse | Jeu de 10 questions 3-juridictions, script RAGAS baseline/final, mesures coût/latence, rédaction rapport, tableau disclosure IA |

**Point de synchronisation** : les signatures publiques sont **gelées** — `hybrid_search(query, top_k, data_dir, jurisdiction)`, `SearchResult.document.{title,text,source,context,jurisdiction,status}`, `self_consistency(question, contexts, k)`, `l1_filter/authorize_action/TokenBudget`. Toute modification cassante se discute dans le groupe d'abord.

---

## 9. Checklist de soumission finale (porte pass/fail + rubric)

- [ ] Repo public et accessible
- [ ] `pip install -r requirements.txt` sans erreur **depuis un dossier vierge** (pas votre venv de dev)
- [ ] `python src/agent.py` tourne et produit une sortie en suivant le README (avec ET sans index construit)
- [ ] `python -m pytest tests/test_security.py` sans erreur d'import, tests verts
- [ ] `.env.example` liste toutes les clés (LLM + Langfuse), sans valeurs
- [ ] REPORT.md ≤ 4 pages, tableau RAGAS avec vrais chiffres, disclosure IA honnête
- [ ] `docs/architecture.md` : diagramme conforme au code réel (ingest → index → hybrid → synthèse → critique)
- [ ] Email au prof : `[PGE5 HW] Groupe N — AI governance`

---

## 10. Ordre de repli si le temps manque

1. **Multilingue** → sauter (documenter en limitation §6 du rapport, c'est même valorisé)
2. **`compare_jurisdiction` MCP** → si coupé, l'angle comparaison reste démontrable via le filtre `jurisdiction` de `hybrid_search`
3. **RAGAS 10 questions → 5** (score 5-7/12 au lieu de 11-12, mieux que 0)
4. **Critique LLM → garder le critique déterministe** actuel (verdict visible = l'exigence minimale)
5. **Ne jamais sacrifier** : la porte pass/fail, le LLM dans la synthèse (sans lui, self-consistency et CoT sont factices → D et une partie de A du rubric s'effondrent), le rapport (20 pts), les tests de sécurité (déjà verts — ne pas les casser)
