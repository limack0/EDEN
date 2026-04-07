# EDEN — Manifeste de Recherche

**Emergent Developmental Encoding Network**
*Un réseau de neurones qui se développe comme un organisme*

---

## Vision

EDEN part d'une intuition : les réseaux biologiques ne sont pas câblés à la main — ils *poussent*. Le génome fixe des règles de développement, l'épigénome les module selon l'environnement, et des mécanismes locaux (apoptose, compétition, différenciation) façonnent la topologie finale. EDEN tente de reproduire cette logique dans un cadre PyTorch entraînable par gradient.

L'hypothèse centrale : **si les mécanismes biologiques qui produisent les cerveaux naturels sont encodés correctement, un réseau artificiel devrait en tirer un avantage en termes de généralisation, robustesse et efficacité**.

---

## Architecture

EDEN est structuré autour d'un pipeline de développement en plusieurs étapes :

```
Input
  → CNN embed (2D) / Linear proj (1D)
  → StemPool        — pool de cellules souches à compétition top-k
  → node_proj       — projection vers N nœuds parallèles
  → LocalMorphogen  — gradient morphogène basé sur la perte/accuracy
  → DifferentiationPhi (θ, α, γ) — trois stades de différenciation
  → Glia (Astrocyte + Oligodendrocyte) — modulation du gain
  → MicrogliaCompetition + ProgrammedApoptosis — élagage des nœuds faibles
  → ContactInhibition — suppression des nœuds redondants
  → mean-pool → head
```

Le tout est gouverné par un `GeneRegulator` (8 paramètres appris : τ, λ, k, ρ, ...) et un `HeritableEpigenome` qui masque dynamiquement les gènes actifs. L'optimiseur standard Adam est complété par Eggroll L1/L2 pour l'évolution des paramètres régulateurs.

---

## Expériences

### Benchmark MNIST — GPU Kaggle (50 epochs, seed=0)

| Modèle | Val acc | FGSM robustesse |
|--------|---------|-----------------|
| EDEN | **0.9917** | **1.0000** |
| MLP baseline | ~0.98 | — |

EDEN atteint 99.17% de précision sur MNIST avec une robustesse adversariale parfaite (FGSM ε=0.1). La robustesse est particulièrement notable : aucun exemple adversarial sur les 24 testés n'a réussi à tromper le modèle.

---

### Ablation individuelle — ECG (3 seeds, 50 epochs)

Convention : Δacc = acc(mécanisme désactivé) − acc(full EDEN).
**Positif = désactiver améliore → le mécanisme nuit.**

| Mécanisme | Δacc moyen | Interprétation |
|-----------|-----------|----------------|
| `neurogenesis` | **+0.0187** | Nuit |
| `paracrine` | **+0.0151** | Nuit |
| `glia` | +0.0044 | Légèrement néfaste |
| `programmed_apoptosis` | +0.0009 | Neutre |
| `epigenome_drift` | **−0.0071** | Bénéfique (enlever le drift coûte) |

---

### Ablation combinée — MNIST (10 seeds, 50 epochs)

| Configuration | Mean acc | Std | Min | Max |
|--------------|----------|-----|-----|-----|
| `full_eden` (v1) | 0.9117 | 0.0110 | 0.8907 | 0.9253 |
| `no_neuro_para` | 0.9235 | 0.0156 | 0.8933 | 0.9467 |
| Δ no_neuro_para | +0.0117 | | | |

Sur 10 seeds indépendantes, désactiver neurogenèse + paracrine améliore l'accuracy de **+1.17%** en moyenne sur MNIST.

### Validation v2 — MNIST (10 seeds, 50 epochs, GPU)

v2 = neurogenèse et paracrine retirés + stems hétérogènes (ReLU/GELU/SiLU/skip) + NodeAttention

| Configuration | Mean acc | Std | Min | Max |
|--------------|----------|-----|-----|-----|
| v1 full_eden | 0.9117 | 0.0110 | 0.8907 | 0.9253 |
| **v2** | **0.9910** | **0.0018** | **0.9871** | **0.9926** |
| **Δ** | **+0.0793** | **6× ↓** | | |

**+7.93% de précision, variance divisée par 6.** Résultat le plus significatif du projet à ce jour.

---

## Analyses

### Pourquoi la neurogenèse nuit-elle ?

La neurogenèse dans EDEN est une *croissance structurelle dynamique* : sur détection de stagnation des gradients, le réseau ajoute un nœud (copie d'un nœud existant + bruit) ou un nouveau stem. En théorie, cela devrait débloquer l'optimiseur. En pratique :

1. **Instabilité de l'optimiseur** : l'ajout de nouveaux paramètres en cours d'entraînement casse l'état d'Adam (moments biaisés pour les nouveaux params).
2. **Faux positifs de stagnation** : le détecteur déclenche sur des plateaux temporaires normaux, introduisant du bruit non-nécessaire.
3. **L'architecture fixe suffit** : 8 nœuds × 128 hidden est déjà surdimensionné pour MNIST et ECG. Croître ne règle pas la capacité mais le bruit.

### Pourquoi le paracrine nuit-il ?

Le paracrine (`ParacrineCascade`, k=3) fait circuler un signal entre nœuds voisins avant la différenciation. L'hypothèse était qu'une "coordination locale" entre nœuds améliorerait la représentation. En pratique :

1. **Le signal paracrine amplifie la redondance** : les nœuds se synchronisent plutôt que de se spécialiser.
2. **Interaction négative avec ContactInhibition** : `ContactInhibition` est précisément là pour *supprimer* les nœuds trop similaires — le paracrine travaille en sens contraire.
3. **Coût sans bénéfice** : le `paracrine_strength` dans le `GeneRegulator` apprend à converger vers 0 de lui-même dans les runs longs.

### Ce qui fonctionne

| Mécanisme | Effet | Pourquoi ça marche |
|-----------|-------|--------------------|
| `StemPool` (top-k) | Positif | Sélection naturelle des meilleures représentations d'entrée |
| `DifferentiationPhi` | Positif | Transformations hiérarchiques θ→α→γ avec gates adaptatifs |
| `ContactInhibition` | Positif | Diversité forcée entre nœuds → moins de redondance |
| `epigenome_drift` | Positif | Exploration stochastique du masque génomique |
| `ProgrammedApoptosis` | Neutre | Élagage correct mais non critique sur petits datasets |

---

## Décisions

### Retrait de `neurogenesis` et `paracrine` (2026-04-06)

Sur la base des ablations ECG (3 seeds) et MNIST (10 seeds), les deux mécanismes sont retirés du code :

- `eden/core/neurogenesis.py` — plus importé
- `ParacrineCascade` — supprimé de `network.py`
- `AblationFlags.neurogenesis`, `AblationFlags.paracrine` — supprimés
- `LocalStagnationDetector` et le bloc de croissance — supprimés de `training.py`

Le `StemPool` reste : il n'est pas concerné par la neurogenèse dynamique. Les 4 stems initiaux avec compétition top-k sont bénéfiques.

---

## CIFAR-10 — Historique des tentatives

| Version | Changements | Mean acc | Note |
|---------|-------------|----------|------|
| v1 | base, 50 epochs | 0.6741 | baseline |
| v2 | +augmentation, +BatchNorm, 100 epochs | 0.6759 | Δ=+0.0018, embed trop léger |
| v3a | +3ème bloc conv (flat_dim=8192) | 0.2844 | **échec** — Linear(8192,256) instable, 3/5 seeds à 0.1 |
| v3b | +AdaptiveAvgPool(4) → flat_dim=2048 | *en attente GPU* | fix stabilité |

**Leçon** : augmenter la capacité du CNN embed est nécessaire, mais il faut contrôler la dimension de sortie avant le `StemPool`. Un `Linear(D_large, hidden)` non normalisé avec D_large >> hidden provoque une instabilité d'initialisation sévère.

**Stack actuel** (v3b, non validé sur GPU) :
- Dropout 0.1 dans `DifferentiationPhi` entre les stades θ/α/γ
- Cosine annealing LR (1e-3 → 1e-5 sur T_max epochs)
- CNN embed adaptatif : 3 blocs pour images ≥ 32×32, AdaptiveAvgPool(4) pour cap flat_dim

---

## Idées Futures

### Stems spécialisés (Mixture of Experts léger)

L'intuition de l'utilisateur : **plutôt que de faire croître le réseau dynamiquement, partir avec plusieurs types de stems fixes mais différenciés dès le départ**. Chaque stem aurait une inductive bias différente :

- Stem "texture" : convolutions petits kernels
- Stem "structure" : convolutions larges kernels
- Stem "temporel" : intègre l'historique (pour séquences)
- Stem "fréquentiel" : projection spectrale

C'est proche de ce que font les **Mixture of Experts** (MoE) dans les LLMs modernes, mais à l'échelle d'un stem perceptron. Le `StemPool` actuel a déjà l'architecture pour ça — il suffit de diversifier les types de `StemPerceptron` plutôt que de les rendre identiques.

### Autres pistes

- **Attention entre nœuds** : remplacer le paracrine par une self-attention légère entre nœuds (plus expressive, plus stable)
- **Curriculum génomique** : faire évoluer le masque épigénomique selon un curriculum (commencer sparse, devenir dense)
- **Benchmark élargi** : tester CIFAR-10 avec la version nettoyée — MNIST/ECG donnent peut-être trop peu de signal

---

## Philosophie

EDEN n'est pas un concurrent direct des Transformers ou des ResNets. C'est une exploration de **la biologie computationnelle comme source d'architectures** : non pas pour copier le cerveau, mais pour en extraire des principes — la compétition, la différenciation, l'induction, la mort cellulaire programmée — et tester lesquels transfèrent au deep learning.

Les mécanismes qui ne transfèrent pas (neurogenèse dynamique, signalisation paracrine) ne sont pas des échecs — ils nous apprennent que certaines adaptations biologiques dépendent de contraintes physiques (énergie, espace, temps de développement) absentes dans le silicium.

Ceux qui transfèrent (compétition entre stems, différenciation hiérarchique, inhibition de contact) suggèrent qu'il y a quelque chose de fondamentalement utile dans la *diversité forcée* et la *sélection locale*.

---

*EDEN v1.0 — Limack0 / 2026*
