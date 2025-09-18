# InsolvenzaBancaria

Progetto per la previsione dell’insolvenza/bancruptcy aziendale utilizzando il Polish Companies Bankruptcy Dataset (UCI ID 365).

Contenuti principali:
- Caricamento automatico del dataset via `ucimlrepo`.
- Preprocessamento (imputazione mediana, standardizzazione).
- Modelli baseline: Regressione Logistica, Random Forest, Gradient Boosting.
- Valutazione con CV stratificata (ROC-AUC, PR-AUC, F1, Recall, Precision, Balanced Accuracy) e scelta soglia ottimizzata su F2.
- Salvataggio artefatti in `artifacts/` (modello e metriche).

## Confronto modelli e report

È disponibile un report completo con tutte le figure (ROC/PR, matrici di confusione, calibrazione, istogrammi dei punteggi, importanze):

- Report: `artifacts/report_modelli.md`

Per rigenerare i risultati del report (metriche comparative e plot per tutti i modelli):

```
# confronto metriche (salva artifacts/model_comparison.csv)
python scripts/compare.py --models logreg,rf,gb --folds 5

# plot per tutti i modelli indicati (salva in artifacts/plots)
python scripts/plots.py --models logreg,rf,gb --folds 5 --outdir artifacts/plots
```

Apri quindi `artifacts/report_modelli.md` (che referenzia le immagini dentro `artifacts/plots/`).

## Setup rapido

1) Installare i requisiti

```
make setup
```

2) Addestrare un modello (default: logistic regression)

```
make train
# oppure
python scripts/train.py --model rf --cv-folds 5
```

Gli artefatti vengono scritti in `artifacts/` con timestamp.

## Plot e visualizzazioni

Per generare curve ROC/PR, matrice di confusione (alla soglia ottimizzata), calibrazione, istogrammi dei punteggi e importanza delle feature:

```
python scripts/plots.py --model logreg --folds 3 --outdir artifacts/plots
# altri modelli: --model rf | gb
```

I file prodotti includono:
- `artifacts/plots/<model>.roc.png` e `.pr.png`
- `artifacts/plots/<model>.cm.png` (confusion matrix)
- `artifacts/plots/<model>.calibration.png`
- `artifacts/plots/<model>.score_hist.png`
- `artifacts/plots/<model>.features.png` (importanza/coefficenti top)

## Note sul dataset
Il dataset viene scaricato con:

```
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=365)
```

Le feature sono numeriche (65 colonne) e il target è binario `class` (0/1). Se nel futuro si vorranno aggiungere variabili macroeconomiche annuali (es. World Bank), si potrà estendere la pipeline unendo le serie per anno, qualora il dataset includa un campo temporale esplicito.
