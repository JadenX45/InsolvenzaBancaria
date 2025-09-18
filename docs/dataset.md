# Documentazione del dataset

## Origine e contesto
- **Nome:** Polish Companies Bankruptcy (UCI Machine Learning Repository, ID 365).
- **Licenza:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Periodo coperto:** bilanci di aziende polacche tra il 2000 e il 2013 (aziende fallite monitorate 2000-2012, aziende attive 2007-2013).
- **Fonte originale:** dati raccolti per lo studio "Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction" (Ziȩba, Tomczak & Tomczak, Expert Systems with Applications, 2016). DOI: 10.1016/j.eswa.2016.04.001 (dataset DOI: 10.24432/C5F600).

Il dataset viene scaricato automaticamente tramite `ucimlrepo.fetch_ucirepo(id=365)` all'interno di `src/insolvenza/data.py:16`. La funzione restituisce un oggetto `Dataset` con feature (`X`), target (`y`) e metadati.

## Composizione del dataset
- **Osservazioni totali:** 43.405 righe dopo il merge dei 5 sotto-dataset annuali forniti da UCI.
- **Feature disponibili:** 65 colonne numeriche.
  - `year`: intero da 1 a 5, indica quanti anni prima del fallimento è stato osservato il bilancio (i 5 file originali vengono accodati e identificati con questo campo).
  - `A1`-`A64`: 64 indicatori finanziari anonimizzati (rapporti di redditività, liquidità, solvibilità, struttura patrimoniale). UCI non pubblica una legenda ufficiale; le definizioni sono reperibili solo nella pubblicazione originale.
- **Target:** colonna binaria `class` con valori `0` (azienda attiva) e `1` (azienda fallita).
- **Squilibrio di classe:** 2.091 casi positivi (4,82%). La tabella seguente mostra il dettaglio per anno:

| year | osservazioni | fallimenti | tasso positivo |
| --- | --- | --- | --- |
| 1 | 7.027 | 271 | 3,86% |
| 2 | 10.173 | 400 | 3,93% |
| 3 | 10.503 | 495 | 4,71% |
| 4 | 9.792 | 515 | 5,26% |
| 5 | 5.910 | 410 | 6,94% |

- **Valori mancanti:** complessivamente ~1,46% delle celle. I rapporti con più mancanti sono `A37` (18.984 valori NA), `A21`, `A27`, `A60` e `A45`. Solo `year` è sempre valorizzata.
- **Tipi:** tutte le feature sono trattate come numeriche (`float64`, eccetto `year` che è intera).

## Preprocessamento applicato
Il pre-processore definito in `src/insolvenza/pipeline.py:17` è identico per tutti i modelli:
1. `SimpleImputer(strategy="median")` sulle 65 colonne numeriche per colmare i valori mancanti.
2. `StandardScaler()` per standardizzare media e deviazione standard.
3. `ColumnTransformer` associa il pipeline numerico a tutte le colonne (`remainder="drop"`).

Non viene effettuata ulteriore selezione di feature: ogni modello vede l’intero vettore di indicatori trasformati.

## Strategie di suddivisione del dataset
- **Cross-validation principale:** `StratifiedKFold` con 5 fold, shuffle e `random_state=42` (`cross_validate` in `src/insolvenza/pipeline.py:44`). Restituisce probabilità out-of-fold usate per metriche e ottimizzazione della soglia F2.
- **Split hold-out (script di diagnostica):** `train_test_split` con `test_size=0.25`, stratificazione sul target e `random_state=123` (`scripts/holdout_eval.py:19`).
- **Addestramento finale:** dopo la CV, l’intero dataset viene riutilizzato (`fit_full` in `src/insolvenza/pipeline.py:67`) per allenare il modello definitivo prima del salvataggio.

## Utilizzo nei modelli
Tutti i modelli condividono le stesse feature preprocessate e il target `class`. Le differenze riguardano solo l’algoritmo supervisionato:

- **Regressione Logistica (`logreg`):**
  - Implementazione: `sklearn.linear_model.LogisticRegression` (`max_iter=2000`, `class_weight="balanced"`).
  - Motivazione: migliora il recall in presenza di forte sbilanciamento. Output: probabilità calibrate usate per ottimizzare la soglia F2.

- **Random Forest (`rf`):**
  - Implementazione: `sklearn.ensemble.RandomForestClassifier` con `n_estimators=200`, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1`.
  - Pur essendo un modello tree-based, sfrutta comunque l’imputazione e la standardizzazione del pre-processore condiviso (lo scaling non è strettamente necessario ma non degrada le prestazioni).

- **Gradient Boosting (`gb`):**
  - Implementazione: `sklearn.ensemble.GradientBoostingClassifier` con parametri di default (`random_state=42`).
  - Non dispone di `class_weight`; lo sbilanciamento viene gestito a valle tramite scelta della soglia ottimale su F2.

Per ciascun modello `cross_validate` calcola metriche ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall e F1, con soglia selezionata per massimizzare F2 (`src/insolvenza/metrics.py:33`). Gli script `scripts/train.py`, `scripts/compare.py` e `scripts/plots.py` generano rispettivamente artefatti addestrati, confronti tabellari e grafici salvati in `artifacts/`.

## Riproducibilità e riferimenti
- **Caricamento dati:** `src/insolvenza/data.py` (`fetch_polish_companies_bankruptcy`, `train_test_split`).
- **Pipeline completa:** `src/insolvenza/pipeline.py` (preprocessore, modelli, cross-validation, fit finale).
- **Metriche e soglia:** `src/insolvenza/metrics.py`.
- **Script CLI:** cartella `scripts/` (`train.py`, `compare.py`, `holdout_eval.py`, `plots.py`).

Questa documentazione descrive la configurazione attuale del repository; eventuali modifiche future a preprocessamento, feature engineering o strategie di split andranno riportate qui per mantenere l’allineamento.
