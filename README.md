## Setup dell'ambiente

### 1. Creazione dell'ambiente virtuale

```bash
python3 -m venv pinn
```

### 2. Attivazione dell'ambiente virtuale

```bash
source pinn/bin/activate
```

### 3. Installazione delle dipendenze

```bash
python3 -m pip install -r requirements.txt
```

## Struttura del progetto

* `pinn/` — ambiente virtuale (non versionato)
* `data/` — dataset (non incluso nel repository)
* `requirements.txt` — dipendenze del progetto
* `README.md` — documentazione

## Note

* La cartella `data/` non è inclusa nel repository. Assicurati di scaricare i dati separatamente.

Per farlo, prima si esegue tutto il codcie decommentando la cella 3 ed eseguendo solo la cella 1 e 3, poi si comamnde il download del dataset nelal cella 3 e si esegue la cella 2 e 3

* L’ambiente virtuale `pinn/` è ignorato da Git.

## Autore

Matteo Aruta

## Plot della soluzione

È disponibile uno script di utilità `plot_solution.py` per visualizzare gli snapshot della soluzione.

Esempi:

 - Usare la prima traiettoria Dynabench caricata da `esperimento.crea_esperimento()` e mostrare il frame 10:

```bash
python plot_solution.py --index 10
```

 - Caricare un file HDF5 Dynabench e salvare la figura del frame 5:

```bash
python plot_solution.py --input data/advection/cloud/low/advection_test_cloud_low_0_499.h5 --index 5 --out snapshot5.png
```

 - Specificare un tempo reale invece dell'indice (trova lo snapshot più vicino):

```bash
python plot_solution.py --input snapshots.npz --time 50.0
```

Requisiti aggiuntivi:

- `matplotlib`
- `h5py` (solo se si vogliono aprire file .h5)

## Esperimento PINN minimale

`esperimento.py` e' minimale: carica solo la prima simulazione Dynabench
`cloud/high` dello split train, allena la PINN con condizione iniziale,
periodicita' del dominio, fisica dell'advection e, se attivata, loss DMD, e
stampa una validation MSE su 200 punti Dynabench campionati dalla simulazione.
Se i file `cloud/high` non sono presenti in `data/`, lo script prova a
scaricarli automaticamente.

La loss iniziale/bordo e' la somma tra lo snapshot iniziale e i vincoli
periodici `u(0,y,t)=u(1,y,t)` e `u(x,0,t)=u(x,1,t)`.
La condizione iniziale usa tutti i punti disponibili del primo snapshot
Dynabench high resolution e viene calcolata sui punti reali `sample.pos`, senza
ricostruire una griglia fittizia.

La DMD loss e' disattivata di default per evitare di usare operatori low
resolution sui punti high. Se la riattivi con `--lambda-dmd 1`, passa un
operatore DMD coerente con high resolution tramite `--dmd-operator`.

Esempio:

```bash
python esperimento.py --evals 100 --save-model results/pinn_sim000.pt
```

Di default esegue 100 iterazioni, stampa ogni 10 iterazioni e valida ogni 10
iterazioni. L'early stopping controlla `validation_mse_200` e si ferma dopo 20
validation senza miglioramento. Puoi cambiare la densita' dei punti PDE, dei
punti periodici, delle transizioni DMD, della validation e la pazienza:

```bash
python esperimento.py --batch-collocation 16384 --batch-periodic 4096 --batch-dmd 16 --validation-points 200 --early-stopping-patience 20
```
