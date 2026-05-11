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

## Operatori Koopman-piDMD per la loss DMD

La loss DMD usa gli operatori generati dal notebook
`Copia_di_05_koopman_pidmd_pipeline_FINAL.ipynb`. Per rigenerarli da script:

```bash
python koopman_pidmd.py --split train --out-dir results/per_sim_final
```

Lo script salva file del tipo:

```text
results/per_sim_final/koopman_pidmd_r58_sim00000.npz
```

`esperimento.py` ha `lambda_dmd=0` di default, quindi il vincolo DMD e' spento
nel training principale. Se lo riattivi con `--lambda-dmd 1`, usa questa
directory come sorgente DMD di default. Se durante
il training manca un operatore per una simulazione richiesta, lo genera con la
stessa pipeline del notebook `Copia_di_05_koopman_pidmd_pipeline_FINAL.ipynb`
e lo salva nella directory DMD. Per disattivare questo comportamento:

```bash
python esperimento.py --lambda-dmd 1 --no-dmd-auto-generate
```

Esempio di training principale con modello CNN full-field, fisica
differenziale spenta, DMD spenta e ottimizzazione AdamW/AMSGrad con OneCycleLR:

```bash
python esperimento.py --evals 5000 --save-model results/cnn_advection.pt
```

Il default usa `--model cnn`: i 225 punti cloud vengono interpolati su una
griglia regolare `15x15`, la CNN predice i 16 step futuri sulla griglia, poi
l'output viene riportato sui punti cloud per calcolare la loss. La vecchia PINN
pointwise resta disponibile con:

```bash
python esperimento.py --model pinn
```

Con `--model cnn` tieni `--lambda-fisica 0`, perche' il termine PDE pointwise
richiede la vecchia architettura PINN.

L'ottimizzatore di default usa `adamw_amsgrad`, `max_lr=1e-3`,
`weight_decay=1e-5`, `betas=(0.9, 0.99)`, scheduler `onecycle`, clipping del
gradiente a `1.0`, `batch_size=32`, CNN con `64` canali e `6` blocchi
residuali. Puoi cambiarli da CLI con `--optimizer`, `--learning-rate`,
`--weight-decay`, `--lr-scheduler`, `--clip-grad-norm`, `--hidden-dim`,
`--num-layers`, `--dropout` e `--cnn-neighbors`.

Se riattivi la DMD, resta calcolata con l'operatore completo `225x225`, ma la
loss viene mediata solo sui punti indicati da `--dmd-punti-per-snapshot`.
