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
