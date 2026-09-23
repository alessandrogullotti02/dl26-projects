# Atari Pong — DQN, Double DQN e Reward Shaping

Progetto di Deep Learning / Reinforcement Learning su `ALE/Pong-v5` con Vanilla DQN, Double DQN, ablation della schedule epsilon-greedy, tre famiglie di reward shaping e diagnostiche post-hoc della policy.

## Stato sperimentale

Tutte le famiglie considerate nel report dispongono di tre training indipendenti

- Tutte le famiglie nel confronto quantitativo primario: seed `42, 2, 3`

Le statistiche `media ± std` riportate nel report sono calcolate sulle **medie di evaluation dei training seed**, usando la deviazione standard campionaria tra seed (`ddof=1`). Non sono medie delle deviazioni standard tra episodi

## Setup

```bash
conda env create -f environment.yml
conda activate pong-dqn
```

oppure

```bash
python -m pip install -r requirements.txt
```

## Training modulare

Esempio di dry-run, che valida configurazione e parametri senza avviare il training

```bash
python -m src.training.run --config experiments/configs/baseline.json --dry-run
```

Training completi

```bash
python -m src.training.run --config experiments/configs/baseline.json
python -m src.training.run --config experiments/configs/ddqn.json
python -m src.training.run --config experiments/configs/epsilon.json
python -m src.training.run --config experiments/configs/rally.json
python -m src.training.run --config experiments/configs/potential.json
python -m src.training.run --config experiments/configs/contact.json
```

Per eseguire un singolo seed in una sessione separata

```bash
python -m src.training.run \
  --config experiments/configs/rally.json \
  --seed 3 \
  --output-dir experiments/logs/rally_seed3
```

## Rigenerazione delle figure

Curve DQN e Double DQN, più il confronto paired al checkpoint finale, dai risultati machine-readable. Lo script aggiorna direttamente le figure canoniche in `figures/report/`.

```bash
python scripts/plot_dqn_ddqn_multiseed.py
```

Curve aggregate di epsilon, rally, PBRS e contact direttamente dai 12 notebook eseguiti

```bash
python scripts/plot_multiseed_from_notebooks.py
```

In alternativa le stesse quattro curve possono essere rigenerate dal JSON aggregato

```bash
python scripts/plot_multiseed_results.py
```

Le sole figure mantenute nel repository sono quelle citate nella relazione o utilizzate nella presentazione; sono raccolte in `figures/report/`.

## Risultati finali principali

| Esperimento | Return finale medio ± std tra seed |
|---|---:|
| Vanilla DQN | 12.77 ± 2.05 |
| Double DQN | 15.88 ± 1.93 |
| Epsilon esponenziale | 14.88 ± 1.92 |
| Rally moderato | 15.18 ± 0.90 |
| PBRS `kappa=0.50` | 16.42 ± 0.83 |
| Contact `c=0.10` | 14.57 ± 2.04 |

Con soli tre training seed, questi confronti restano descrittivi e non costituiscono evidenza inferenziale forte. Il confronto primario è però ora paired seed-by-seed perché tutte le famiglie usano lo stesso insieme `{42, 2, 3}`. Per Double DQN rispetto alla baseline, i delta finali sui tre seed sono `+4.80`, `+1.45` e `+3.10`, con media `+3.12` e deviazione standard campionaria `1.68`.

## Struttura del repository

- `src/` implementazione modulare dell'ambiente, DQN, training, reward shaping ed evaluation
- `experiments/configs/` configurazioni riproducibili delle sei suite
- `notebooks/` notebook eseguiti principali, mantenuti come evidenza sperimentale; il codice di training canonico risiede in `src/`
- `notebooks/replicates/` repliche seed `2` e `3` necessarie al confronto multi-seed
- `scripts/` script per rigenerare le visualizzazioni multi-seed
- `docs/REPORT.md` relazione organizzata secondo il template del corso
- `docs/results/` metriche aggregate e dati machine-readable; `seed42_checkpoint_additions.json` contiene solo i valori numerici importati dalle due run aggiuntive
- `figures/report/` figure canoniche citate nella relazione, inclusa la visualizzazione paired DQN/DDQN
- `data/` placeholder: i dati di training sono generati online da ALE e non esiste un dataset statico da versionare

## Documentazione

- [Report](docs/REPORT.md)
- [Risultati](docs/results/SUMMARY.md)
- [Presentazione](docs/Pong_presentazione.pptx)

