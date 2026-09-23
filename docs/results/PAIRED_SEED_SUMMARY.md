# Confronto finale sui training seed comuni {42, 2, 3}

Tutte le famiglie nel confronto quantitativo primario usano gli stessi training seed `42, 2, 3`. La deviazione standard è la deviazione standard campionaria tra le tre medie di evaluation (`ddof=1`).

| Famiglia | Configurazione | Seed 42 | Seed 2 | Seed 3 | Media ± std | Δ paired medio vs Vanilla |
|---|---|---:|---:|---:|---:|---:|
| baseline | Vanilla DQN | 12.80 | 14.80 | 10.70 | 12.77 ± 2.05 | +0.00 |
| ddqn | Double DQN | 17.60 | 16.25 | 13.80 | 15.88 ± 1.93 | +3.12 |
| Epsilon | Lineare | 12.80 | 14.70 | 6.95 | 11.48 ± 4.04 | -1.28 |
| Epsilon | Esponenziale | 17.10 | 13.75 | 13.80 | 14.88 ± 1.92 | +2.12 |
| Epsilon | Costante 0.10 | -2.20 | 14.20 | 13.10 | 8.37 ± 9.17 | -4.40 |
| Rally | Piccolo | 15.35 | 12.15 | 13.20 | 13.57 ± 1.63 | +0.80 |
| Rally | Moderato | 16.10 | 15.15 | 14.30 | 15.18 ± 0.90 | +2.42 |
| Rally | Aggressivo | 16.70 | -21.00 | 7.50 | 1.07 ± 19.66 | -11.70 |
| PBRS | 0.02 | 12.95 | 14.60 | 15.45 | 14.33 ± 1.27 | +1.57 |
| PBRS | 0.10 | 16.05 | 13.45 | 15.75 | 15.08 ± 1.42 | +2.32 |
| PBRS | 0.50 | 15.50 | 16.65 | 17.10 | 16.42 ± 0.83 | +3.65 |
| Contact | 0.02 | 8.20 | 13.95 | 14.55 | 12.23 ± 3.51 | -0.53 |
| Contact | 0.10 | 16.25 | 15.15 | 12.30 | 14.57 ± 2.04 | +1.80 |
| Contact | 0.50 | 6.95 | -21.00 | 6.80 | -2.42 ± 16.09 | -15.18 |

Per Double DQN rispetto alla baseline, i delta paired finali sono `+4.80`, `+1.45` e `+3.10`; la loro media è `+3.12` e la deviazione standard campionaria è `1.68`.

Con `n=3` il confronto resta descrittivo. Il pairing elimina la differenza di seed tra famiglie, ma non rende il campione statisticamente ampio.

