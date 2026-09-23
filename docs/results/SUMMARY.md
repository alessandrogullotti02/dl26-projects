# Risultati multi-seed aggiornati

Tutte le famiglie sperimentali nel confronto quantitativo primario usano gli stessi **tre training seed `42, 2, 3`**. Le statistiche riportate come `media ± std` sono la media e la deviazione standard campionaria **tra le medie di evaluation delle tre run**, non la deviazione tra episodi della singola rete.

## Risultati finali

| Esperimento | Variante | Return finale medio ± std tra seed |
|---|---|---:|
| Vanilla DQN | baseline | **12.77 ± 2.05** |
| Double DQN | DDQN | **15.88 ± 1.93** |
| Epsilon | lineare | 11.48 ± 4.04 |
| Epsilon | esponenziale | **14.88 ± 1.92** |
| Epsilon | costante 0.10 | 8.37 ± 9.17 |
| Rally | piccolo | 13.57 ± 1.63 |
| Rally | moderato | **15.18 ± 0.90** |
| Rally | aggressivo | 1.07 ± 19.66 |
| PBRS | kappa 0.02 | 14.33 ± 1.27 |
| PBRS | kappa 0.10 | 15.08 ± 1.42 |
| PBRS | kappa 0.50 | **16.42 ± 0.83** |
| Contact | c=0.02 | 12.23 ± 3.51 |
| Contact | c=0.10 | **14.57 ± 2.04** |
| Contact | c=0.50 | -2.42 ± 16.09 |

## Confronto paired Vanilla DQN / Double DQN

Al checkpoint finale da 1.8M step, sui seed comuni `42, 2, 3`, i delta Double DQN meno Vanilla DQN sono rispettivamente `+4.80`, `+1.45` e `+3.10`. La differenza paired media è `+3.12`, con deviazione standard campionaria `1.68` tra i tre delta. Con `n=3` il dato resta descrittivo.

Le diagnostiche post-hoc di loss/target gap per Double DQN e TD error/Q-value/heatmap per la baseline provengono dai modelli eseguiti nei notebook con seed `1, 2, 3`; non sono usate per le statistiche quantitative primarie.

## Interpretazione sintetica

- **Epsilon:** la schedule esponenziale ha la media finale più alta e una dispersione finale contenuta; la costante è fortemente instabile tra seed.
- **Rally:** il bonus moderato è il più stabile; l'aggressivo mostra una varianza estrema perché una replica collassa a `-21`.
- **PBRS:** la scala `0.50` ottiene il miglior finale medio, pur con apprendimento intermedio lento e variabile.
- **Contact:** `c=0.10` è il compromesso più robusto; `c=0.50` è molto instabile e una replica collassa a `-21`.

Dati completi: `multiseed_results.json`, `checkpoint_metrics.json` e `paired_seed_results.json`.
