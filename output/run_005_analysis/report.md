# Analisi run_005 — 30 ottobre 2026

Run attuale addestrata da zero con osservazione di 219 valori, boundary_penalty=0.2 e stagnation_penalty=0.05. Training completato a 1.001.472 passi. I precedenti report in output/ non descrivono questi checkpoint.

Fonte TensorBoard: tensorboard_logs/drone_RolloutPPO_drone_run_005_0. Le metriche di rollout sono finestre mobili fino a 100 episodi di training, non valutazioni indipendenti. Il reward è quello originale dell’ambiente: VecMonitor precede VecNormalize.

| Punto nel training | Reward medio | Rilevamenti | Durata media |
| --- | ---: | ---: | ---: |
| Best salvato, 401.408 passi | -19,67 | 72% | 824,25 passi |
| Picco di successo, 495.616 passi | -21,06 | 81% | 766,31 passi |
| Finale, 1.001.472 passi | -61,14 | 54% | 1.173,85 passi |

Il best è selezionato per reward medio di training, non per il massimo success_rate. La std delle azioni diminuisce da circa 1 a 0,165; learning_rate resta 0,0003 ed ent_coef=0. Questi dati mostrano minore variabilità, ma non dimostrano da soli la causa del peggioramento.

## Verifica dei checkpoint

20 seed ambiente (0–19), stessi seed per entrambi i checkpoint. Ogni episodio termina al primo rilevamento o dopo 2.000 passi. Robot fermo durante la ricerca, come imposto dal training. Seed Torch reimpostato a 1000 + seed ambiente per episodio. Nessuna normalizzazione delle osservazioni (norm_obs=False).

| Checkpoint / azioni | Rilevamenti | Reward medio | Passi medi inclusi timeout | Velocità media pesata | Passi con clipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| best_model/stochastic | 12/20 | -51.25 | 953.40 | 0.474 m/s | 21.24% |
| best_model/deterministic | 13/20 | -70.48 | 947.50 | 0.225 m/s | 27.87% |
| final_model/stochastic | 9/20 | -72.45 | 1327.50 | 0.255 m/s | 5.52% |
| final_model/deterministic | 7/20 | -109.36 | 1483.05 | 0.136 m/s | 14.41% |

Il campione è esplorativo: non stabilisce una percentuale di successo generale. In questa prova il finale riduce il clipping ma rallenta e rileva meno bottiglie. Il problema non appare risolto e non consiste solo nei bordi. Le metriche riguardano la scoperta del drone; la CLI evaluate attuale prosegue dopo la scoperta e riporta il successo finale del robot, quindi il suo success_rate misura un risultato diverso.

Per la prossima prova scegliere best_model.zip. Prima di prolungare il training, confrontare su episodi separati i checkpoint e valutare aggiornamenti più piccoli (learning rate inferiore), evitando di presumere che più passi migliorino la policy. Nessun codice di training o checkpoint è stato modificato.

[Grafici TensorBoard](tensorboard.png) · [Dati completi della verifica](evaluation.json)

## Identificazione dei checkpoint

- `models/ppo_drone/runs/run_005/best_model.zip` SHA256 `c7d6c6e7eb50462310d5512367b32aad6d07bd363ba07fa47439addd9e345897`
- `models/ppo_drone/runs/run_005/final_model.zip` SHA256 `6cdea5e7c8597d19d7d4354ec73a32891991fb6ac6eca7c8db27f7eac13cab18`
