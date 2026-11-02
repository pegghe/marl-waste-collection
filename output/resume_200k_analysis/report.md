# Analisi ripresa di 200.000 passi — 30 ottobre 2026

La nuova models/ppo_drone/runs/run_005 riprende models/ppo_drone/runs/border_penalty/best_model.zip. Il contatore passa da 401.408 a 602.112: 200.704 transizioni aggiunte, perché PPO completa rollouts interi. Learning rate effettivo: 0,0001; n_envs=1. Osservazioni di 219 valori; reward e penalità attuali invariati.

## Training

| Checkpoint | Passi cumulativi | Reward medio | Successo | Durata media |
| --- | ---: | ---: | ---: | ---: |
| Sorgente, border_penalty/best | 401.408 | -19,67 | 72% | 824,25 |
| Nuovo run_005/best | 505.856 | -28,91 | 68% | 904,43 |
| Nuovo run_005/final | 602.112 | -34,83 | 63% | 995,88 |

Le percentuali di training sono medie mobili sugli ultimi 100 episodi, con policy in aggiornamento. All’inizio della ripresa, il buffer delle statistiche caricato dal checkpoint può ancora includere episodi precedenti. La selezione del best ricomincia e richiede 100 nuovi episodi. Le metriche di training non sono una prova indipendente.

La cartella tensorboard_logs/drone_RolloutPPO_drone_run_005_0 contiene sia il vecchio training da un milione di passi, ora denominato border_penalty, sia la nuova ripresa da 200k. I contatori di passo si sovrappongono. Per questa analisi ogni event file è stato letto separatamente; i file originali non sono stati modificati.

## Valutazione controllata

Usato ma_evaluate del professore, con RobotMAEnv, finish_on_discovery=True e gli stessi seed ambiente 0–19. Azioni del drone stocastiche; robot con policy fissa deterministica, come durante il training MASB3. Ogni episodio termina alla prima scoperta o a 2.000 passi. Seed Torch 1000 + seed ambiente a ogni reset. Le osservazioni non vengono normalizzate.

| Modello | Trovate | Reward medio | Passi medi inclusi timeout | Velocità media pesata | Comandi limitati dal bordo |
| --- | ---: | ---: | ---: | ---: | ---: |
| source_best | 12/20 | -51.25 | 953.40 | 0.474 m/s | 21.24% |
| resumed_best | 13/20 | -34.14 | 961.95 | 0.504 m/s | 7.96% |
| resumed_final | 14/20 | -16.50 | 742.70 | 0.505 m/s | 0.21% |

Su questi 20 scenari la nuova policy finale migliora il reward, riduce molto i comandi bloccati e mantiene la velocità. I rilevamenti aumentano di due episodi; il campione è troppo piccolo per stabilire un miglioramento generale del successo. Restano sei timeout. Nel test il finale supera il best selezionato sulle medie di training, mostrando il valore della valutazione separata. Prima di altro training lungo, ampliare la valutazione su altri seed e monitorare successi, tempi di scoperta, copertura e clipping. Il campione usato qui è già stato osservato per confrontare i modelli; per la stima finale delle prestazioni usare seed ulteriori.

[Grafici separati per sessione](tensorboard_comparison.png)

## Checkpoint valutati

- source_best: `models/ppo_drone/runs/border_penalty/best_model.zip`, SHA256 `c7d6c6e7eb50462310d5512367b32aad6d07bd363ba07fa47439addd9e345897`.
- resumed_best: `models/ppo_drone/runs/run_005/best_model.zip`, SHA256 `74762cd892d523b174937909bdefeeb4c0cfc5e94f9a361a7674add820f0b5d4`.
- resumed_final: `models/ppo_drone/runs/run_005/final_model.zip`, SHA256 `afea49289e2a894337a9cbc226675d8aca88c8a35842e02b57ec898d279fa4fc`.
