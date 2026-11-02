# Ultima run_005: analisi e confronto su 100 episodi

La run attuale riparte da models/ppo_drone/runs/border_penalty/best_model.zip, non dal finale della prova da 200k. Aggiunge 1.001.472 passi ai 401.408 del checkpoint, arrivando a 1.402.880. Learning rate 0,0001, un ambiente, osservazioni di 219 valori. Il vecchio training da 200k è ora sotto border_penalty_+200.

## TensorBoard

Letto soltanto events.out.tfevents.1793385340.HPMatteo.55026.0: la directory drone_RolloutPPO_drone_run_005_0 contiene altre due sessioni con contatori sovrapposti.

| Checkpoint | Passi cumulativi | Reward medio di training | Successo di training | Durata media |
| --- | ---: | ---: | ---: | ---: |
| Ultimo best | 878.592 | -10,80 | 85% | 631,58 |
| Ultimo final | 1.402.880 | -34,28 | 68% | 957,91 |

Le medie di training usano gli ultimi 100 episodi e includono diverse versioni della policy. Il best è selezionato per reward medio. La variabilità delle azioni scende da circa 0,347 a 0,127; la riduzione non dimostra da sola la causa del peggioramento.

## Confronto controllato

Usato ma_evaluate del professore con RobotMAEnv e finish_on_discovery=True. Per ogni modello sono stati eseguiti gli stessi seed ambiente 100–199 e reimpostato Torch a 1000 + seed per episodio. Azioni del drone stocastiche, robot con policy fissa deterministica. Episodi terminati al rilevamento o dopo 2.000 passi. Verificata l’uguaglianza esatta delle coordinate iniziali di drone, robot e bottiglia. I risultati del finale precedente da 200k sono stati riprodotti esattamente.

| Modello | Trovate | Reward medio | Passi medi con timeout | Velocità media pesata | Passi limitati dal bordo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final precedente +200k | 76/100 | -18.45 | 807.99 | 0.550 m/s | 3.50% |
| Ultimo best | 81/100 | -21.73 | 706.93 | 0.527 m/s | 13.40% |
| Ultimo final | 67/100 | -41.58 | 910.71 | 0.413 m/s | 17.79% |

Su questi scenari il best dell’ultima run trova più bottiglie e termina gli episodi prima del finale della stessa run. Il finale perde risultati sia rispetto al best sia rispetto alla precedente prova da 200k. Il best ha cinque rilevamenti in più del finale precedente da 200k, ma peggiora il clipping e ha un reward medio più basso: i diversi obiettivi non migliorano tutti insieme. Per studiare il tasso di rilevamento, il nuovo best è il candidato con la stima più alta nel campione; il vantaggio rispetto al precedente 76/100 rimane incerto.

## Differenze appaiate

- latest_best_vs_previous_200k_final: 13 successi guadagnati e 8 persi. Differenza di successo +5 punti; intervallo bootstrap appaiato al 95% [-4, 14]. Differenza reward -3.29, intervallo [-17.12, 9.44].
- latest_final_vs_previous_200k_final: 9 successi guadagnati e 18 persi. Differenza di successo -9 punti; intervallo bootstrap appaiato al 95% [-19, 1]. Differenza reward -23.14, intervallo [-36.66, -9.97].
- latest_best_vs_latest_final: 22 successi guadagnati e 8 persi. Differenza di successo +14 punti; intervallo bootstrap appaiato al 95% [4, 24]. Differenza reward +19.85, intervallo [5.00, 34.96].

Il confronto usa 100 scenari fissi già utilizzati per scegliere fra più modelli: sono un insieme di validazione. Per stimare le prestazioni finali dopo la selezione servono ulteriori scenari non usati per scegliere il checkpoint. Nessun codice di training o checkpoint è stato modificato.

[Grafici TensorBoard](tensorboard_latest.png) · [Grafico di valutazione](evaluation_comparison.png) · [Dati numerici](comparison.json) · [Episodi CSV](episodes.csv) · [Identificazione checkpoint e ambiente](metadata.json)
