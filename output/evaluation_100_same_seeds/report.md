# Valutazione su 100 episodi con gli stessi seed

Confronto del best sorgente (border_penalty/best_model.zip) con i checkpoint best e final della nuova run_005, dopo la ripresa di 200k passi. Sono stati eseguiti 100 episodi per modello, 300 in totale. Nessun checkpoint o codice di training è stato modificato.

## Metodo

- Valutazione attraverso ma_evaluate del professore e RobotMAEnv.
- Seed ambiente 100–199 per ogni modello: ogni episodio ha uno scenario diverso, condiviso fra i tre checkpoint. Questi seed sono diversi dai precedenti 0–19.
- Verificata l’uguaglianza esatta delle coordinate iniziali di drone, robot e bottiglia in tutti i 100 confronti.
- A ogni episodio il generatore Torch viene reimpostato a 1000 + seed ambiente, quindi 1100–1199, per usare lo stesso flusso di rumore delle azioni fra checkpoint.
- Azioni del drone stocastiche; policy del robot fissa e deterministica, come nel training MASB3. I pesi di entrambi gli agenti restano fermi durante la valutazione.
- Episodi terminati al primo rilevamento o dopo 2.000 passi. Successo significa scoperta della bottiglia, senza proseguire la navigazione del robot.
- Reward originali dell’ambiente e osservazioni senza normalizzazione; comportamento coerente con norm_obs=False dei checkpoint.

## Risultati

| Modello | Successi | Reward medio | Passi medi con timeout | Velocità media pesata | Passi limitati dal bordo |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sorgente best | 75/100 | -28.22 | 851.90 | 0.532 m/s | 10.22% |
| Nuovo best | 74/100 | -26.35 | 873.60 | 0.529 m/s | 6.66% |
| Nuovo final | 76/100 | -18.45 | 807.99 | 0.550 m/s | 3.50% |

Il final ha stime medie favorevoli per reward, durata e clipping, ma 76 successi contro 75 della sorgente non dimostrano un miglioramento generale della probabilità di scoperta. Il precedente confronto su 20 episodi mostrava 14 contro 12 e dava un’impressione più marcata.

## Confronti appaiati con la sorgente

- Nuovo best: 13 scenari risolti in più e 14 scenari precedentemente risolti diventati fallimenti. Differenza nel successo: -1 punti percentuali; intervallo bootstrap appaiato al 95%: [-11, 9]. Differenza di reward medio: +1.87, intervallo [-10.51, 14.07].
- Nuovo final: 10 scenari risolti in più e 9 scenari precedentemente risolti diventati fallimenti. Differenza nel successo: +1 punti percentuali; intervallo bootstrap appaiato al 95%: [-8, 10]. Differenza di reward medio: +9.77, intervallo [-1.06, 20.60].

Gli intervalli della differenza nel successo e nel reward includono zero per entrambi i checkpoint. Nel campione il finale riduce il clipping dal 10,22% al 3,50% e mantiene la velocità. La ricerca rimane incompleta in 24 episodi su 100. Il finale è un candidato pratico per studiare il miglioramento del movimento, mentre il tasso di scoperta appare simile a quello della sorgente. Questi risultati riguardano checkpoint di un singolo training e un flusso di rumore per scenario.

Intervalli delle differenze ottenuti ricampionando in coppia i 100 seed, 20.000 volte, con generatore bootstrap inizializzato a 20261030.

[Grafico](comparison.png) · [Confronti numerici](comparison.json) · [Dati per episodio in CSV](episodes.csv)

## Riproduzione

Dalla directory del progetto, usare lo script output/resume_200k_analysis/evaluate_checkpoints.py. Esempio per il finale:

```bash
python output/resume_200k_analysis/evaluate_checkpoints.py resumed_final models/ppo_drone/runs/run_005/final_model.zip --episodes 100 --seed-start 100 --output-dir output/evaluation_100_same_seeds
```

Usare gli stessi argomenti --episodes e --seed-start per gli altri checkpoint.

## Checkpoint

- source_best: `models/ppo_drone/runs/border_penalty/best_model.zip`, SHA256 `c7d6c6e7eb50462310d5512367b32aad6d07bd363ba07fa47439addd9e345897`.
- resumed_best: `models/ppo_drone/runs/run_005/best_model.zip`, SHA256 `74762cd892d523b174937909bdefeeb4c0cfc5e94f9a361a7674add820f0b5d4`.
- resumed_final: `models/ppo_drone/runs/run_005/final_model.zip`, SHA256 `afea49289e2a894337a9cbc226675d8aca88c8a35842e02b57ec898d279fa4fc`.
