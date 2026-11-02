# Confronto checkpoint drone

100 episodi per checkpoint, seed ambiente e RNG azioni 0–99, policy stocastiche (`deterministic=False`), CPU, un thread Torch per processo. Robot fisso: `models/ppo_go/final_model.zip`. Ambiente corrente: `RobotMAEnv(drone_training=True, finish_on_discovery=False)`, come nella valutazione standard. Successo: `info["is_success"]` (completamento del compito, non sola scoperta del drone). Nessuna normalizzazione delle osservazioni applicata, come nella valutazione standard.

Verificata l'identità delle posizioni iniziali del robot e dei target per ciascun seed tra tutti i sei checkpoint. Ogni episodio riavvia anche i generatori Python, NumPy e Torch con lo stesso seed. Ogni seed identifica un episodio diverso, comune a tutti i checkpoint.

| Run | Checkpoint | Successi / 100 | Passi medi |
|---|---|---:|---:|
| border_penalty_1.4mln | best_model | 78 | 764.73 |
| border_penalty_1.4mln | final_model | 74 | 898.33 |
| border_penalty | best_model | 69 | 1002.89 |
| border_penalty_+200 | final_model | 69 | 933.07 |
| border_penalty_+200 | best_model | 65 | 996.11 |
| border_penalty | final_model | 48 | 1322.36 |

`run_001` esclusa: entrambi i checkpoint attendono osservazioni (23,), incompatibili con le (219,) attuali. `run_002` e `run_004` non contengono checkpoint best/final.

Tutti i modelli sono valutati nello stesso ambiente attuale; non vengono ricostruite le versioni storiche dell'ambiente o le penalità del training. I risultati precedenti senza seed non sono direttamente accoppiati a questo campione. Il distacco 78% contro 74% su 100 episodi non basta da solo a dimostrare una superiorità stabile.

Dati per episodio nei file JSON, classifica in `summary.csv`. Riproduzione dalla directory del progetto: `/home/rl/miniconda3/envs/rl/bin/python compare_runs.py --output output/comparison_seed_0_99_repeat`. Una cartella di output già esistente riutilizza i risultati salvati con lo stesso numero di episodi.
