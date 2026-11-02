# Stato del progetto per la tesi — 2 novembre 2026

Questo documento ricostruisce lo stato verificabile del progetto `marl-waste-collection` dal codice attuale, dal README, dalle configurazioni delle run, dai metadati dei checkpoint e dal confronto salvato in `output/comparison_seed_0_99`. Non costituisce una ricostruzione completa delle modifiche storiche: non tutte le versioni precedenti dell'ambiente sono archiviate. I risultati numerici riguardano il protocollo specificato qui sotto.

## 1. Obiettivo e compito effettivamente implementato

Il progetto realizza una cooperazione fra un drone e un robot terrestre in MuJoCo. Il drone esplora l'area, identifica un rifiuto rappresentato da una bottiglia e comunica la sua posizione al robot. Dopo la comunicazione, il robot raggiunge il target.

La raccolta è attualmente rappresentata dal raggiungimento della bottiglia entro una soglia spaziale: non sono implementate la presa, la manipolazione o il trasporto del rifiuto. Il contesto applicativo è la raccolta di rifiuti, ma la misura sperimentale corrente è il completamento della sequenza ricerca–comunicazione–raggiungimento.

## 2. Ambiente simulato

- Simulazione fisica MuJoCo, interfaccia Gymnasium e coordinamento multiagente MASB3.
- Pavimento rappresentato visivamente su 8 × 8 m; limiti operativi di posizione XY pari a ±3,5 m.
- Spawn del robot, del target e, in modalità drone, del drone nel quadrato [-3, 3] × [-3, 3] m.
- Distanza iniziale robot–target almeno pari a 0,75 m.
- Timestep fisico 0,01 s e frame skip 5: ogni azione corrisponde a 0,05 s simulati.
- Durata massima dell'episodio: 2.000 passi, pari a 100 s simulati.
- Drone a quota fissa di 3 m, con movimento nel piano XY.
- Il robot ha due giunti traslazionali XY; l'aspetto quadrupede è decorativo. Non si apprende una locomozione articolata delle zampe.

## 3. Architettura software

`scene.xml` definisce la scena, gli oggetti e gli attuatori. `agents.py` separa la logica del robot e del drone nelle classi `RobotAgent` e `DroneAgent`. `waste_collection_env.py` contiene la simulazione condivisa `WasteCollectionEnv` e l'interfaccia MASB3 `RobotMAEnv`. `raycast.py` implementa il supporto ai raggi sensoriali. `robot_experiment.py` gestisce training, ripresa, selezione e valutazione dei checkpoint. `compare_runs.py` esegue il confronto riproducibile dei checkpoint drone.

I due agenti agiscono nella stessa simulazione; non si tratta di due ambienti fisici separati.

## 4. Robot terrestre

L'osservazione è un vettore di 7 elementi:

`[x, y, vx, vy, target_x - x, target_y - y, go]`.

Le azioni sono due comandi motore continui in [-1, 1], lungo gli assi X e Y.

Nel compito robot-only, il robot conosce fin dall'inizio la posizione relativa del target. Il segnale `go` rimane inizialmente a zero per un ritardo casuale estratto fra 0 e 100 passi; successivamente autorizza la navigazione. In questa modalità, la permanenza vicino al punto di partenza è incentivata dal reward.

Il reward corrente del robot comprende:

- penalità temporale -0,01 a passo;
- con `go=1`, penalità aggiuntiva pari alla distanza euclidea dal target;
- con `go=0`, penalità pari alla velocità e a cinque volte la distanza dal punto iniziale;
- bonus +10 al raggiungimento del target.

Il successo richiede `go=1` e distanza dal target non superiore a 0,25 m.

Il robot utilizzato nelle valutazioni cooperative è `models/ppo_go/final_model.zip`. I suoi metadati riportano 600.064 timestep e learning rate 0,0003. Il codice e il README descrivono l'evoluzione del reward, ma senza una versione storica completa non si attribuisce automaticamente a tutti i checkpoint preesistenti la funzione di reward attuale.

## 5. Drone: movimento e sensori

Il drone riceve due azioni continue in [-1, 1]. Il comando determina lo spostamento XY con limite di 1 m/s per asse e passo massimo di 0,05 m per asse. Con entrambe le componenti al massimo, il modulo della velocità può raggiungere circa 1,414 m/s. La posizione viene limitata a [-3,5, 3,5] m.

Il drone impiega nove raggi diretti verso il terreno: alla quota iniziale, le direzioni corrispondono a una griglia 3 × 3 con spaziatura di 0,6 m. La portata massima dei raggi è 5 m. Per ciascun raggio sono disponibili distanza e indicazione dell'intersezione con una geometria della bottiglia.

Il rilevamento avviene tramite ray casting, non tramite immagini o una rete di visione. Il simulatore permette di associare la geometria colpita al target. Al rilevamento, la posizione del target viene fornita al canale di comunicazione dal simulatore: non è implementata una stima rumorosa della posizione a partire dai raggi.

## 6. Osservazione e memoria di esplorazione del drone

L'osservazione corrente comprende 219 valori float32:

| Indici | Contenuto | Dimensione |
|---|---|---:|
| 0:3 | Comunicazione: trovato, target_x/5, target_y/5 | 3 |
| 3:5 | Posizione XY del drone divisa per 5 | 2 |
| 5:23 | Nove coppie distanza normalizzata / indicatore bottiglia | 18 |
| 23:219 | Mappa binaria delle celle visitate | 196 |

La mappa ha dimensione 14 × 14 e celle di lato 0,5 m, coprendo il dominio [-3,5, 3,5] m. All'inizio è marcata soltanto la cella di partenza. Una cella diventa visitata quando viene attraversata dal drone.

Questa mappa fornisce una memoria esplicita dell'esplorazione. Registra le posizioni visitate, non tutta l'area osservata dai raggi; non rivela la bottiglia prima del rilevamento. Non va descritta come una mappa completa degli ostacoli o come un sistema SLAM.

## 7. Reward corrente del drone

Il reward è composto da incentivi alla ricerca e penalità per comportamenti improduttivi:

- Tempo: -0,05 a passo prima della scoperta, escluso il passo di rilevamento.
- Esplorazione: +0,5 al primo ingresso in una nuova cella dell'episodio; la cella iniziale è già visitata e non genera un bonus iniziale.
- Scoperta: `20 × max(0, 1 - (steps - 1)/2000)`, premiando il rilevamento anticipato.
- Bordo: `-0,2 × blocked_fraction`, proporzionale alla componente di spostamento rifiutata dal limite e limitata in modulo a 0,2.
- Stagnazione: penalità lineare fino a -0,05 quando la velocità effettiva è inferiore a 0,3 m/s; nulla da tale soglia in su.

Le penalità di bordo e stagnazione sono nulle nel passo di scoperta e dopo la scoperta. Dopo la scoperta non vengono più assegnati reward al drone. Un eventuale bonus per una nuova cella può sommarsi al bonus nel passo di rilevamento.

La penalità di bordo riguarda comandi respinti dal clipping, non la sola vicinanza al confine. Restare immobili costa complessivamente -0,10 a passo prima della scoperta. Un comando interamente bloccato verso l'esterno costa complessivamente -0,30 a passo.

L'intento delle penalità è scoraggiare immobilità e comandi improduttivi ai bordi. .

## 8. Comunicazione e cooperazione

Prima del rilevamento, il robot riceve vettore relativo del target nullo e `go=0`. In modalità drone, l'ambiente forza inoltre a zero i comandi motore del robot fino alla comunicazione; l'attesa non dipende soltanto dalla policy appresa.

Il primo rilevamento imposta il messaggio condiviso `[1, target_x/5, target_y/5]`, che persiste fino al reset. Il robot ricostruisce le coordinate, riceve il vettore relativo e passa a `go=1`. Il suo controllo appreso viene utilizzato nelle azioni successive alla scoperta.

L'addestramento è sequenziale: una policy robot già disponibile viene mantenuta congelata mentre si addestra la policy drone. Non si stanno aggiornando contemporaneamente entrambe le policy e non è documentato un algoritmo con critico centralizzato.

## 9. Addestramento e checkpoint

Il training usa PPO di Stable-Baselines3 attraverso MASB3 (`ma_train`, `MultiAgentSharedVecEnv`). La policy è MLP; i checkpoint drone confrontati condividono:

| Parametro | Valore |
|---|---:|
| Seed salvato | 0 |
| Passi per rollout | 2.048 |
| Batch size | 64 |
| Epoche per aggiornamento | 10 |
| Gamma | 0,99 |
| GAE lambda | 0,95 |
| Clip range | 0,2 |
| Coefficiente entropia | 0 |

Il training corrente usa CPU, un thread Torch e `VecNormalize` con normalizzazione dei reward e senza normalizzazione delle osservazioni. Le statistiche vengono salvate e ricaricate quando disponibili. La selezione del best usa il ritorno non normalizzato registrato da `VecMonitor`.

Il codice attuale imposta learning rate 0,0001; i checkpoint della run base `border_penalty` riportano invece 0,0003. Per descrivere esperimenti passati fanno fede i rispettivi metadati, non soltanto il codice presente.

Ogni nuova esecuzione produce una directory distinta. `--resume` riprende un checkpoint e salva la prosecuzione separatamente. I passi richiesti possono essere superati per completare il rollout.

## 10. Provenienza delle tre run confrontabili

| Run | Origine registrata | Passi richiesti nella run | Learning rate salvato | Timestep best | Timestep final |
|---|---|---:|---:|---:|---:|
| border_penalty | Da zero (`resume_from=null`) | 1.000.000 | 0,0003 | 401.408 | 1.001.472 |
| border_penalty_+200 | Best di border_penalty | 200.000 aggiuntivi | 0,0001 | 505.856 | 602.112 |
| border_penalty_1.4mln | Best di border_penalty | 1.000.000 aggiuntivi | 0,0001 | 878.592 | 1.402.880 |

I timestep delle prosecuzioni comprendono il contatore ripreso dal checkpoint sorgente. Entrambe le prosecuzioni derivano dal best della run base, al passo 401.408, e non dal suo final al passo 1.001.472. Non formano una catena base-final → +200-final → 1.4mln-final. Le traiettorie scartate dopo il best della run base non vanno conteggiate come esperienza del checkpoint ripreso.

Le configurazioni delle tre run riportano speed 1,0, min_search_speed 0,3, boundary_penalty 0,2 e stagnation_penalty 0,05. Il nome `1.4mln` si riferisce al budget di addestramento e non a una penalità di bordo pari a 1,4.

## 11. Significato di best e final

`final_model.zip` è lo stato al termine della run. `best_model.zip` viene scelto confrontando la media del ritorno non normalizzato degli ultimi 100 episodi completati durante il training, includendo fallimenti e timeout.

La selezione avviene a fine rollout, prima dell'aggiornamento PPO. Un best viene salvato solo dopo almeno 100 episodi completati; successivamente occorre un miglioramento stretto della media. Non si tratta di una selezione su un insieme separato di episodi di validazione e non massimizza direttamente il success rate cooperativo.

## 12. Distinzione fra successo di training e di valutazione

Durante il training drone, l'episodio termina quando viene scoperta la bottiglia oppure al timeout: il successo misura il rilevamento.

Nella valutazione cooperativa corrente, `finish_on_discovery=False`: dopo la scoperta l'episodio continua fino al raggiungimento del target da parte del robot oppure al timeout. Il successo richiede quindi ricerca, comunicazione e navigazione completate entro il budget condiviso di 2.000 passi. Non viene assegnato un nuovo budget di 2.000 passi alla navigazione.

Di conseguenza il success rate del training e quello della valutazione non rappresentano la stessa metrica. Un fallimento cooperativo può dipendere dalla ricerca, dalla navigazione o dal tempo residuo insufficiente.

## 13. Valutazione standard e confronto controllato

La valutazione standard usa `ma_evaluate`. Il numero di episodi è impostato da `--episodes` (default 10). `--agent drone` abilita il sistema cooperativo; `--model` seleziona la run; `--best` seleziona il best. Senza modello esplicito si sceglie il final modificato più recentemente fra le run completate, non il migliore per success rate.

`ma_evaluate` chiama `reset()` senza seed e `predict()` senza specificare `deterministic`: PPO usa quindi `deterministic=False`. La policy campiona dalla distribuzione appresa. Con `deterministic=True`, per le azioni continue PPO usa invece la media di tale distribuzione; questo non elimina la variabilità degli scenari iniziali e non garantisce un miglioramento del successo. Non è stata ancora effettuata qui una campagna di confronto con azioni deterministiche.

Per confrontare i checkpoint è stato creato `compare_runs.py`, mantenendo le azioni stocastiche e il robot fisso. Ogni checkpoint è stato valutato su 100 episodi: al primo episodio seed 0, al secondo seed 1, fino al seed 99. A ogni episodio sono stati reimpostati anche i generatori Python, NumPy e Torch con il seed corrispondente.

Tutti i modelli hanno quindi incontrato la stessa sequenza di scenari. Sono state verificate esplicitamente le posizioni iniziali del robot e del target fra i sei checkpoint. Il reset del drone usa lo stesso generatore dell'ambiente. Stesso seed non implica stesse azioni: le policy sono differenti e reagiscono alle proprie osservazioni.

Sono stati completati 600 episodi: sei checkpoint per 100 episodi. Tutti hanno usato l'ambiente corrente; non sono state ricostruite le versioni storiche della simulazione. Nel confronto originario `run_001` è stata esclusa per input 23 anziché 219 e `run_002`/`run_004` non contenevano checkpoint best/final.

## 14. Risultati

| Run | Checkpoint | Successi / 100 | Success rate | Passi medi su tutti gli episodi |
|---|---|---:|---:|---:|
| border_penalty_1.4mln | best | 78 | 78% | 764,73 |
| border_penalty_1.4mln | final | 74 | 74% | 898,33 |
| border_penalty | best | 69 | 69% | 1.002,89 |
| border_penalty_+200 | final | 69 | 69% | 933,07 |
| border_penalty_+200 | best | 65 | 65% | 996,11 |
| border_penalty | final | 48 | 48% | 1.322,36 |

I passi medi includono i fallimenti, che possono raggiungere il limite di 2.000 passi: non sono tempi medi condizionati al successo e non rappresentano la sola durata della ricerca.

Il miglior risultato osservato sul campione è il 78% del best di `border_penalty_1.4mln`. Nella run base, il final ottiene meno successi del best; un addestramento più lungo non ha quindi prodotto automaticamente un checkpoint con risultato migliore su questo test. Nella run +200, il final supera il best scelto dal ritorno di training: i criteri di selezione e di valutazione sono diversi.

Le prosecuzioni non sono repliche indipendenti e il campione contiene solo 100 episodi per modello. Non si conclude una superiorità statistica stabile del 78% sul 74% e non si attribuisce il risultato a una singola modifica del reward.

In una precedente valutazione manuale era stato riportato il 65% per il best della run 1.4mln. Il confronto controllato ha prodotto 78% con la sequenza 0–99. La precedente esecuzione non aveva seed espliciti; senza il suo log completo e la conferma del checkpoint non si può attribuire con certezza l'intera differenza alla sola casualità. Le due percentuali descrivono esecuzioni distinte e non devono essere fuse in un unico risultato.

## 15. Materiale disponibile e limiti aperti

Risultati per episodio: `output/comparison_seed_0_99/*.json`. Tabella aggregata: `output/comparison_seed_0_99/summary.csv`. Report del confronto: `output/comparison_seed_0_99/report.md`. Configurazioni e metriche di selezione: directory delle singole run. Andamento del training: log TensorBoard.

Sono da distinguere dai risultati già ottenuti le possibili estensioni: valutazione su ulteriori seed, confronto `deterministic=False/True`, misura separata del tasso e del tempo di scoperta, analisi dei fallimenti del robot dopo la comunicazione, repliche di training con seed diversi e confronti controllati dei termini di reward. Non sono ancora documentati qui risultati per tali esperimenti.

Prima di una nuova campagna conviene congelare una copia della versione dell'ambiente e associare ai risultati gli hash dei checkpoint e le versioni delle dipendenze. Le percentuali presenti descrivono lo stato verificato dal confronto già salvato.
