# Bozza di progetto – Footsies RL

Oct 28, 2026

## Obiettivo e domanda di ricerca

Il progetto addestra agenti per Footsies con un solo algoritmo, PPO in self-play, e studia come la funzione di reward cambia ciò che l'agente impara, misurandone forza e robustezza, non solo il win rate.

La domanda guida è: **un agente che vince contro avversari semplici è anche difficile da sfruttare?** Il paper di FootsiesGym mostra che le due cose possono divergere (un metodo può battere gli altri in scontro diretto ed essere comunque il più sfruttabile). Il progetto riproduce in piccolo questa analisi, cambiando la funzione di reward con lo stesso algoritmo, e la collega alla teoria del corso (modellazione, bias e varianza).

Il punto di partenza sono le due baseline PPO del paper, entrambe con il reward terminale: PPO con entropia fissa (0,025) e PPO (Sched.) con entropia che scende da 0,1 a 0. Sopra queste, il confronto principale è tra quattro funzioni di reward con lo stesso PPO, scritto da noi in PyTorch:

1. **Reward terminale**: ±w alla vittoria o sconfitta, 0 al timeout (baseline).
2. **Reward denso sulla guardia con budget**: bonus a ogni calo della guardia dell'avversario, sottratto dal premio finale (già nella libreria).
3. **Shaping basato su potenziale**: da scrivere come wrapper; in teoria non cambia la policy ottima.
4. **Vittoria scalata sul tempo**: vincere prima vale di più, resta a somma zero (wrapper).

Il pool di checkpoint (PFSP) e le altre ablation sono estensioni se resta budget (vedi Esperimenti e ablation).

## Perché Footsies e formulazione come POMDP

Footsies è un gioco a due giocatori, a somma zero e a informazione imperfetta, quindi si presta a essere formalizzato come POMDP e come problema multi-agente competitivo (cap. 2.4 e 11 del libro).

- **Stato**: configurazione completa del gioco per entrambi i giocatori, incluse le feature che un giocatore non vede dell'altro.
- **Osservazione**: vettore di 85 dimensioni (88 con le azioni di carica) con la parte comune, le feature di sé, quelle dell'avversario e un blocco "privilegiato" solo per sé (dash disponibile, progresso della carica, azione precedente).
- **Azioni**: 6 discrete di default (NONE, BACK, FORWARD, ATTACK, BACK+ATTACK, FORWARD+ATTACK), 9 con le azioni di carica per le mosse speciali.
- **Reward**: a somma zero, ±10 alla vittoria o sconfitta, 0 al timeout (4000 step); opzionale reward denso per i guard break (vedi la sezione sulla funzione di reward).
- **Parziale osservabilità**: l'agente non vede le intenzioni dell'avversario, che deve prevedere. Con `action_delay` alto il gioco diventa quasi pura previsione.

Come nel progetto Briscola preso a modello, la scelta di formulazione va scritta esplicitamente nella relazione e nelle slide: stato, osservazione, azioni, transizioni, reward, e la catena s→o→a.

## Dinamica, inizializzazione e terminazione

Il simulatore Unity applica le regole del gioco; per quanto risulta dal paper non ci sono elementi casuali nelle transizioni (da verificare), quindi la casualità di un episodio viene dalle policy dei due giocatori. Ogni passo dell'ambiente sono 4 frame di gioco (`frame_skip`) e l'azione scelta viene eseguita 8 frame dopo (`action_delay`): l'ambiente tiene una coda di azioni in attesa e inserisce la propria azione precedente nell'osservazione privilegiata.

**Inizializzazione.** A ogni `reset()` l'ambiente chiama `ResetGame` sul server Unity; lato Python non c'è nessuna casualità e l'argomento `seed` viene ignorato. Si azzerano la coda delle azioni (riempita di NONE), la carica speciale, il budget del reward (riportato a w) e lo stato dell'encoder. La posizione iniziale dei giocatori è decisa dal server e il paper non la descrive: va verificata nella repo del gioco, ma probabilmente è sempre la stessa. Conseguenza: ogni partita parte dalla stessa configurazione, quindi la varietà dei dati dipende solo dalla stocasticità delle policy (per questo l'entropia conta) e la valutazione alterna i lati P1 e P2.

**Terminazione.** Un episodio termina quando un giocatore è sconfitto (`is_dead`; secondo il paper, quando è colpito da un attacco speciale) oppure è troncato a 4000 passi, circa 267 secondi di gioco a 60 Hz: in quel caso entrambi ricevono 0. Le due cose vanno trattate in modo diverso nel PPO: alla sconfitta il valore dello stato successivo è zero, alla troncatura si fa bootstrap con il valore stimato dal critico (l'ambiente vettorizzato sostituisce l'ultima osservazione con quella di reset, quindi come approssimazione si usa V dell'osservazione precedente), altrimenti si introduce un bias perché lo stato non è davvero terminale (vedi Bias e varianza).

## Funzione di reward: perché, limiti e conseguenze

**Cosa è definito.** Il reward è a somma zero: +w al vincitore e -w allo sconfitto (w = `win_reward_scaling_coeff` = 10), 0 a entrambi al timeout. Nel paper il default è w = 10, ma nella libreria installata `win_reward_scaling_coeff` vale 1,0: il nostro codice lo imposta a 10 in `src/env.py`. La valutazione usa sempre il solo esito terminale (vittoria, pareggio, sconfitta), qualunque sia la reward di training. Le varianti sono in `src/rewards.py` e si scelgono con `--reward`.

**Perché questa scelta.** Il premio terminale coincide con l'obiettivo vero (vincere il round) e non cambia il problema: la forma del reward non introduce bias. La somma zero rende il gioco puramente competitivo (cap. 11) e il self-play ben definito: ciò che guadagna un agente lo perde l'altro. Il timeout a 0 tratta il pareggio come neutro. La reward terminale è il default dell'ambiente e quello del paper, quindi è la baseline del confronto.

**Conseguenze sull'apprendimento.** Il segnale è sparso e ritardato: arriva solo a fine episodio, fino a 4000 passi dopo; con gamma 0,995 un'azione compiuta più di circa 920 passi prima della fine pesa meno dell'1%, quindi il credit assignment è difficile e la stima del ritorno ha varianza alta. Con il timeout a 0, contro un avversario passivo (il No-op) ogni partita non vinta è un pareggio senza segnale, e il paper osserva che le policy migliori diventano passive o reattive. La scala conta: il coefficiente di entropia (0,025) e quello della value loss (1,0) sono tarati su w = 10; con w = 1 il bonus di entropia peserebbe relativamente 10 volte di più. Un reward denso riduce la varianza ma può cambiare l'obiettivo: è il compromesso bias/varianza che il confronto vuole misurare.

**Varianti confrontate.** Tutte restano a somma zero e hanno lo stesso budget (20M passi, 5 seed, self-play).

| Sigla | Reward di training | Effetto atteso | Rischio di bias | Run |
|---|---|---|---|---|
| R0 | solo terminale ±10, timeout 0 (come il paper) | varianza alta, nessun bias | nessuno | `r0_s1`–`r0_s5` (entropia fissa 0,025); `r0_sched_s1`–`r0_sched_s5` (entropia 0,1→0) |
| R1 | guard break: ±2 per ogni punto di guardia (3 punti, 6 in totale) sottratto dal premio finale (`use_reward_budget`), quindi una vittoria vale in tutto 10 | segnale più frequente, meno varianza | l'agente può imparare a far calare la guardia (farsi bloccare gli attacchi) invece di vincere | `r1_s1`–`r1_s5`; `r1_floor_s1`–`r1_floor_s5` (vedi sotto) |
| R2 | shaping basato su potenziale F = gamma·Phi(s') - Phi(s), con Phi = 6·(guardia propria - guardia avversaria)/3 | in teoria stessa policy ottima del reward terminale (Ng et al., 1999; non è nel libro) | nessuno in teoria; in pratica la troncatura (Phi(s') approssimato con Phi(s)) e la guardia che si rigenera possono romperlo | `r2_s1`–`r2_s5` |
| R3 | vittoria/sconfitta ±10 moltiplicata per un fattore che scende da 1 a 0,5 con la durata della partita | vincere prima è meglio | preferenza per partite rapide e rischiose | `r3_s1`–`r3_s5` |

La guardia vale 0-3 punti per giocatore e, nel codice, il calo della guardia dell'avversario coincide con un attacco bloccato.

**Correzione degli stalli (`--adv-std-floor`, run `r0_floor` e `r1_floor`).** Non è una reward nuova: è R0 o R1 allenata con `--adv-std-floor 0,1`. Quando i due agenti diventano passivi il reward è quasi sempre 0 e i vantaggi hanno deviazione standard ~1e-3; la normalizzazione per minibatch divide per quella std e amplifica il rumore fino a scala 1, mentre il bonus di entropia spinge la policy verso la distribuzione uniforme e il learning rate che scende a 0 impedisce di uscire dall'ultimo stallo. La correzione divide per max(std, floor); con floor = 0 (default) il comportamento è identico all'originale, quindi i baseline non cambiano. È stata introdotta dopo aver osservato che le varianti di reward non si distinguevano e che molte run finivano in stallo (vedi Risultati), con 5 seed per R0 e per R1. Limite: la correzione toglie l'amplificazione del rumore ma non la passività verso gli avversari fissi; l'ipotesi che l'inattività reciproca sia un punto fisso del self-play simmetrico non è verificata.

**Risultati.** Sono nella sezione dedicata più avanti. In sintesi: nessuna reward batte chiaramente le altre; R3 è la peggiore; la dispersione tra seed supera le differenze tra varianti; il floor non cambia molto la media ma riduce gli stalli e la dispersione.

**Idea esclusa (non eseguita): penalità sul timeout (R4).** Scoraggia la passività ma, se uguale per entrambi i giocatori, rompe la somma zero e fa perdere il significato esatto di best response ed exploitability. Si cita solo come limite e possibile sviluppo.

## Ambiente e setup tecnico

L'ambiente è già installato nella repo e funziona con agenti casuali (`test.py`, `visual_test.py`); il PPO in PyTorch, il training, le varianti di reward e la valutazione sono scritti e funzionanti (`src/`); i run e le analisi sono completati (vedi Risultati).

- **Ambiente**: pacchetto `footsies-gym` (API PettingZoo), simulatore Unity headless via gRPC, vettorizzabile con `num_envs`. Il paper riporta circa 19.000 step/s con un solo processo e circa 52.500 con 4 processi e 128 ambienti su una workstation da 24 core.
- **Configurazione di base**: `frame_skip` 4, `action_delay` 8, `max_t` 4000, reward di vittoria 10 (valore del paper; il default della libreria installata è 1,0, quindi va impostato \`win\_reward\_scaling\_coeff\` a 10). Le variazioni (delay, carica speciale, reward denso) sono leve per gli esperimenti.
- **Dove si lavora**: sviluppo e test brevi sul Mac (`platform="mac"`), run lunghe sul cluster Orfeo (SLURM, partizione THIN, senza GPU perché la rete è piccola) (`platform="linux"`). Windows non è supportato.
- **Librerie**: PyTorch. PPO, wrapper di reward, valutazione e avversari sono scritti da noi.
- **Versione di Python**: fissata a 3.12 con `.python-version`, la stessa su Mac e su Orfeo. PyTorch e la libreria dell'ambiente funzionano con questa versione (verificato su entrambi).

## Avversari fissi (scala di difficoltà)

Servono quattro avversari, dal più debole al più forte, per dare un'unità di misura comune a tutte le valutazioni; il progetto Briscola ne usava cinque. Tre sono banali da scrivere (Random, No-op e Back sono un campionamento uniforme e due azioni costanti) e uno è già nella libreria: il bot a regole FootsiesBot.

| Avversario | Logica | Cosa misura |
| --- | --- | --- |
| Random | Azione uniforme tra le 6 disponibili (da scrivere) | Soglia minima di competenza |
| No-op | Non fa mai nulla (azione costante) | Se l'agente sa iniziare lo scontro (nel paper i policy migliori diventano passivi) |
| Back | Indietreggia sempre (azione costante) | Se l'agente sa inseguire e colpire un avversario che fugge |
| FootsiesBot | IA a regole del gioco originale (BattleAI), riscritta nella libreria | Livello contro un benchmark usato anche dal paper |

FootsiesBot (footsiesgym/footsies/game/footsies\_bot.py) usa lo stato della partita e un po' di casualità, quindi è il più vicino a come gioca un umano. Non è una rete: prende lo stato della partita, quindi serve un adattatore, già scritto in src/opponents.py (FootsiesBotOpponent). Funziona solo con return\_fight\_state\_in\_infos=True e num\_envs=1, perché con gli ambienti vettorizzati le info sono vuote: si usa quindi nella valutazione, un ambiente alla volta, e non nel training vettorizzato. Copre solo le 6 azioni base.

## Algoritmo: PPO scritto da noi in self-play

Gli agenti condividono osservazione, azioni, rete e iperparametri di PPO e differiscono per la funzione di reward; l'avversario è sempre la stessa rete in self-play.

**Implementazione: PPO scritto da noi (PyTorch).** Rete MLP 2×256 con ReLU e due teste (policy e valore), raccolta su ambienti vettorizzati della libreria (`num_envs`), vantaggio con GAE, loss con clipping più termine di valore e bonus di entropia, 8 epoche su minibatch. Nel self-play semplice la stessa rete gioca entrambi i lati e usa i dati di entrambi i giocatori; nel pool si usano solo i dati del lato che sta imparando. Il nostro codice si valida con le curve contro il Random e con le metriche di diagnostica (loss, KL, frazione di clipping, varianza esplicata). Si parte dall'esempio CleanRL ufficiale di FootsiesGym (adattato da ppo\_pettingzoo\_ma\_atari): ha già l'ambiente vettorizzato con slot p1 e p2 che condividono la rete, GAE, loss e le metriche di diagnostica (varianza esplicata, KL, frazione di clipping). Si cambiano iperparametri (learning rate 1e-4, rollout 64, entropia come da paper), trattamento della troncatura, reward e valutazione. Collegamento al libro: vantaggio e actor-critic (cap. 9.3.2-9.3.3); GAE e clipping non sono nel libro.

**1. PPO con self-play semplice.** Iperparametri di partenza dal paper: rete MLP 2×256 con ReLU, gamma 0,995, GAE lambda 0,95, clip 0,3, 8 epoche con 8 minibatch, rollout di 64 passi, 48 ambienti in parallelo, learning rate 1e-4 che decresce a 0, ottimizzatore Adam, coefficiente della value loss 1,0, norma massima del gradiente 0,5 (Tabella 3 del paper). Le due baseline del paper si addestrano entrambe con il reward terminale: PPO con coefficiente di entropia fisso (0,025) e PPO (Sched.) con entropia che scende linearmente da 0,1 a 0. Le varianti di reward si confrontano con entropia fissa (0,025), così la sola variabile è il reward.

**2. Estensione facoltativa: PPO con pool di checkpoint.** Ogni partita è giocata contro la policy corrente con probabilità 0,8 e contro uno snapshot del pool altrimenti. Gli snapshot (fino a 16, uno ogni circa 614.000 step) sono scelti con peso proporzionale a (1 - p\_i)^(1/2), dove p\_i è la frequenza di vittoria stimata contro lo snapshot i: si gioca di più contro chi ci batte. Con il PPO scritto da noi la scelta dell'avversario è un campionamento fatto a ogni reset dell'ambiente, senza toccare learner o moduli. Un curriculum che aggiunge avversari quando il win rate supera una soglia non è un PFSP, quindi non è la scelta fatta qui.

**Scala degli esperimenti.** Il paper usa 25 milioni di step per 5 seed. Per il progetto si usano 20 milioni di step per run, perché un job su Orfeo ha un limite di 2 ore (vedi Budget di calcolo), e 5 seed (1-5) per ciascuna delle cinque varianti del confronto (R0, R1, R2, R3 e PPO Sched.), con gli stessi seed per tutte, così il confronto è appaiato. Ablation, studio su lambda e gamma e `r1_floor` restano a 1 seed. Se serve semplificare, si riducono prima ablation e best response.

## Bias e varianza nell'algoritmo

In RL bias e varianza hanno un significato diverso dall'apprendimento supervisionato: non riguardano la complessità del modello, ma la qualità dello stimatore del target che guida l'aggiornamento (il ritorno, il vantaggio, il gradiente della policy), calcolato da poche traiettorie campionate. Si può usare il reward realmente osservato (in media corretto, ma molto rumoroso: poco bias, molta varianza) oppure una stima appresa del valore futuro (stabile, ma sbagliata quanto sbaglia il critico: poca varianza, ma bias). È lo spettro tra Monte Carlo e TD del cap. 6 (n-step); in questo progetto il compromesso si regola con lambda, gamma e il clipping.

```latex
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

| Scelta | Effetto | Bias | Varianza | Libro |
| --- | --- | --- | --- | --- |
| Ritorno Monte Carlo (REINFORCE) | usa tutto il ritorno G\_t | nessuno | alta: dipende da tutti i reward fino a 4000 passi | 9.3 |
| Baseline V(s) | sottrae una stima del valore dello stato | nessuno (aspettativa nulla) | più bassa | 9.3.1 |
| Bootstrapping con il critico | sostituisce il futuro con V(s') | sì, pari all'errore del critico | più bassa | 8.3, 9.3.3 |
| GAE con lambda 0,95 | media pesata di TD a più passi | intermedio: lambda 0 è il TD a un passo, lambda 1 è Monte Carlo | intermedia | 6 (spettro n-step); GAE non è nel libro |
| Sconto gamma 0,995 | orizzonte efficace di circa 200 passi | sì: si ottimizza il ritorno scontato, non il vincere | più bassa | 2.5-2.7 |
| Clipping e 8 epoche (epsilon 0,3) | riusa gli stessi dati più volte | sì: si ottimizza una surrogata, non il gradiente esatto | più bassa per campione; il rapporto di importanza è la fonte di varianza off-policy | 5 (off-policy); PPO non è nel libro |
| Bonus di entropia | spinge verso policy più casuali | sì, voluto; sparisce alla fine nella variante decrescente (0,1 a 0) | non la riduce direttamente |  |
| Troncatura a 4000 passi | se trattata come fine vera | sì; si evita con il bootstrap su V |  |  |

**Come si legge per PPO.** PPO parte dall'actor-critic (cap. 9.3.3), che abbassa la varianza di REINFORCE a prezzo del bias del critico, e con GAE regola il compromesso con un solo parametro. Il problema è tipico di questo gioco: con reward solo finale ed episodi lunghi, un Monte Carlo puro avrebbe varianza altissima (per questo REINFORCE è stato scartato), mentre un TD a un passo propaga il reward di una vittoria un passo alla volta e si affida a un critico che deve stimare un valore molto lontano.

**Perché non un metodo basato sul massimo** (Q-learning, DQN): PPO non soffre della maximization bias del cap. 5.7, che nasce dal prendere il massimo di stime rumorose, e produce una policy stocastica, necessaria in un gioco a strategie miste.

**Come si misura.** Con due ablation (lambda 0 e 1 contro 0,95; gamma 0,99 contro 0,995) e con la diagnostica durante il training: varianza esplicata del critico (quanto del ritorno spiega V, indice del suo bias), deviazione standard del vantaggio per batch (indice di varianza), KL e frazione di campioni tagliati dal clipping. Anche le varianti di reward si leggono così: il reward denso e lo shaping danno un segnale più frequente (meno varianza del ritorno) a costo di un possibile bias nell'obiettivo. Le ipotesi da verificare, non risultati: lambda 1 più lento e rumoroso (varianza), lambda 0 più stabile ma più facilmente bloccato su policy peggiori (bias). Lo stesso vocabolario vale per la valutazione: la best response approssimata dà un bias per difetto sulla sfruttabilità, e un solo seed non permette di stimare la varianza tra addestramenti.

## Valutazione e metriche

Ogni agente è valutato su tre livelli (contro avversari fissi, in scontro diretto e contro una best response), perché il solo win rate contro euristiche nasconde la sfruttabilità.

| Livello | Come | Cosa dice |
| --- | --- | --- |
| Avversari fissi | 100 partite per avversario, lati P1 e P2 alternati, win/draw/loss con intervallo di Wilson al 95% | Livello di abilità sulla scala di difficoltà |
| Scontro diretto | Matrice tra i 26 modelli finali (5 seed per variante), 100 partite per coppia con i lati alternati, punteggio = (vittorie + 0,5 pareggi) / partite | Quale variante batte quale |
| Exploitability approssimata | Si congela l'agente finale e si addestra contro di esso una best response con PPO (2 seed di BR per target, sul seed 1 di ogni variante) | Limite inferiore di quanto l'agente è sfruttabile |
| Curve di apprendimento | Win rate contro gli avversari fissi ogni N step | Velocità e stabilità dell'addestramento |

Quattro scelte di metodo da dichiarare:

- **Evaluation stocastica, non greedy.** In un gioco con strategie miste una policy deterministica è sfruttabile per costruzione, quindi si valuta campionando dalla distribuzione. Il progetto Briscola usava la scelta greedy; qui non è adatto.
- **5 seed per il confronto, 1 per il resto.** Come il paper, le varianti del confronto (R0, R1, R2, R3, PPO Sched.) e le due varianti con floor (R0 floor, R1 floor) hanno 5 seed ciascuna. Gli intervalli di Wilson sulle partite misurano il rumore della valutazione; la variabilità tra addestramenti si legge dalla dispersione tra i 5 seed (media e min-max per variante, e deviazione standard). Per il confronto floor contro base si usa un bootstrap sui seed (intervallo al 95% sulla differenza delle medie): con 5 seed per gruppo è ampio, la differenza minima rilevabile è circa 0,3 sul win rate, quindi l'assenza di differenze non è equivalenza. Le ablation (λ, γ, charge, delay) hanno 1 seed e sono descrittive; la best response ha 2 seed di BR per target ma un solo seed del target.
- **Best response come limite inferiore, rumoroso.** La BR è approssimata, quindi stima per difetto la sfruttabilità; i risultati mostrano che per R0, Sched e R3 dipende molto dal seed della BR (un seed sfrutta il target, l'altro resta nei pareggi), perciò si riporta il massimo tra i due seed accanto alla dispersione, e si leggono sempre anche i pareggi.
- **Metrica comune.** Il ritorno non si può confrontare tra reward diverse: tutte le varianti si giudicano con win rate contro gli avversari fissi, scontro diretto e best response, misurati con il reward terminale (anche la best response si addestra con quello).

## Confronto con il paper

Il paper usa PPO in self-play con gli stessi iperparametri del nostro progetto, ma confronta più algoritmi e valuta in più modi; noi riproduciamo una parte e cambiamo il reward.

| Aspetto | Paper | Questo progetto |
| --- | --- | --- |
| Algoritmi | PPO (entropia fissa 0,025), PPO (Sched.) (entropia da 0,1 a 0), EMAgnet, PFSP | PPO scritto da noi, con entropia fissa e Sched. come nel paper (reward terminale); PFSP come estensione facoltativa; niente EMAgnet |
| Step e seed | 25 milioni di step, 5 seed | 20 milioni di step (limite di 2 ore per job su Orfeo), 5 seed per le varianti del confronto e per le varianti floor, 1 per il resto |
| Reward | terminale ±10, timeout 0 | R0 identico al paper; R1 guard break, R2 shaping a potenziale, R3 vittoria scalata nel tempo |
| Avversari fissi | random e no-op | random, no-op, back e FootsiesBot |
| Scontro diretto | tutte le coppie di algoritmi, 96 partite per coppia di seed | final di ogni run (26 modelli, 325 coppie), 100 partite per coppia; matrice con più checkpoint non fatta |
| Sfruttabilità | best response addestrata con PPO contro il checkpoint finale (3 seed per ogni seed) | fatta: 2 seed di BR per target (5 target, seed 1), 500 partite, 20M step per BR |

**Reattività.** Il paper osserva che le policy migliori diventano reattive e iniziano meno lo scontro contro un avversario fermo: tutte le partite non vinte contro il no-op sono timeout. Lo vediamo anche noi: i modelli finali di quasi tutte le varianti fanno 0 vittorie e 100 pareggi contro no-op e back, ma vincono il 70-90% contro il random e il 55-85% contro FootsiesBot, che attacca. Un dato in più: contro il no-op molti agenti vincono il 100% dei match a 5M passi (R1 fino a 15M) e poi lo perdono (a 20M fanno 100% pareggi); nel self-play la policy si specializza sull'avversario che incontra e dimentica come sfruttarne uno passivo. L'unico finale che batte ancora il no-op è l'agente con azioni di carica (86% in valutazione).

## Esperimenti e ablation

Il nucleo sono tre confronti: le due baseline del paper, le funzioni di reward e il compromesso bias-varianza; le ablation sull'ambiente e le estensioni si aggiungono in ordine di priorità se resta tempo di calcolo.

| Priorità | Esperimento | Cosa varia | Domanda |
| --- | --- | --- | --- |
| Core | Baseline del paper | PPO con entropia fissa (0,025) contro PPO (Sched.) (0,1 a 0), entrambe con reward terminale | La schedule di entropia cambia lo stallo e la robustezza, come nel paper? |
| Core | Reward | terminale; denso sulla guardia con budget; shaping a potenziale; vittoria scalata sul tempo | Quale reward produce l'agente più forte e meno sfruttabile? |
| Core | Correzione degli stalli | R0 e R1 con `--adv-std-floor 0,1`, 5 seed ciascuno | Il floor sui vantaggi riduce gli stalli e la dispersione tra seed? |
| Core | Bias e varianza | su R0, 1 seed: GAE lambda 0 e 1 contro 0,95; gamma 0,99 contro 0,995 | Come cambiano velocità e stabilità con più bias (TD) o più varianza (Monte Carlo)? |
| Secondaria | Azione di carica | su R0, 1 seed: `use_special_charge_action` on/off (9 azioni, osservazione da 88) | L'agente scopre e usa le mosse speciali? |
| Secondaria | Ritardo d'azione | su R0, 1 seed: `action_delay` 0, 8, 12 | Con delay 0 il gioco diventa reattivo e fragile, come nel paper? |
| Facoltativa | Pool di checkpoint | self-play contro snapshot passati con campionamento prioritario | Il pool riduce la sfruttabilità? |
| Facoltativa | Penalità sul timeout | penalità simmetrica, fuori dalla somma zero | Scoraggia la passività, e cosa perde la valutazione? |

Le varianti del confronto usano 5 seed, le altre run un seed solo e sono descrittive. Tutte le run usano la stessa rete di base, così che le differenze si attribuiscano alla variabile cambiata.

## Risultati

Tutte le run sono da 20M passi, in self-play, con valutazione stocastica e lati alternati. Win rate finale contro avversari fissi (100 partite per modello): media sui 5 seed e, tra parentesi, minimo e massimo. «Stallo finale»: ultimo 10% dei training con meno dell'1% di partite decise (misurabile solo sulle run con la colonna `decided_rate`, quindi non su `r0_s1`, `r1_s1` e `r2_s1`).

| Variante | vs Random | vs FootsiesBot | Stallo finale |
| --- | --- | --- | --- |
| R0 (terminale) | 0,74 (0,41–0,88) | 0,68 (0,53–0,90) | 1 run su 4 |
| R0 Sched (entropia 0,1→0) | 0,82 (0,74–0,94) | 0,65 (0,13–0,87) | 3 su 5 |
| R1 (guard break) | 0,90 (0,84–0,94) | 0,74 (0,41–0,85) | 1 su 4 |
| R2 (shaping a potenziale) | 0,75 (0,18–0,90) | 0,63 (0,05–0,86) | 0 su 4 |
| R3 (vittoria scalata) | 0,52 (0,05–0,77) | 0,35 (0,18–0,65) | 5 su 5 |
| R0 floor | 0,88 (0,82–0,92) | 0,75 (0,61–0,84) | 0 su 5 |
| R1 floor | 0,91 (0,87–0,97) | 0,77 (0,72–0,84) | 0 su 4 |

**Confronto tra reward.** Nessuna variante domina. Le differenze tra le medie sono più piccole della dispersione tra seed della stessa variante (deviazione standard 0,15–0,33 contro il bot), e con 5 seed per gruppo l'intervallo sulla differenza di due medie è circa ±0,3: possiamo dire che R3 è la più debole e che R1 è la più stabile contro il random, non ordinare R0, R1, R2 e Sched. Nello scontro diretto tra i 26 modelli finali (325 coppie, 100 partite per coppia) il punteggio medio per variante è R0 0,53, Sched 0,51, R2 0,51, R1 0,49, R3 0,45; dentro una variante i seed variano molto di più (Sched da 0,36 a 0,65). La reward conta meno della varianza dell'addestramento.

**Stalli.** Molte run finiscono in stallo totale, cioè tutte le partite sono timeout anche in self-play: 5 su 5 per R3, 3 su 5 per Sched. Non è passività in senso stretto: nelle diagnostiche gli agenti in stallo attaccano nel 43–54% dei passi, più dei modelli senza stallo (14–38%), senza arrivare a colpire; il termine corretto è «timeout con attacchi che non concludono». Tre fatti li legano all'ottimizzazione: nello stallo il reward è quasi nullo, la std dei vantaggi cade a ~1e-3 e la normalizzazione amplifica il rumore; il bonus di entropia spinge verso la policy uniforme; il learning rate a 0 impedisce di uscire. Sono ipotesi coerenti con i dati, non dimostrate.

**Effetto del floor sui vantaggi.** Su 5 seed per gruppo, differenza floor meno base con bootstrap sui seed (intervallo al 95%) e rapporto tra le deviazioni standard:

| Confronto | Base (media, dev. std.) | Floor (media, dev. std.) | Differenza | Rapporto dev. std. |
| --- | --- | --- | --- | --- |
| R0, vs Random | 0,74, 0,19 | 0,88, 0,04 | +0,14 [+0,02, +0,31] | 0,20 |
| R0, vs bot | 0,68, 0,15 | 0,75, 0,09 | +0,07 [−0,08, +0,20] | 0,60 |
| R1, vs Random | 0,90, 0,04 | 0,91, 0,04 | +0,02 [−0,03, +0,06] | 1,10 |
| R1, vs bot | 0,74, 0,19 | 0,77, 0,04 | +0,04 [−0,08, +0,21] | 0,23 |

Sulla media l'effetto è piccolo e quasi sempre non distinguibile da zero; sulla dispersione è coerente in 3 casi su 4 (la deviazione standard scende di un fattore 2–5; con 5 seed è una stima debole). I training che finiscono in stallo sono 2 su 8 senza floor e 0 su 9 con floor. Lettura: il floor non migliora il valore atteso, riduce la varianza dell'addestramento, ed è coerente con il tema bias/varianza.

**Bias e varianza (ablation su R0, 1 seed).** Con λ = 0 (TD a un passo, bias massimo) l'agente non impara (0,05 contro Random, 0,04 contro il bot); con λ = 1 (Monte Carlo, varianza massima) impara (0,81 e 0,83) e con γ = 0,99 pure (0,69 e 0,85), dentro la dispersione tra seed di R0 (0,41–0,88 e 0,53–0,90). Quindi il risultato netto è solo che troppo bias blocca l'apprendimento; λ = 1 e γ = 0,99 non sono distinguibili da λ = 0,95 con un seed. Il motivo di λ = 0 va letto come ipotesi: con un reward solo terminale il TD a un passo deve propagare il valore di una vittoria un passo alla volta attraverso un critico ancora sbagliato.

**Opzioni di gioco (R0, 1 seed).** Con le azioni di carica l'agente vince 0,82 contro il Random e 0,76 contro il bot e batte ancora il no-op (0,86), ma usa le mosse speciali solo nell'1–2% dei passi: il risultato non è spiegato da un uso strategico delle speciali, e va attribuito anche al diverso spazio di azioni e osservazione. Con `action_delay` 0 si ottiene 0,64 contro il Random e 0,55 contro il bot, con 12 si ottiene 0,70 e 0,39 e una vittoria sul no-op e sul back (1,00 e 0,50), mentre l'agente con delay 0 pareggia sempre contro entrambi. Sono osservazioni di un seed, nello stesso senso di quanto riportato dal paper sul ritardo d'azione.

**Best response.** PPO di 20M passi contro il checkpoint finale congelato, 500 partite, 2 seed di BR per target (vittorie della BR, pareggi, sconfitte):

| Target | BR seed 1 | BR seed 2 | Massimo |
| --- | --- | --- | --- |
| R0 | 241 / 222 / 37 | 0 / 500 / 0 | 48% |
| R0 Sched | 1 / 497 / 2 | 249 / 247 / 4 | 50% |
| R1 | 0 / 500 / 0 | 0 / 500 / 0 | 0% |
| R2 | 499 / 0 / 1 | 500 / 0 / 0 | 100% |
| R3 | 494 / 0 / 6 | 157 / 224 / 119 | 99% |

R2 e R3 sono nettamente sfruttabili (contro un avversario che si adatta crollano, anche se nello scontro diretto sembravano alla pari delle altre); R0 e Sched arrivano al 50%. Per R0, Sched e R3 il risultato dipende dal seed della BR: la BR soffre dello stesso problema di stallo degli agenti principali. Per R1 nessuna BR vince in 1000 partite, ma i pareggi possono essere un artefatto, quindi si dice «nessuna BR lo ha sfruttato», non «R1 è robusto». Un test non fatto: rifare le BR con il floor.

**Cosa si può sostenere e cosa no.** Si può sostenere: nessuna reward batte chiaramente le altre; R3 è la peggiore; la varianza tra seed domina; gli stalli sono frequenti e il floor li elimina nei seed osservati; R2 e R3 sono molto sfruttabili; λ = 0 non apprende. Non si può sostenere: che R1 sia più robusto; che il floor migliori la forza media; che il charge sia usato in modo strategico; che le varianti siano equivalenti. Limiti: diagnostiche e ablation su un seed; la distanza tra i giocatori non è stata misurata (le feature di posizione 2 e 43 non risultano informative nel nostro estratto).

## Budget di calcolo

Consuntivo delle run eseguite su Orfeo. Ogni run è di 20 milioni di step, scelti in base alla velocità misurata e al limite di 2 ore per job. Unità: S = una run PPO da 20 milioni di step.

| Gruppo di run | N. run | Costo |
| --- | --- | --- |
| Confronto reward: R0, R1, R2, R3 e Sched, 5 seed ciascuna | 25 | 25 S |
| Correzione degli stalli: R0 floor e R1 floor, 5 seed ciascuna | 10 | 10 S |
| Ablation (λ 0 e 1, γ 0,99, charge, delay 0 e 12), 1 seed | 6 | 6 S |
| Best response: 5 target, 2 seed di BR | 10 | 10 S |
| Totale | 51 | 51 S (1,02 miliardi di step) |

Le valutazioni (contro avversari fissi, scontro diretto, diagnostiche) costano pochi minuti per modello. Su Orfeo si possono avere al più 20 job in coda e 5 in esecuzione, quindi le run sono state lanciate in array con al più 5 task contemporanei. Una BR è andata due volte in errore per un'eccezione di I/O sul filesystem condiviso (log non scritto); il codice ora ignora l'errore di scrittura.

Le velocità del paper (circa 19.000 step/s con un processo, circa 52.500 con 4 processi) misurano solo il simulatore, non l'addestramento della rete. Misurate sul nostro PPO completo (48 partite, 3.072 step per update): circa 6.700 step/s sul Mac e circa 4.050 su un nodo THIN di Orfeo con 4 thread. Quattro job contemporanei su THIN fanno 3.800-4.500 step/s ciascuno, senza rallentamenti. Il tempo si divide tra aggiornamento della rete (56% sul Mac, 63% su Orfeo) e simulatore (37% e 25%). Una run da 20 milioni di step dura quindi circa 50 minuti sul Mac e 1,4 ore su THIN (1,8 su GENOA); sulle run reali su THIN abbiamo misurato 85-100 minuti. GENOA fa circa 3.100 step/s, stabili su 3 job, ed è la seconda scelta; EPYC è la più lenta (circa 750 step/s, misurata con una versione precedente del codice) e non si usa. Il piano gira su Orfeo con un job per run, in parallelo: il tempo reale è circa quello di una run più la coda, non la somma. Se non ci sta, si taglia in quest'ordine: ablation facoltative, best response a 1 seed, step per run.

## Diagnostica e analisi del comportamento

Le percentuali di vittoria non bastano a capire cosa ha imparato l'agente, quindi si registrano e si analizzano le partite, come nella parte di diagnostica del progetto Briscola.

- **Fatto (`src/diagnostics.py`, seed 1 di 9 modelli finali):** distribuzione delle azioni (quota di attacchi e di mosse speciali), esito e durata delle partite, tempo e autore del primo attacco, in self-play e contro no-op, back e random. Serve a distinguere lo stallo (timeout con attacchi che non concludono) dalla reattività e a controllare se le mosse speciali sono usate (1–2% con il charge).
- **Non fatto:** log passo per passo con probabilità delle azioni, video delle partite con la build grafica (`visual_test.py`), distanza tra i giocatori (le feature di posizione non risultano informative).
- **Diagnostica di bias e varianza** nel training: varianza esplicata del critico, deviazione standard del vantaggio per batch (che mostra lo stallo con valori attorno a 1e-3), KL e frazione di campioni tagliati dal clipping, registrate a ogni aggiornamento.

Questa parte è quella che più aiuta all'orale: permette di dire cosa fa l'agente e perché, non solo quanto vince.

## Collegamento con il libro del corso

Ogni scelta del progetto si appoggia a un capitolo delle dispense, da citare esplicitamente in relazione e all'orale.

| Elemento del progetto | Capitolo del libro |
| --- | --- |
| Stato, osservazione, belief, parziale osservabilità | 2.4 POMDP |
| Reward terminale, ritorno, fattore di sconto | 2.5-2.7 Goals, returns, episodi |
| Politiche e funzioni di valore | 2.8-2.9 |
| Approssimazione con rete neurale | 8 Approximate solution methods |
| Baseline e riduzione della varianza (base teorica di PPO) | 9.3 e 9.3.1 REINFORCE e baseline |
| Vantaggio e actor-critic (alla base di PPO) | 9.3.2-9.3.3 |
| Teorema del policy gradient | 9.2 e 10.1 |
| Ottimalità globale del policy gradient | 10.2 |
| Self-play, non stazionarietà, best response, exploitability | 11 Multi-agent RL e teoria dei giochi |
| Bias e varianza: Monte Carlo, TD, n-step, GAE, bootstrapping | 6 n-step bootstrapping; 8.3 semi-gradient; 9.3, 9.3.1, 9.3.3 |
| Perché non Q-learning o DQN (maximization bias) | 5.7 Maximization bias e double learning |

PFSP e la best response non sono nel libro (il cap. 11 ne dà solo l'idea generale). PPO è scritto da noi come evoluzione dell'actor-critic (cap. 9.3.2-9.3.3): aggiornamenti limitati dal clipping, motivati dall'alta varianza del policy gradient puro (REINFORCE, cap. 9.3). GAE e clipping non sono nel libro e vanno spiegati a parte; PFSP e la best response sono applicazioni delle idee del cap. 11.

## Struttura della repo e consegne

Le consegne sono il codice con un README riproducibile e la presentazione per l'orale, entrambe costruite sulla repo già esistente.

```text
FootsiesRL/
  README.md            come installare, lanciare train e valutazione
  pyproject.toml, uv.lock   dipendenze gestite con uv (uv sync, uv run)
  slurm/               script sbatch per Orfeo (array di run, valutazioni, best response, diagnostiche)
  src/
    ppo.py             PPO scritto da noi (rete, GAE, loss, aggiornamento)
    train.py           ciclo di addestramento, self-play, pool e checkpoint
    rewards.py         varianti di reward (wrapper: potenziale, vittoria scalata sul tempo)
    policy.py          checkpoint come avversario (PolicyOpponent)
    head_to_head.py    matrice degli scontri diretti, a shard
    plot_results.py    curve di training e di valutazione aggregate sui seed
    opponents.py       Random, No-op, Back e adattatore per FootsiesBot
    evaluate.py        partite, intervalli di confidenza, matrice head-to-head
    best_response.py   addestramento della best response
    diagnostics.py     statistiche sulle azioni, durata e primo attacco
  results/             tabelle e figure
  test.py  visual_test.py   già presenti
```

La presentazione segue la struttura che ha funzionato nel progetto Briscola:

1. Introduzione e motivazione (perché Footsies, perché è un POMDP).
2. Formulazione: stato, osservazione, azioni, inizializzazione, reward (scelte, limiti, miglioramenti).
3. Avversari fissi e agenti appresi.
4. Addestramento: PPO scritto da noi in self-play, bias e varianza dell'algoritmo.
5. Confronto tra le reward, valutazione e diagnostica.
6. Risultati e limitazioni.

## Calendario, rischi e limiti

Il lavoro procede in sei fasi. Scadenza: 21 novembre 2026. Stato al 14 novembre: gli esperimenti sono conclusi (confronto, floor, ablation, scontri diretti, best response, diagnostiche); restano la scrittura (bozza, README), le slide e la prova dell'orale.

1. Setup e verifica dell'ambiente (versione di Python), misura della velocità reale di addestramento sul server, avversari fissi e valutazione di base.
2. Implementazione di PPO e validazione con le curve contro il Random, poi self-play semplice e prime curve di apprendimento.
3. Varianti di reward (potenziale, vittoria scalata sul tempo) e confronto con la reward terminale.
4. Best response e matrice degli scontri diretti.
5. Bias e varianza (lambda, gamma), ablation secondarie e diagnostica del comportamento.
6. Slide, README e prova dell'orale.

| Rischio | Come si affronta |
| --- | --- |
| Il PPO scritto da noi ha un bug o impara peggio del previsto | Si controllano le curve contro il Random, loss, KL e frazione di campioni tagliati dal clipping (e si confronta con i valori del paper) |
| Tempo di calcolo insufficiente | Meno step e meno ablation; ambienti vettorizzati e, se serve, più processi |
| Le mosse speciali non vengono scoperte | Si attivano le azioni di carica come ablation |
| Le varianti di reward non producono differenze visibili | Si riporta come risultato e se ne discutono le cause (bias e varianza, dispersione tra i 5 seed) |
| L'agente diventa passivo | Si misura contro il no-op e lo si discute come limite |
| Exploitability rumorosa | Più seed per le best response e intervalli di confidenza |

Limiti dichiarati fin d'ora: con 5 seed per il confronto (1 per le altre run) i risultati restano indicativi e le ablation non separano l'effetto del metodo dalla variabilità tra addestramenti, quindi sono descrittivi (il confronto floor contro base ha un intervallo di confidenza da bootstrap sui seed, ampio); la best response approssima per difetto la sfruttabilità; la policy usa solo l'osservazione corrente (senza rete ricorrente né belief esplicito), quindi la parziale osservabilità è trattata in modo limitato; Footsies isola la sola fase neutrale, molto più semplice di un gioco di combattimento commerciale.
