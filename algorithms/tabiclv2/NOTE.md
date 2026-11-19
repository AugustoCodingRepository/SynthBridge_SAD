# TabICLv2 in SynthBridge – note

## Scelte fatte
- **Da predittore a generatore.** TabICLv2 nasce per classificazione e regressione. Per generare dati si usa la
  regola della catena, P(X) = P(X1) · P(X2 | X1) · … · P(Xd | X1..Xd-1): ogni colonna è campionata dalla
  distribuzione predittiva di TabICLv2, condizionata alle colonne già generate. È l'approccio della classe
  ufficiale `tabicl.TabICLUnsupervised` (libreria `tabicl` 2.2.0, checkpoint v2 del paper).
- **Colonne categoriche** → classificatore TabICLv2 (probabilità delle classi).
  **Colonne numeriche** → regressore TabICLv2 a 999 quantili (distribuzione predittiva completa).
- **Bilanciamento.** Con `target_column` e `target_class` si usano come contesto solo le righe reali della classe
  indicata e si generano solo righe di quella classe. Con il solo `target_column`, il target viene generato per
  primo e tutte le altre colonne sono condizionate su di esso.
- L'adapter ripete il ciclo di `TabICLUnsupervised.generate()` colonna per colonna per aggiornare la barra di
  avanzamento; per questo la versione di `tabicl` è fissata a 2.2.0.
- Colonne costanti (o tutte mancanti) non vengono modellate ma copiate.
- I valori numerici generati sono riportati entro [min, max] dei dati reali (`clip_to_range`, disattivabile).

## Limiti
- Le colonne sono generate nell'ordine del dataset: la qualità può dipendere dall'ordine.
- Le colonne intere con pochi valori (es. `cyl` di mtcars) sono trattate come numeriche: possono comparire valori
  intermedi (es. 7). Se sono in realtà categorie, convertirle in fattore prima di generare.
- I dati sintetici non contengono valori mancanti, anche se i dati reali li hanno.
- Al primo uso vengono scaricati da Hugging Face i pesi del classificatore e del regressore (serve internet).
- TabICLv2 è pre-addestrato con al massimo 100 feature.

## Tempi di esecuzione osservati
Prove del 19/11/2026 su PC Windows, dispositivo `auto`, parametri di default
(4 stimatori, temperatura 1, max 3000 righe di contesto). Tempi comprensivi del caricamento del modello.

| Dataset | Righe reali × colonne | Righe sintetiche | Modalità | Tempo |
|---|---|---|---|---|
| airquality | 153 × 6 (con NA) | 200 | intero dataset | 19,8 s |
| iris | 150 × 5 | 200 | intero dataset | 7,8 s |
| mtcars | 32 × 11 | 200 | intero dataset | 9,5 s |
| iris | 50 × 4 (solo `virginica`) | 100 | bilanciamento (`target_class = virginica`) | 6,2 s |

Il primo run (airquality) è il più lento perché include il primo caricamento dei pesi.
Il tempo cresce circa linearmente con il numero di colonne (una predizione di TabICLv2 per colonna).

## Osservazioni sulle prove
- **Fedeltà marginale** (distanza KS per le numeriche, TVD per le categoriche, calcolate da SynthBridge):
  airquality 0,03–0,11; iris 0,03–0,13. Le distribuzioni delle singole colonne sono riprodotte bene.
- **mtcars**: solo 32 righe di contesto, quindi fedeltà più bassa (KS fino a 0,33). Inoltre in R le colonne
  `cyl`, `vs`, `am`, `gear`, `carb` sono di tipo `double`, quindi vengono generate come numeriche continue
  (es. `cyl = 5,99`). Per trattarle come categorie vanno convertite in fattore prima della generazione.
- **Bilanciamento**: con `target_class = virginica` tutte le righe generate sono `virginica`. Le distanze
  riportate da SynthBridge sono alte (KS fino a 0,60, TVD 0,67) perché il confronto è fatto con l'intero
  dataset (tre specie) e non con le sole righe `virginica`: è un effetto atteso, non un errore.
  Per valutare la fedeltà in questa modalità occorre confrontare con le sole righe reali della classe.