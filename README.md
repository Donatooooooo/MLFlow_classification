# Indice
1. [**Struttura del repository**](#struttura-del-repository)
2. [**MLFlow**](#mlflow)
   - [Test 1: Tracciare il modello attraverso MLFlow](#test-1-tracciare-il-modello-attraverso-mlflow)
   - [Test 2: Creazione di una Model Card attraverso le informazioni tracciate](#test-2-creazione-di-una-model-card-attraverso-le-informazioni-tracciate)
3. [**Fonti consultate**](#fonti-consultate)

# Struttura del repository
```
|   README.md                                            <- File descrittivo della repository.
|   
+---.github/workflows                                    <- Directory che contiene le Github Actions.
+---mlartifacts                                          <- Directory usata da MLFlow per lo storage degli artifacts.
+---mlruns                                               <- Directory usata da MLFlow per lo storage degli esperimenti e run relativi.
+---ModelCardsGenerator
|   +---Data                                             <- Directory di supporto al workflow di creazione della Model Card.
|   +---src
|       |   ModelCardGenerator.py                        <- Script che crea la Model Card.
|       |   ModelCardIntegrator.py                       <- Script che integra le informazioni della Model Card.
|       +---Utils
|           |   exceptions.py                            <- Eccezioni personalizzate.
|           |   logger.py                                <- Gestisce l'output del programnma di creazione della Model Card.
|           |   utility.py                               <- Contiene delle funzioni di supporto per la creazione della Model Card.
|           +---Templates
|               |   modelCard_template.md                <- Template usato per la creazione della Model Card.
|               +---_parts                               <- Directory contenente i template usati per l'integrazione della Model Card.
+---ModelTracker
    |   main.py                                          <- Script che avvia i task di classificazione.
    |   MLFlowTracker.py                                 <- Gestisce il tracciamento dei modelli.
    +---Classifiers                                      <- Directory contenente i modelli usati per la classificazione.
    +---Dataset
    |       brest_cancer.csv                             <- Dataset usato per l'addestramento dei modelli.
    |       dataset.py                                   <- Gestisce un dataset csv.
    +---Utils
        |   kmeans.py                                    <- Contiene i metodi per effettuare il clustering di dati.
        |   predictions.csv                              <- File di predizioni usato come test nel caricamento di artifacts.
        |   utility.py                                   <- Contiene delle funzioni di supporto per l'addestramento e il tracciamento.
        +---best_params                                  <- Directory che contiene i migliori iperparametri trovati 
        +---img                                          <- Directory contenente le immagini usate nel file README.md.
```
# MLFlow
Negli ultimi anni lo sviluppo e l'utilizzo di modelli di machine learning è diventato sempre più significativo. L'implementazione del software ML richiede varie fasi di sperimentazione fondamentali per garantirne il corretto funzionamento: gli sviluppatori utilizzano costantemente nuovi dataset, librerie e diversi iperparametri. Di fronte alla necessità di gestire i modelli testati e le informazioni affini, sono state introdotte varie piattaforme progettate per semplificare la gestione dell'intero ciclo di vita di strumenti ML.

MLFlow è una piattaforma open source per la gestione del ciclo di vita di modelli apprendimento automatico progettata per funzionare con qualsiasi libreria e linguaggio di programmazione. Secondo la documentazione fornita da Databricks, MLflow si concentra sull'intero ciclo di vita dei progetti di apprendimento automatico, assicurando che ogni fase sia gestibile, tracciabile e riproducibile.

MLFlow fornisce una serie di strumenti volti a semplificare il flusso di lavoro in ambito machine learning:
1. Tracking: fornisce un API e una UI per la registrazione di parametri, versioni di codice, metriche e artefatti durante il processo ML. Attraverso la cattura delle informazioni, Tracking facilita la registrazione dei risultati su file locali o su un server, semplificando il confronto di più esecuzioni tra diversi utenti.
2. Model Registry: aiuta a gestire diverse versioni dei modelli, a discernere il loro stato attuale e a garantire una produzione efficiente. Con Model Registry è possibile gestire in modo collaborativo l'intero ciclo di vita di un modello MLflow, inclusi lignaggio del modello, versioning, aliasing, tagging e annotazioni.
3. MLflow Deployments for LLMs:  permette di monitorare e aggiornare modelli LLM in modo efficace attraverso l'implementazione e la gestione di modelli su infrastrutture di produzione.
4. Evaluate: consente di automatizzare la valutazione di modelli addestrati e fornisce un sistema standardizzato per calcolare metriche di performance, sia standard che custom. Evaluate genera dei report dettagliati utili per confrontare diversi modelli.
5. Recipes: sono flussi di lavoro modulari preconfigurati per gestire processi di machine learning che aiutano a standardizzare ogni fase del processo ML attraverso template predefiniti e riutilizzabili.
6. Projects: permette di condividere facilmente esperimenti tra membri di un team o eseguire lo stesso esperimento in ambienti diversi, garantendo che le esecuzioni siano ripetibili indipendentemente dalla macchina o dall'utente che le esegue.

Sono stati analizzati gli strumenti 'Tracking' e 'Model Registry'.

## Test 1: Tracciare il modello attraverso MLFlow
Per studiare i due strumenti di MLFlow, viene utilizzato un task di classificazione su un dataset che ha come obiettivo la diagnosi di tumori al seno. Il modello deve classificare se un tumore è maligno o benigno, basandosi sui valori di un insieme di feature (ad esempio, dimensioni e forma della massa tumorale). Poiché i dati di addestramento contengono osservazioni precedentemente etichettate, il problema rientra nell’apprendimento supervisionato. Dunque, il modello potrà generalizzare sui nuovi dati in input fornendo una previsione in base alle informazioni apprese in fase di addestramento. 

Per risolvere il problema viene utilizzato il modello Random Forest. La Random Forest è un algoritmo ensamble di apprendimento supervisionato che funziona combinando diversi decision tree. Ogni decision tree viene addestrato su un campione casuale di dati di training e la previsione finale è il voto della maggioranza degli alberi sulla categoria predetta. La random Forest è stata scelta principalmente per la sua robustezza e per la capacità di generalizzare sui nuovi dati con un basso rischio di overfitting.

Dopo una breve fase di preprocessing dei dati, la Random Forest viene addestrata attraverso i dati di training. Successivamente, per esplorare le potenzialità di MLFlow oltre che strategie di miglioramento delle prestazioni, viene testata una variante del modello, combinando il clustering k-Means nel preprocessing. Questo permette di identificare gruppi simili di dati prima di applicare la Random Forest, fornendo un approccio potenzialmente più informato al task di classificazione. 

L'addestramento del modello avviene all'interno di un 'run' di MLFlow. Un 'run' è un singolo esperimento di machine learning attraverso cui è possibile tracciare tutte le informazioni come iperparametri, metriche, artifacts e modelli in modo semplice e organizzato. Inoltre, svolgendo l'addestramento all'interno di un 'run', è possibile monitorare lo stato e l'avanzamento del processo in tempo reale attraverso l'interfaccia MLFlow. 

### Tracking
Come già detto, MLFlow tracking è organizzato attorno al concetto di 'run'. Attraverso il comando 'mlflow.start_run()' tutte le operazioni di logging vengono associate al run corrente, permettendo di distinguere i diversi esperimenti. Inoltre, ogni volta che viene avviato un run, MLflow assegna un ID univoco al run stesso, facilitando la ripetizione degli esperimenti e il confronto attraverso l'interfaccia grafica messa a disposizione. Va specificato che, attraverso MLFlow Tracking, le informazioni tracciate possono essere salvate in locale o su un server remoto.

![experiments](/ModelTracker/Utils/img/experiments.png)

In ogni esperimento vengono salvate le metriche di valutazione, gli iperparametri scelti tramite GridSearch e informazioni legate al modello, allo stato e al dataset utilizzato oltre che informazioni di carattere più generale. 

![sample](/ModelTracker/Utils/img/sample.png)

Gli artifacts sono file e dati generati o utilizzati durante il ciclo di vita di un esperimento di machine learning. Questi possono includere i modelli addestrati, le configurazioni e i dati di input/output. Gli artifacts sono fondamentali per tracciare e riprodurre esperimenti, poiché contengono informazioni cruciali per comprendere come è stato addestrato e utilizzato un modello. Alcuni artifact vengono generati automaticamente da MLFLow durante il tracciamento del modello. Per testare il tracciamento manuale degli artifacts è stato sfruttato il modello caricato per calcolare delle predizioni e salvarle nel file predictions.csv che viene loggato al termine delle operazioni. 

![artifacts](/ModelTracker/Utils/img/artifacts.png)

### Model Registry

Tutti i modelli, con le diverse versioni, tag e altre informazioni utili, vengono salvate in un archivio centralizzato: il Model Registry.

![registry](/ModelTracker/Utils/img/registry.png)

## Test 2: Creazione di una Model Card attraverso le informazioni tracciate
Come già evidenziato, negli ultimi anni si è registrata un'espansione significativa nell'impiego di modelli di Machine Learning, anche in contesti critici (come ad esempio la medicina), dove le decisioni possono avere conseguenze di grande rilevanza. In un tale contesto, la documentazione assume un ruolo cruciale per poter comprendere come vengono prese le decisioni e quali dati influenzano i processi decisionali. Di conseguenza, è fondamentale che gli sviluppatori si impegnino a fornire una documentazione chiara sui loro modelli.

Un tipo di documentazione popolare per i modelli di Machine Learning sono le Model Cards. Ampiamente sfruttate su Hugging Face, una piattaforma incentrata sullo sviluppo e sulla condivisione di modelli di apprendimento, una Model Card è un documento che fornisce una descrizione dell'algoritmo attraverso delle informazioni essenziali come ad esempio i task che è in grado di compiere, le prestazioni, dati di addestramento e metriche di valutazione, oltre che le limitazioni del modello stesso. Lo scopo principale di una Model Card è quello di favorire una comprensione approfondita garantendo un uso responsabile dei modelli di intelligenza artificiale.

Una Model Card segue un formato standard che include generalmente:
- Nome del modello: Identifica il modello e il suo sviluppatore.
- Descrizione: Descrive brevemente il modello e i suoi obiettivi.
- Dati di addestramento: Spiega su quali dati è stato addestrato il modello.
- Metriche di prestazione: Include i risultati delle valutazioni delle prestazioni del modello.
- Limitazioni: Specifica i limiti d'uso del modello e i contesti in cui potrebbe non essere adeguato.
- Uso raccomandato: Spiega in quali contesti è adatto il modello.

Creare la documentazione, tuttavia, può rivelarsi un processo impegnativo e oneroso in termini di tempo. Pertanto, è essenziale avere a disposizione degli strumenti che consentano di generarla automaticamente, in particolare per tutte quelle informazioni standard come prestazioni, dati e parametri di addestramento e versioni del modello, al fine di garantirne una certificazione adeguata. L'automazione diventa particolarmente importante quando si utilizzano strumenti come MLFlow, che registrano automaticamente informazioni durante la fase di addestramento, rendendole immediatamente disponibili per la documentazione senza la necessità di riscriverle manualmente.

Il test eseguito si è concentrato sulla raccolta di informazioni archiviate tramite MLFlow e relative a un modello specifico, identificato tramite nome e versione. Queste informazioni sono state poi organizzate e inserite in un file markdown, utilizzando quindi il formato di file delle Model Cards. Grazie all'uso di MLFlow e alle informazioni tracciate durante il Test 1, è stato possibile generare automaticamente delle Model Cards di modelli registrati nel Model Registry. 

L'interrogazione del Model Registry avviene mediante la classe 'MlFlowCLient', la quale fornisce diversi metodi per ottenere i dettagli di un modello. Una volta trovata la versione del modello specificato, viene estratto il relativo run ID, ossia un identificatore unico assegnato a ogni esecuzione di un esperimento.

L'identificatore ottenuto consente di individuare la run specifica all'interno degli esperimenti svolti. Grazie alla run è possibile recuperare una vasta gamma di informazioni, tra cui metriche, parametri scelti e altri dati generali che descrivono il contesto dell’addestramento (come ad esempio il dataset utilizzato) relativi al modello. Quindi, i dati vengono opportunamente strutturati, automatizzando la creazione di alcune sezioni della Model Card.

Per testare la versatilità dell'algoritmo di creazione delle Model Cards è stato inserito nel progetto un K-nearest neighbors Classifier che svolge lo stesso task di classificazione della Random Forest del Test 1 e viene tracciato da MLFlow allo stesso modo. Tutte le Model Cards create in fase di test sono state salvate nella cartella ModelCards del repository e hanno la seguente struttura:

---
### Model Name - version
#### General Information 
- Developed by: indica chi ha sviluppato il modello
- Model Type: specifica il modello utilizzato
- Libreria usata per l'apprendimento: indica il nome e la versione
- Python Version: indica la versione Python utilizzata
#### Training Details
- Dataset: specifica il dataset utilizzato
- Parameters: indica i parametri usati per l'apprendimento
   - `Elenco dei parametri`
- Training started at: specifica il momento in cui è iniziato l'addestramento
- Training ended at: specifica il momento in cui è finito l'addestramento
#### Evaluation
   - `Elenco delle metriche`: elenco di tutte le metriche usata per valutare il modello
---

# Fonti consultate
A. Chen et al., “Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle,” in Proceedings of the 4th Workshop on Data Management for End-To-End Machine Learning, DEEM 2020 - In conjunction with the 2020 ACM SIGMOD/PODS Conference, 2020. doi: 10.1145/3399579.3399867.

https://mlflow.org/docs/2.16.2/index.html

https://huggingface.co/docs/hub/model-cards
