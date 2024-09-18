# MLFlow
Negli ultimi anni lo sviluppo e l'utilizzo di modelli di machine learning è diventato sempre più significativo. L'implementazione del software ML richiede varie fasi di sperimentazione fondamentali per garantirne il corretto funzionamento: gli sviluppatori utilizzano costantemente nuovi dataset, librerie e diversi iperparametri. Di fronte alla necessità di gestire i modelli testati e le informazioni affini, sono state introdotte varie piattaforme progettate per semplificare la gestione dell'intero ciclo di vita di strumenti ML.

MLFlow è una piattaforma open source per la gestione del ciclo di vita di modelli apprendimento automatico progettata per funzionare con qualsiasi libreria e linguaggio di programmazione. Proprio come Databricks afferma: MLflow si concentra sull'intero ciclo di vita dei progetti di apprendimento automatico, assicurando che ogni fase sia gestibile, tracciabile e riproducibile.

MLFlow fornisce una serie di strumenti volti a semplificare il flusso di lavoro in ambito machine learning:
1. Tracking: fornisce un API e una UI per la registrazione di parametri, versioni di codice, metriche e artefatti durante il processo ML. Attraverso la cattura delle informazioni, Tracking facilita la registrazione dei risultati su file locali o su un server, semplificando il confronto di più esecuzioni tra diversi utenti.
2. Model Registry: aiuta a gestire diverse versioni dei modelli, a discernere il loro stato attuale e a garantire una produzione efficiente. Con Model Registry è possibile gestire in modo collaborativo l'intero ciclo di vita di un modello MLflow, inclusi lignaggio del modello, versioning, aliasing, tagging e annotazioni.
3. MLflow Deployments for LLMs:  permette di monitorare e aggiornare modelli LLM in modo efficace attraverso l'implementazione e la gestione di modelli su infrastrutture di produzione.
4. Evaluate: consente di automatizzare la valutazione di modelli addestrati e fornisce un sistema standardizzato per calcolare metriche di performance, sia standard che custom. Evaluate genera dei report dettagliati utili per confrontare diversi modelli.
5. Recipes: sono flussi di lavoro modulari preconfigurati per gestire processi di machine learning che aiutano a standardizzare ogni fase del processo ML attraverso template predefiniti e riutilizzabili.
6. Projects: permette di condividere facilmente esperimenti tra membri di un team o eseguire lo stesso esperimento in ambienti diversi, garantendo che le esecuzioni siano ripetibili indipendentemente dalla macchina o dall'utente che le esegue.

Sono stati analizzati gli strumenti 'Tracking' e 'Model Registry'.

## Test eseguito
Per poter studiare i due strumenti viene utilizzato un modello di apprendimento supervisionato. Attraverso la random forest è possibile identificare la categoria a cui appartiene una nuova osservazione, basandosi su un insieme di dati di addestramento che contengono osservazioni già etichettate. 

Dopo una breve fase di preprocessing dei dati viene addestrato il modello. Per confrontare diverse opzioni tramite MLFLow, viene addestrata inizialmente una random forest e successivamente una random forest integrata con k-Means. Al termine dell'addestramento entra in gioco MLFlow attraverso il comando 'mlflow.start_run()'.

### Tracking
Tutti gli esperimenti, distinti dalle run, vengono salvati da MLFlow. Attravreso l'interfaccia grafica è possibile metterli a confronto.

![experiments](Evaluation\img\experiments.png)

In ogni esperimento vengono salvate le metriche di valutazione, gli iperparametri scelti tramite GridSearch e informazioni generiche. 

![sample](Evaluation\img\sample.png)

Gli artifacts sono file e dati generati o utilizzati durante il ciclo di vita di un esperimento di machine learning. Questi possono includere i modelli addestrati, le configurazioni e i dati di input/output. Gli artifacts sono fondamentali per tracciare e riprodurre esperimenti, poiché contengono informazioni cruciali per comprendere come è stato addestrato e utilizzato un modello.

![artifacts](Evaluation\img\artifacts.png)

### Model Registry

Tutti i modelli, con le diverse versioni, tag e altre informazioni utili, vengono salvate in un archivio centralizzato. 

![registry](Evaluation\img\registry.png)

## Fonti consultate
A. Chen et al., “Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle,” in Proceedings of the 4th Workshop on Data Management for End-To-End Machine Learning, DEEM 2020 - In conjunction with the 2020 ACM SIGMOD/PODS Conference, 2020. doi: 10.1145/3399579.3399867.

https://mlflow.org/docs/latest