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

## Test eseguito
Per studiare i due strumenti di MLFlow, viene utilizzato un task di classificazione su un dataset che ha come obiettivo la diagnosi di tumori al seno. Il modello deve classificare se un tumore è maligno o benigno, basandosi sui valori di un insieme di feature (ad esempio, dimensioni e forma della massa tumorale). Poiché i dati di addestramento contengono osservazioni precendentemente etichettate, il problema rientra nell’apprendimento supervisionato. Dunque, il modello potrà generalizzare sui nuovi dati in input fornendo una previsione in base alle informazioni apprese in fase di addestramento. 

Per risolvere il problema, viene utilizzato il modello Random Forest. La Random Forest è un algoritmo ensamble di apprendimento supervisionato che funziona combinando diversi decision tree. Ogni decision tree viene addestrato su un campione casuale dei dati, e la previsione finale viene è il voto della maggioranza degli alberi. La random Forest è stata scelta principalmente per la sua robustezza e per la capacità di generalizzare sui nuovi dati con un basso rischio di overfitting.

Dopo una breve fase di preprocessing dei dati, la Random Forest viene addestrata attraverso i dati di training. Successivamente, per esplorare diverse strategie di miglioramento delle prestazioni, viene testata una variante del modello, combinando il clustering k-Means nel preprocessing. Questo permette di identificare gruppi simili di dati prima di applicare la Random Forest, fornendo un approccio potenzialmente più informato al task di classificazione. 

L'addestramento del modello avviene all'interno di un 'run' di MLFlow. Un 'run' è un singolo esperimento di machine learning attraverso cui è possibile tracciare tutte le informazioni come iperparametri, metriche, artefatti e modelli in modo semplice e organizzato.

### Tracking
Tutti gli esperimenti, distinti dalle run, vengono salvati da MLFlow. Attraverso l'interfaccia grafica è possibile metterli a confronto.

![experiments](/Evaluation/img/experiments.png)

In ogni esperimento vengono salvate le metriche di valutazione, gli iperparametri scelti tramite GridSearch e informazioni generiche. 

![sample](/Evaluation/img/sample.png)

Gli artifacts sono file e dati generati o utilizzati durante il ciclo di vita di un esperimento di machine learning. Questi possono includere i modelli addestrati, le configurazioni e i dati di input/output. Gli artifacts sono fondamentali per tracciare e riprodurre esperimenti, poiché contengono informazioni cruciali per comprendere come è stato addestrato e utilizzato un modello.

![artifacts](/Evaluation/img/artifacts.png)

### Model Registry

Tutti i modelli, con le diverse versioni, tag e altre informazioni utili, vengono salvate in un archivio centralizzato. 

![registry](/Evaluation/img/registry.png)

## Fonti consultate
A. Chen et al., “Developments in MLflow: A System to Accelerate the Machine Learning Lifecycle,” in Proceedings of the 4th Workshop on Data Management for End-To-End Machine Learning, DEEM 2020 - In conjunction with the 2020 ACM SIGMOD/PODS Conference, 2020. doi: 10.1145/3399579.3399867.

https://mlflow.org/docs/latest
