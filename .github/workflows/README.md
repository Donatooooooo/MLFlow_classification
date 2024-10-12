# ask_name_and_version.yml
È un'azione che ha come obbiettivo creare un commento in una pull request attivandosi quando viene creata o aggiornata la pull request stessa. Si compone di diverse operazioni:
1. Controllo dei commenti: La prima operazione che esegue è controllare se esiste già il commento all'interno della pull request. Questo commento chiede agli utenti di specificare il nome e la versione del modello in un file chiamato name_version.md, che si trova in una directory designata. Se il commento è già presente, l'azione non prosegue oltre.
2. Generazione del commento: Se il commento non è stato trovato, l'azione procede a generarlo e a postarlo nella pull request. Questo serve a garantire che chiunque lavori sulla pull request riceva le istruzioni necessarie per avviare il processo di creazione della Model Card.
3. Controllo dei file: L'azione verifica se due file specifici (name_version.md e _parts.md) esistono già nella directory ModelCardGenerator/Data.
4. Creazione dei file: Se i file non esistono, l'azione crea la directory necessaria e inizializza entrambi i file, rendendoli pronti per essere utilizzati.
5. Commit e push: Infine, se i file sono stati creati, l'azione esegue un push sul branch. 

# get_name_and_version.yml
È un'azione che ha come obbiettivo creare una Model Card in base al modello e alla versione inserita nel file name_version.md. Si attiva specificamente quando viene modificato il file ModelCardGenerator/Data/name_version.md all'interno di una pull request. L'azione si compone di diverse operazioni:
1. Controllo del contenuto del filename_version.md:  verifica se il file esiste e se non è vuoto. Se il file non esiste o è vuoto, l'azione si interrompe, evitando di procedere con passaggi non necessari.
2. Impostazione di Python: Se il file esiste e contiene dati, l'azione configura un ambiente Python.
3. Installazione delle dipendenze: Una volta impostato l'ambiente Python, l'azione installa le librerie necessarie per elaborare la Model Card.
4. Lettura del file e passaggio al programma principale: L'azione quindi legge il contenuto di name_version.md e lo passa a uno script Python chiamato main.py. Questo passaggio è fondamentale per avviare il processo di creazione della Model Card.
5. Controllo dell'esistenza del file generato: Dopo aver eseguito main.py, l'azione controlla se il file generato esiste già nella cartella ModelCards. Se il file è già presente, l'azione rimuove il file appena creato per evitare conflitti e non procede con il push delle modifiche.
6. Commit e push della Model Card: Se il file generato non esiste, l'azione configura le informazioni dell'utente per il commit, aggiunge i file al repository, effettua un commit con un messaggio predefinito e infine esegue un push per aggiornare la pull request con il nuovo file.
