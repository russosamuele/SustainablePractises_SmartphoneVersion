# Analisi del Consumo Energetico e Performance dei Modelli di Machine Learning [Smartphone Version]

# Descrizione del Progetto

Durante il mio tirocinio, ho lavorato su differenti dataset e ho addestrato vari modelli utilizzando una serie di algoritmi, tra cui K-Nearest Neighbors (KNN), Random Forest (RF), Decision Tree Classifier (DCT), Adaptive Boosting (AdaBoost), Support Vector Machine (SVM) e altri. 
Il focus principale del mio lavoro è stato non solo valutare l'accuratezza dei modelli, ma anche misurare il loro consumo energetico in joule durante la fase di addestramento.
Successivamente, durante la fase di predizione, per ogni modello, ho calcolato diverse metriche chiave, tra cui l'accuratezza, il peso del modello, il consumo energetico, l'utilizzo della CPU e della memoria. 
Queste misurazioni sono state ripetute più volte, anche in maniera randomica tra i vari modelli, ed è stata calcolata la media per ottenere risultati più accurati.
Tra i vari run sono state considerate anche una serie di pause, e un riscaldamento iniziale per avere dei risultati più accurati.
Infine, ho generato una serie di grafici per visualizzare i risultati ottenuti, inclusi plot delle correlazioni tra l'accuratezza dei modelli e le metriche rilevate.

# Organizzazione progetto
Il progetto si divide in 3 cartelle distinte, una per ogni dataset. All'interno di ognuna di queste cartelle è possibile trovare, nelle corrispondenti sottocartelle, il dataset, i modelli addestrati, i vari plot e risultati generati. Il core è, per ogni dataset, il file rilevamentiSmartphone.py, che effettuerà in autonomia tutto, a partire dal processing dei dati, al training, fino ad arrivare al plot dei risultati. 

# Ambiente di Lavoro
Tutto il lavoro è stato eseguito utilizzando uno smartphone, con sistema operativo Kali-linux, più precisamente un Samsung galaxy A70, 6+128 anno 2022.


