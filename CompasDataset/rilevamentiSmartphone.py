from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from joblib import dump, load
import pandas as pd
import psutil
import time
import os
from codecarbon import EmissionsTracker
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

#riscaldamento del dispositivo
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

print("Inizio riscaldamento")

start_time = time.time()
end_time = start_time + 300  # 300 secondi = 5 minuti

fib_sequence = fibonacci_generator()

while time.time() < end_time:
    next(fib_sequence)

print("Riscaldamento effettuato")


#inizio a lavorare sul dataset
df = pd.read_csv('dataset/compas_processed.csv')

#definisco variabili dipendenti e non
X = df.drop(columns=['Probability'])  # Features
y = df['Probability']  # Target

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#funzione per calcolare l'energia consumata in joule
def energy_stats(energy_consumption_kwh, energy_tracker):
    """Estrae e calcola le metriche sull'energia dal tracker dell'energia di codecarbon.
        IMPORTANTE: questa funzione dovrebbe essere chiamata subito dopo aver fermato il tracker.
    """
    energy_consumption_joules = energy_consumption_kwh * 1000 * 3600  # Joules
    duration = energy_tracker._last_measured_time - energy_tracker._start_time
    return energy_consumption_joules, duration

from sklearn.metrics import accuracy_score

#addestro con vari algoritmi e calcolo per ogni algortimo il suo consumo energetico in training

classifiers = {
    'SVM': SVC(class_weight="balanced"),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(class_weight="balanced"),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging Classifier': BaggingClassifier()
}

#accuratezze dei vari modelli
accuracies = {}
#consumi vari modelli
consumes = {}

for algorithm, classifier in classifiers.items():
    # Inizializzo l'EmissionsTracker per tracciare il consumo di energia durante il training
    predict_tracker = EmissionsTracker(save_to_file=False)
    predict_tracker.start()
    # Addestro del modello
    classifier.fit(X_train, y_train)
    # Fermo il tracker
    predict_energy_consumption_kwh = predict_tracker.stop()
    # statistiche sull'energia consumata
    predict_energy_consumption, _ = energy_stats(predict_energy_consumption_kwh, predict_tracker)
    consumes[algorithm] = predict_energy_consumption;
    #Salvo il modello addestrato
    dump(classifier, f'models/{algorithm}_model.joblib')
    # Valutazione
    Y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, Y_pred)
    # Memorizzo l'accuratezza per stampare grafico dopo
    accuracies[algorithm] = accuracy
    time.sleep(60)


#plot grafico accuratezze

#grafico a barre
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.xlabel('Modello')
plt.ylabel('Accuracy')
plt.title('Accuratezza dei vari modelli')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout();
plt.savefig("output/accuracy_scores.png") # Salvo il grafico

#plot grafico consumi per addestramento

#grafico a barre
plt.bar(consumes.keys(), consumes.values(), color='skyblue')
plt.xlabel('Modello')
plt.ylabel('Consumo (Joule)')
plt.title('Consumo dei vari classificatori in addestramento')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(consumes.values()) * 1.4)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout();
plt.savefig("output/consumo_scores.png") # Salvo il grafico

# Carica i modelli addestrati
model_paths = {
    'SVM': 'models/SVM_model.joblib',
    'Decision Tree': 'models/Decision Tree_model.joblib',
    'KNN': 'models/KNN_model.joblib',
    'Random Forest': 'models/Random Forest_model.joblib',
    'AdaBoost': 'models/AdaBoost_model.joblib',
    'Bagging Classifier': 'models/Bagging Classifier_model.joblib'
}

# lista per salvare i risultati delle 5 run per ogni algoritmo
all_results = []

model_weight_kb_list = []  # Lista per salvare i pesi del modello

# Lista dei nomi degli algoritmi
algorithm_names = list(model_paths.keys())

# Numero di run per algoritmo
num_runs = 5

# Inizializzo un contatore per tenere traccia delle run per ciascun algoritmo
algorithm_counter = {name: 0 for name in algorithm_names}

# Inizializzo le liste per salvare i risultati di ogni run
cpu_percent_lists = {name: [] for name in model_paths}
mem_percent_lists = {name: [] for name in model_paths}
inference_time_lists = {name: [] for name in model_paths}
accuracy_lists = {name: [] for name in model_paths}
energy_consumption_lists = {name: [] for name in model_paths}
model_weight_kb_lists = {name: [] for name in model_paths}

# Continua fino a quando non sono state eseguite tutte le run per tutti gli algoritmi
while any(count < num_runs for count in algorithm_counter.values()):
    for name, path in model_paths.items():
        # Verifica se sono state eseguite tutte le run per questo algoritmo
        if algorithm_counter[name] < num_runs:
            # Pausa di 60 secondi (per raffreddamento CPU)
            time.sleep(60)
            # carico il modello
            model = load(path)
            # peso del modello
            model_weight_kb = os.path.getsize(path) / 1024
            # Inizializzo l'EmissionsTracker per tracciare il consumo di energia durante la predizione
            predict_tracker = EmissionsTracker(save_to_file=False)
            predict_tracker.start()
            # predizione
            start_time = time.time()
            predictions = model.predict(X_test)
            inference_time = time.time() - start_time
            # Misuro utilizzo della memoria
            mem_percent = psutil.virtual_memory().percent
            # Misuro utilizzo della CPU
            cpu_percent = psutil.cpu_percent()
            # accuratezza
            accuracy = model.score(X_test, y_test)
            # Fermo il tracker
            predict_energy_consumption_kwh = predict_tracker.stop()
            # statistiche sull'energia consumata
            predict_energy_consumption, _ = energy_stats(predict_energy_consumption_kwh, predict_tracker)
            # Salvo i risultati del run
            cpu_percent_lists[name].append(cpu_percent)
            mem_percent_lists[name].append(mem_percent)
            inference_time_lists[name].append(inference_time)
            accuracy_lists[name].append(accuracy)
            energy_consumption_lists[name].append(predict_energy_consumption)
            model_weight_kb_lists[name].append(model_weight_kb)
            # Aggiorno il contatore per questa run
            algorithm_counter[name] += 1
            # Dopo aver fatto una run per ogni algoritmo, calcolo le medie
            if all(count >= num_runs for count in algorithm_counter.values()):
                # Calcolo le medie dei risultati
                for name in model_paths:
                    avg_cpu_percent = np.mean(cpu_percent_lists[name])
                    avg_mem_percent = np.mean(mem_percent_lists[name])
                    avg_inference_time = np.mean(inference_time_lists[name])
                    avg_accuracy = np.mean(accuracy_lists[name])
                    avg_energy_consumption = np.mean(energy_consumption_lists[name])
                    avg_model_weight_kb = np.mean(model_weight_kb_lists[name])
                    # Salvo le medie dei risultati per questo algoritmo
                    all_results.append({
                        'Modello': name,
                        'Utilizzo CPU (%)': avg_cpu_percent,
                        'Utilizzo memoria (%)': avg_mem_percent,
                        'Tempo inferenza (s)': avg_inference_time,
                        'Accuratezza': avg_accuracy,
                        'Consumo energia (J)': avg_energy_consumption,
                        'Peso del modello (KB)': avg_model_weight_kb
                    })

# Creo un DataFrame pandas con le medie dei risultati ottenuti.
results_df = pd.DataFrame(all_results)

# Imposta l'opzione per visualizzare tutte le colonne del DataFrame
pd.set_option('display.max_columns', None)

# Salvo i risultati in un file CSV
results_df.to_csv('output/results_stats.csv', index=False)

# Grafico per l'utilizzo della memoria e l'accuratezza

fig, ax1 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', x='Modello', y='Utilizzo memoria (%)', ax=ax1, color='blue', label='Utilizzo memoria (%)')
ax1.set_ylabel('Utilizzo Memoria (%)')
ax1.set_title('Utilizzo Memoria (%) e Accuratezza per Modello')
ax1.set_xticklabels(results_df['Modello'], rotation=45, ha='right')
ax1.legend(loc='upper left')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
ax2 = ax1.twinx()
results_df.plot(kind='line', x='Modello', y='Accuratezza', ax=ax2, color='red', marker='o', label='Accuratezza')
# track accuratezza
ax2.set_ylabel('Accuratezza')
ax2.legend(loc='upper right')
plt.savefig('output/utilizzo_memoria_accuratezza.png')
plt.close()

# Grafico per il tempo di inferenza e l'accuratezza
fig, ax1 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', x='Modello', y='Tempo inferenza (s)', ax=ax1, color='green', label='Tempo inferenza (s)')
ax1.set_ylabel('Tempo Inferenza (s)')
ax1.set_title('Tempo Inferenza (s) e Accuratezza per Modello')
ax1.set_xticklabels(results_df['Modello'], rotation=45, ha='right')
ax1.legend(loc='upper left')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
# track accuratezza
ax2 = ax1.twinx()
results_df.plot(kind='line', x='Modello', y='Accuratezza', ax=ax2, color='red', marker='o', label='Accuratezza')
ax2.set_ylabel('Accuratezza')
ax2.legend(loc='upper right')
plt.savefig('output/tempo_inferenza_accuratezza.png')
plt.close()

# Grafico per il consumo di energia e l'accuratezza
fig, ax1 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', x='Modello', y='Consumo energia (J)', ax=ax1, color='purple', label='Consumo energia (J)')
ax1.set_ylabel('Consumo Energia (J)')
ax1.set_title('Consumo Energia (J) e Accuratezza per Modello')
ax1.set_xticklabels(results_df['Modello'], rotation=45, ha='right')
ax1.legend(loc='upper left')
plt.tight_layout()
# Regola i margini per lasciare spazio per le etichette sull'asse x
plt.subplots_adjust(left=0.1, right=0.9)
# track accuratezza
ax2 = ax1.twinx()
results_df.plot(kind='line', x='Modello', y='Accuratezza', ax=ax2, color='red', marker='o', label='Accuratezza')
ax2.set_ylabel('Accuratezza')
ax2.legend(loc='upper right')

plt.savefig('output/consumo_energia_accuratezza.png')
plt.close()

# Grafico per il peso del modello e l'accuratezza
fig, ax1 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', x='Modello', y='Peso del modello (KB)', ax=ax1, color='orange', label='Peso del modello (KB)')
ax1.set_ylabel('Peso del modello (KB)')
ax1.set_title('Peso del Modello (KB) e Accuratezza')
ax1.set_xticklabels(results_df['Modello'], rotation=45, ha='right')
ax1.legend(loc='upper left')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
# track accuratezza
ax2 = ax1.twinx()
results_df.plot(kind='line', x='Modello', y='Accuratezza', ax=ax2, color='red', marker='o', label='Accuratezza')
ax2.set_ylabel('Accuratezza')
ax2.legend(loc='upper right')
plt.savefig('output/peso_modello_accuratezza.png')
plt.close()


# Grafico per la percentuale CPU e l'accuratezza
fig, ax1 = plt.subplots(figsize=(10, 6))
results_df.plot(kind='bar', x='Modello', y='Utilizzo CPU (%)', ax=ax1, color='blue', label='Utilizzo CPU (%)')
ax1.set_ylabel('Utilizzo CPU (%)')
ax1.set_title('Utilizzo CPU (%) e Accuratezza per Modello')
ax1.set_xticklabels(results_df['Modello'], rotation=45, ha='right')
ax1.legend(loc='upper left')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9)
# track accuratezza
ax2 = ax1.twinx()
results_df.plot(kind='line', x='Modello', y='Accuratezza', ax=ax2, color='red', marker='o', label='Accuratezza')
ax2.set_ylabel('Accuratezza')
ax2.legend(loc='upper right')
plt.savefig('output/utilizzo_cpu_accuratezza.png')
plt.close()

#calcolo delle correlazioni

# Rimuovo la colonna 'Modello' dal DataFrame (dato che è un data testuale)
results_df_numeric = results_df.drop(columns=['Modello'])

# matrice correlazione
correlation_matrix = results_df_numeric.corr()

#plot matrice
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Matrice di correlazione tra le variabili')
plt.tight_layout()
plt.savefig('output/matrice_correlazione.png')

# Correlazioni [no matrice]
accuracy_series = results_df['Accuratezza']
columns_to_correlate = results_df_numeric.drop(columns=['Accuratezza'])
correlation_with_accuracy = columns_to_correlate.corrwith(accuracy_series)
correlation_df = pd.DataFrame({'Variabile': correlation_with_accuracy.index, 'Correlazione con Accuratezza': correlation_with_accuracy.values})
# Salvo ed esporto in un file CSV
correlation_df.to_csv('output/results_correlazione.csv', index=False)



