import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from joblib import dump, load
import matplotlib.pyplot as plt
from text_preprocessing import _load_data
from codecarbon import EmissionsTracker
import time

pd.set_option('display.max_colwidth', None)

classifiers = {
    'SVM': SVC(class_weight="balanced"),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(class_weight="balanced"),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging Classifier': BaggingClassifier()
}

# consumi vari modelli
consumes = {}

def my_train_test_split(*datasets):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    The last dataset is used to stratify, to avoid test_sets without spam labels.
    '''
    return train_test_split(*datasets, test_size=0.3, stratify=datasets[-1], random_state=101)

#funzione per calcolare l'energia consumata in joule
def energy_stats(energy_consumption_kwh, energy_tracker):
    """Estrae e calcola le metriche sull'energia dal tracker dell'energia di codecarbon.
        IMPORTANTE: questa funzione dovrebbe essere chiamata subito dopo aver fermato il tracker.
    """
    energy_consumption_joules = energy_consumption_kwh * 1000 * 3600  # Joules
    duration = energy_tracker._last_measured_time - energy_tracker._start_time
    return energy_consumption_joules, duration

def train_classifier(classifier, algorithm, X_train, y_train):
    # Inizializzo l'EmissionsTracker per tracciare il consumo di energia durante il training
    predict_tracker = EmissionsTracker(save_to_file=False)
    predict_tracker.start()
    classifier.fit(X_train, y_train)
    # Fermo il tracker
    predict_energy_consumption_kwh = predict_tracker.stop()
    # statistiche sull'energia consumata
    predict_energy_consumption, _ = energy_stats(predict_energy_consumption_kwh, predict_tracker)
    consumes[algorithm] = predict_energy_consumption;
    time.sleep(60)

def predict_labels(classifier, X_test):
    return classifier.predict(X_test)


def generate_model(classifier, algorithm, raw_data, preprocessed_data):
    (X_train, X_test,
     _, _,
     y_train, y_test) = my_train_test_split(preprocessed_data,
                                            raw_data['message'],
                                            raw_data['label'])
    train_classifier(classifier, algorithm, X_train, y_train)
    return classifier, X_train, X_test, y_train, y_test

def model_validation(classifier, X_test, y_test):
    predictions = predict_labels(classifier, X_test)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label='spam'),
        "recall": recall_score(y_test, predictions, pos_label='spam'),
        "f1": f1_score(y_test, predictions, pos_label='spam')
    }
    return predictions, scores


def main():
    raw_data = _load_data()
    preprocessed_data = load('../dataset/preprocessed_data.joblib')

    pred_scores = dict()

    for algorithm, classifier in classifiers.items():
        classifier, _, X_test, _, y_test = generate_model(classifier, algorithm, raw_data, preprocessed_data)
        pred_scores[algorithm] = model_validation(classifier, X_test, y_test)[1]["accuracy"]
        #Salva il modello addestrato
        dump(classifier, f'../models/{algorithm}_model.joblib')
        #salva i dati di test
        dump((X_test, y_test), '../models/test_data.joblib')



    print('\n############### Accuracy Scores ###############')
    accuracy = pd.DataFrame.from_dict(pred_scores, orient='index', columns=['Accuracy Rate'])
    print('\n')
    print(accuracy)
    print('\n')

    # Plot accuracy scores in a bar plot
    accuracy.plot(kind='bar', ylim=(0.85, 1.0), edgecolor='black', figsize=(10, 5))
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout();
    plt.savefig("../output/accuracy_scores.png")

    # plot grafico consumi per addestramento
    print('Consumi trainig modelli in joule: ')
    print(consumes)

    plt.figure()
    plt.bar(consumes.keys(), consumes.values(), color='skyblue')
    plt.xlabel('Classifier')
    plt.ylabel('Consumo (Joule)')
    plt.title('Consumo dei vari classificatori in addestramento')
    plt.xticks(rotation=45, ha='right')  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Aggiunge una griglia sull'asse y

    plt.tight_layout();

    # Salvo il grafico
    plt.savefig("../output/consumo_scores.png")

if __name__ == "__main__":
    main()
