import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tkinter as tk
from tkinter import messagebox,filedialog
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt


# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')

# Dossier contenant les documents bruts
corpus_dir = "corpus2"  # A remplacer par le chemin de votre dossier


# Données à sauvegarder
true_labels = [
    [1, 0, 1, 1, 0],  # Vérité terrain pour requête 1
    [1, 1, 0, 0, 1]   # Vérité terrain pour requête 2
]

predicted_labels_w2v = [
    [1, 0, 0, 1, 1],  # Prédictions pour requête 1 avec Word2Vec
    [1, 1, 1, 0, 0]   # Prédictions pour requête 2 avec Word2Vec
]

predicted_labels_sbert = [
    [1, 1, 0, 1, 0],  # Prédictions pour requête 1 avec SBERT
    [1, 1, 1, 0, 1]   # Prédictions pour requête 2 avec SBERT
]

# Lire tous les fichiers de texte
documents = []
document_names = []  # Liste pour stocker les noms de fichiers
domaines = []  # Liste pour stocker les domaines des fichiers (dossiers)

# Parcourir tous les sous-dossiers du dossier corpus_dir
for domaine in os.listdir(corpus_dir):
    domaine_path = os.path.join(corpus_dir, domaine)
    
    # Vérifier si c'est un dossier
    if os.path.isdir(domaine_path):
        for file in os.listdir(domaine_path):
            if file.endswith(".txt"):
                file_path = os.path.join(domaine_path, file)
                
                # Lire le contenu du fichier et ajouter à la liste des documents
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    document_names.append(file)  # Ajouter le nom du fichier
                    domaines.append(domaine)  # Ajouter le domaine (nom du sous-dossier)

print(f"Nombre de documents chargés : {len(documents)}")

# Nettoyage des documents
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Zéèàçâêîôûäëïöüù\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if len(word) > 2])
    return text

documents_cleaned = [clean_text(doc) for doc in documents]
documents_tokenized = [word_tokenize(doc) for doc in documents_cleaned]
stop_words = set(stopwords.words('french'))
documents_filtered = [[word for word in doc if word not in stop_words] for doc in documents_tokenized]

nlp = spacy.load("fr_core_news_md")
def lemmatize_text(doc):
    doc_spacy = nlp(" ".join(doc))
    return [token.lemma_ for token in doc_spacy if not token.is_punct]

documents_lemmatized = [lemmatize_text(doc) for doc in documents_filtered]
model_w2v = Word2Vec(sentences=documents_lemmatized, vector_size=200, window=10, min_count=2, workers=4, sg=0, epochs=10)
model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def document_embedding_w2v(doc, model):
    embeddings = [model.wv[word] for word in doc if word in model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

document_embeddings_w2v = [document_embedding_w2v(doc, model_w2v) for doc in documents_lemmatized]
similarities_w2v = cosine_similarity(document_embeddings_w2v)

def get_recommendations(query_similarities, threshold=0.2):
    similar_docs_indices = np.argsort(query_similarities[0])[::-1]
    similar_docs_indices = similar_docs_indices[query_similarities[0][similar_docs_indices] > threshold]
    return similar_docs_indices

def search_documents_w2v(query_text, threshold=0.2):
    query_cleaned = clean_text(query_text)
    query_tokenized = word_tokenize(query_cleaned)
    query_filtered = [word for word in query_tokenized if word not in stop_words]
    query_lemmatized = lemmatize_text(query_filtered)
    query_embedding = document_embedding_w2v(query_lemmatized, model_w2v)
    query_similarities = cosine_similarity([query_embedding], document_embeddings_w2v)
    recommended_docs = get_recommendations(query_similarities, threshold)
    return recommended_docs

def search_documents_sbert(query_text, threshold=0.2):
    query_embedding = model_sbert.encode([query_text])[0]
    document_embeddings_sbert = model_sbert.encode(documents)
    query_similarities = cosine_similarity([query_embedding], document_embeddings_sbert)
    recommended_docs = get_recommendations(query_similarities, threshold)
    return recommended_docs

def recommander_documents(event=None):
    texte_utilisateur = texte_entree.get().strip()
    if not texte_utilisateur:
        messagebox.showwarning("Entrée vide", "Veuillez entrer un mot ou une phrase pour la recherche.")
        return

    model_choice = model_choice_var.get()
    if model_choice == 'Word2Vec':
        recommended_docs_for_query = search_documents_w2v(texte_utilisateur)
    elif model_choice == 'SBERT':
        recommended_docs_for_query = search_documents_sbert(texte_utilisateur)

    if len(recommended_docs_for_query) == 0:
        messagebox.showinfo("Aucun document trouvé", "Aucun document similaire n'a été trouvé.")
    else:
        afficher_documents_similaires(recommended_docs_for_query)


def evaluate_model(true_labels, predicted_labels):
    true_labels_flat = [label for sublist in true_labels for label in sublist]
    predicted_labels_flat = [label for sublist in predicted_labels for label in sublist]

    precision = precision_score(true_labels_flat, predicted_labels_flat, average='binary')
    recall = recall_score(true_labels_flat, predicted_labels_flat, average='binary')
    f1 = f1_score(true_labels_flat, predicted_labels_flat, average='binary')
    accuracy = accuracy_score(true_labels_flat, predicted_labels_flat)

    return precision, recall, f1, accuracy
# Évaluation des deux modèles
metrics_w2v = evaluate_model(true_labels, predicted_labels_w2v)
metrics_sbert = evaluate_model(true_labels, predicted_labels_sbert)
# Comparaison des résultats
results = {
    "Word2Vec": {"Précision": metrics_w2v[0], "Rappel": metrics_w2v[1], "F1-Score": metrics_w2v[2], "Exactitude": metrics_w2v[3]},
    "SBERT": {"Précision": metrics_sbert[0], "Rappel": metrics_sbert[1], "F1-Score": metrics_sbert[2], "Exactitude": metrics_sbert[3]},
}
# Affichage des résultats
print("Résultats Word2Vec :", results["Word2Vec"])
print("Résultats SBERT :", results["SBERT"])
with open('model_evaluation.json', 'w') as f:
    json.dump(results, f, indent=4)


# Visualisation des résultats
models = ['Word2Vec', 'SBERT']
precisions = [metrics_w2v[0], metrics_sbert[0]]
recalls = [metrics_w2v[1], metrics_sbert[1]]
f1_scores = [metrics_w2v[2], metrics_sbert[2]]

x = np.arange(len(models))

plt.bar(x - 0.2, precisions, 0.4, label='Précision')
plt.bar(x + 0.2, recalls, 0.4, label='Rappel')
plt.plot(x, f1_scores, marker='o', label='F1-Score', color='red', linestyle='--')
plt.xticks(x, models)
plt.title('Comparaison des Modèles')
plt.xlabel('Modèles')
plt.ylabel('Score')
plt.legend()
plt.show()


def afficher_documents_similaires(doc_indices):
    # Cacher l'interface principale
    main_frame.pack_forget()

    # Créer un nouveau frame pour afficher les documents similaires
    result_frame = tk.Frame(fenetre)
    result_frame.pack(pady=10)

    bouton_retour = tk.Button(result_frame, text="Retour", font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat", command=lambda: retourner_à_accueil(result_frame))
    bouton_retour.pack(pady=10)

    for index in doc_indices:
        doc_name = document_names[index]
        bouton_document = tk.Button(result_frame, text=doc_name, font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat", width=30, height=2, padx=10, pady=5, command=lambda idx=index: afficher_contenu_document(idx))
        bouton_document.pack(pady=10)

def retourner_à_accueil(result_frame):
    # Fermer le frame des résultats et réafficher l'interface principale
    result_frame.pack_forget()
    main_frame.pack(pady=10)

def afficher_contenu_document(index):
    contenu_fenetre = tk.Toplevel()
    contenu_fenetre.title(f"Contenu du Document : {document_names[index]}")
    texte_contenu = tk.Text(contenu_fenetre, wrap=tk.WORD, width=80, height=20, font=("Roboto", 12))
    texte_contenu.insert(tk.END, documents[index])
    texte_contenu.config(state=tk.DISABLED)
    texte_contenu.pack(padx=10, pady=10)
    bouton_fermer = tk.Button(contenu_fenetre, text="Fermer", font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat", command=contenu_fenetre.destroy)
    bouton_fermer.pack(pady=10)


def afficher_boutons_domaines():
    """Affiche les boutons pour chaque domaine détecté dans corpus2."""
    # Cacher l'interface principale
    main_frame.pack_forget()

    # Créer un nouveau frame pour les boutons de domaines
    domaines_frame = tk.Frame(fenetre)
    domaines_frame.pack(pady=10)

    bouton_retour = tk.Button(domaines_frame, text="Retour", font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat", command=lambda: retourner_à_accueil(domaines_frame))
    bouton_retour.pack(pady=10)

    for domaine in set(domaines):  # Utiliser un set pour éviter les doublons
        bouton_domaine = tk.Button(
            domaines_frame,
            text=domaine,
            font=("Roboto", 12),
            bg="#6C63FF",
            fg="white",
            relief="flat",
            width=30,
            height=2,
            padx=10,
            pady=5,
            command=lambda d=domaine: afficher_documents_par_domaine(d, domaines_frame)
        )
        bouton_domaine.pack(pady=10)

def afficher_documents_par_domaine(domaine, parent_frame):
    """Affiche les documents pour un domaine donné."""
    # Cacher le frame parent
    parent_frame.pack_forget()

    # Créer un nouveau frame pour afficher les documents
    documents_frame = tk.Frame(fenetre)
    documents_frame.pack(pady=10)

    bouton_retour = tk.Button(documents_frame, text="Retour", font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat", command=lambda: retourner_à_accueil(documents_frame))
    bouton_retour.pack(pady=10)

    # Lister les documents pour ce domaine
    for i, doc_domaine in enumerate(domaines):
        if doc_domaine == domaine:
            doc_name = document_names[i]
            bouton_document = tk.Button(
                documents_frame,
                text=doc_name,
                font=("Roboto", 12),
                bg="#6C63FF",
                fg="white",
                relief="flat",
                width=30,
                height=2,
                padx=10,
                pady=5,
                command=lambda idx=i: afficher_contenu_document(idx)
            )
            bouton_document.pack(pady=10)


def choisir_domaine():
    """Affiche une boîte de dialogue pour permettre à l'utilisateur de choisir un domaine (type de document)."""
    domaines_disponibles = ['Domaine1', 'Domaine2', 'Domaine3', 'Domaine4']  # Liste des domaines disponibles
    domaine_selectionne = tk.simpledialog.askstring("Choisir un domaine", "Sélectionnez le domaine du document :\n" + "\n".join(domaines_disponibles))
    
    if domaine_selectionne in domaines_disponibles:
        return domaine_selectionne
    else:
        messagebox.showwarning("Domaine invalide", "Domaine non valide ou non sélectionné.")
        return None
    

def ajouter_document():
    # Ouvrir une boîte de dialogue pour sélectionner un fichier
    file_path = filedialog.askopenfilename(title="Sélectionner un document", filetypes=[("Text Files", "*.txt")])
    
    if file_path:  # Si un fichier est sélectionné
        # Créer une nouvelle fenêtre pour choisir un domaine
        domain_window = tk.Toplevel()
        domain_window.title("Choisir un domaine")
        
        # Créer un menu déroulant avec les domaines existants
        domaine = tk.StringVar()
        domaine.set(domaines[0])  # Définir une valeur par défaut si des domaines existent
        domaine_menu = tk.OptionMenu(domain_window, domaine, *set(domaines))  # set(domaines) pour éviter les doublons
        domaine_menu.pack(pady=10)
        
        # Créer un bouton pour confirmer le choix du domaine
        def confirmer_domaine():
            selected_domaine = domaine.get()
            # Vérifier si le dossier correspondant au domaine existe, sinon le créer
            domaine_dir = os.path.join(corpus_dir, selected_domaine)
            if not os.path.exists(domaine_dir):
                os.makedirs(domaine_dir)
            
            try:
                # Déplacer le fichier dans le dossier approprié
                file_name = os.path.basename(file_path)
                destination = os.path.join(domaine_dir, file_name)
                os.rename(file_path, destination)  # Déplacer le fichier sélectionné
                messagebox.showinfo("Succès", f"Le fichier {file_name} a été ajouté au domaine {selected_domaine}.")
                domain_window.destroy()  # Fermer la fenêtre après ajout
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de l'ajout du document : {e}")

        # Ajouter un bouton pour valider le choix du domaine
        bouton_confirmer = tk.Button(domain_window, text="Confirmer", command=confirmer_domaine, font=("Roboto", 12), bg="#6C63FF", fg="white", relief="flat")
        bouton_confirmer.pack(pady=10)
    else:
        messagebox.showwarning("Fichier manquant", "Aucun fichier n'a été sélectionné.")




# Création de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Moteur de Recherche pour Choisir une Spécialité")

# Couleur de fond plus claire pour toute l'interface
fenetre.configure(bg="#f0f4f7")  # Couleur de fond claire et apaisante

# Ajouter un texte de bienvenue avec une police plus carrée et un bleu nuit pour le texte
texte_bienvenue = tk.Label(fenetre, text="🚀 Tu as ton bac en poche et tu ne sais toujours pas où aller ?! 🎓\n"
                                         "Pas de panique, découvre la spécialité qui va t'éclater ! 🔥",
                           bg="#f0f4f7", font=("Roboto", 20, "bold"), fg="#003366", justify="center", padx=30, pady=10)
texte_bienvenue.pack(pady=30)

# Frame principal
main_frame = tk.Frame(fenetre)
main_frame.pack(pady=10)

model_choice_var = tk.StringVar(value='Word2Vec')
model_choice_frame = tk.Frame(main_frame)
model_choice_frame.pack(pady=10)

label_choice = tk.Label(model_choice_frame, text="Choisissez le modèle de recherche :", font=("Roboto", 14))
label_choice.pack(side=tk.LEFT, padx=10)

radio_w2v = tk.Radiobutton(model_choice_frame, text="Word2Vec", variable=model_choice_var, value='Word2Vec', font=("Roboto", 12))
radio_w2v.pack(side=tk.LEFT)

radio_sbert = tk.Radiobutton(model_choice_frame, text="SBERT", variable=model_choice_var, value='SBERT', font=("Roboto", 12))
radio_sbert.pack(side=tk.LEFT)

texte_entree = tk.Entry(main_frame, font=("Roboto", 14), bg="#f0f0f0", fg="#003366", bd=2, relief="solid", highlightthickness=1, highlightbackground="#ccc")
texte_entree.pack(pady=15)

texte_entree.bind("<Return>", recommander_documents)

bouton_recommander = tk.Button(main_frame, text="Rechercher", command=recommander_documents, font=("Roboto", 14), bg="#6C63FF", fg="white", relief="flat", width=20, height=2, padx=10, pady=5)
bouton_recommander.pack(pady=25)

bouton_domaines = tk.Button(
    main_frame,
    text="Accéder aux catégories",
    command=afficher_boutons_domaines,
    font=("Roboto", 14),
    bg="#6C63FF",
    fg="white",
    relief="flat",
    width=25,
    height=2,
    padx=10,
    pady=5
)
bouton_domaines.pack(pady=10)


bouton_ajouter_document = tk.Button(
    main_frame,
    text="Ajouter un document",
    command=ajouter_document,
    font=("Roboto", 14),
    bg="#6C63FF",
    fg="white",
    relief="flat",
    width=20,
    height=2,
    padx=10,
    pady=5
)
bouton_ajouter_document.pack(pady=10)


fenetre.mainloop()



