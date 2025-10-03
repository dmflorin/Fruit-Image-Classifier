import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

ims = 200

def load_images_from_folder(folder, label, image_size=(ims, ims)):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img = img.resize(image_size)
            images.append(np.array(img))
            labels.append(label)
            filenames.append(filename)
        except Exception as e:
            print(f"Eroare la incarcarea imaginii {filename}: {e}")
    return images, labels, filenames

apple_dir = "C:\\Users\\PC\\Desktop\\Stuff\\Fac\\an3\\TIA\\images\\apple fruit"
banana_dir = "C:\\Users\\PC\\Desktop\\Stuff\\Fac\\an3\\TIA\\images\\banana fruit"
orange_dir = "C:\\Users\\PC\\Desktop\\Stuff\\Fac\\an3\\TIA\\images\\orange fruit"

apple_images, apple_labels, apple_filenames = load_images_from_folder(apple_dir, label=0)
banana_images, banana_labels, banana_filenames = load_images_from_folder(banana_dir, label=1)
orange_images, orange_labels, orange_filenames = load_images_from_folder(orange_dir, label=2)

print(f"Apple images: {len(apple_images)}")
print(f"Banana images: {len(banana_images)}")
print(f"Orange images: {len(orange_images)}")


images = np.array(apple_images + banana_images + orange_images)
labels = np.array(apple_labels + banana_labels + orange_labels)
filenames = np.array(apple_filenames + banana_filenames + orange_filenames)

images = images / 255.0

images = images.reshape(images.shape[0], -1)

X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    images, labels, filenames, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training data per class: {np.bincount(y_train)}")
print(f"Test data per class: {np.bincount(y_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_probabilities = nb_model.predict_proba(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_probabilities = knn_model.predict_proba(X_test_scaled)

print()

# Evaluare pe setul de antrenament
nb_train_predictions = nb_model.predict(X_train)
knn_train_predictions = knn_model.predict(X_train_scaled)

nb_train_accuracy = accuracy_score(y_train, nb_train_predictions)
knn_train_accuracy = accuracy_score(y_train, knn_train_predictions)

print("Performanță pe setul de antrenament:")
print(f"Naive Bayes - Acuratețe (Train): {nb_train_accuracy * 100:.2f}%")
print(f"K-Nearest Neighbors - Acuratețe (Train): {knn_train_accuracy * 100:.2f}%")

print()

print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions, target_names=["apple", "banana", "orange"]))

print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, knn_predictions, target_names=["apple", "banana", "orange"]))

nb_accuracy = accuracy_score(y_test, nb_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f"K-Nearest Neighbors Accuracy: {knn_accuracy * 100:.2f}%")

nb_cm = confusion_matrix(y_test, nb_predictions)
knn_cm = confusion_matrix(y_test, knn_predictions)

class_names = ["apple", "banana", "orange"]
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(nb_cm, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(knn_cm, "K-Nearest Neighbors Confusion Matrix")

def save_predictions_with_probabilities(output_folder, X_test, y_test, predictions, probabilities, filenames):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(X_test)):
        img = X_test[i].reshape(ims, ims, 3) * 255 
        img = Image.fromarray(img.astype('uint8'))

        true_label = class_names[y_test[i]]
        predicted_label = class_names[predictions[i]]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"True: {true_label}, Pred: {predicted_label}", fontsize=10)

        prob_text = "\n".join([
            f"{class_names[j]}: {probabilities[i][j] * 100:.2f}%" for j in range(len(class_names))
        ])
        plt.text(
            65, 0, prob_text, fontsize=10, color='black',
            verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7)
        )

        output_filename = f"{filenames[i]}_true_{true_label}_pred_{predicted_label}.png"
        plt.savefig(os.path.join(output_folder, output_filename), bbox_inches='tight')
        plt.close()

save_predictions_with_probabilities("output_predictions_nb", X_test, y_test, nb_predictions, nb_probabilities, filenames_test)
save_predictions_with_probabilities("output_predictions_knn", X_test, y_test, knn_predictions, knn_probabilities, filenames_test)