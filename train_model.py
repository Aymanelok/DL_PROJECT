import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# ---------------------------------------------------------
# 1. PRÉPARATION DES DONNÉES
# ---------------------------------------------------------
from data_cleaner import prepare_dataset

print("==================================================")
print("1. CHARGEMENT DU DATASET LOCAL (PARQUET) VIA DATA CLEANER")
print("==================================================")

try:
    print("Chargement des fichiers Parquet locaux...")
    train_data = prepare_dataset('train-00000-of-00001.parquet')
    test_data = prepare_dataset('test-00000-of-00001.parquet')
except Exception as e:
    print(f"Erreur lors du chargement des fichiers locaux : {e}")
    exit(1)

train_texts = train_data['clean_text'].tolist()
train_labels = train_data['HS'].tolist()

test_texts = test_data['clean_text'].tolist()
test_labels = test_data['HS'].tolist()


print("==================================================")
print("2. VECTORISATION TF-IDF")
print("==================================================")
# Limitation à 5000 features pour la rapidité
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(train_texts).toarray()
X_test_tfidf = vectorizer.transform(test_texts).toarray()

# Conversion en Tenseurs PyTorch
X_train = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Création des DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 3. DÉFINITION DU MODÈLE DL (PyTorch)
# ---------------------------------------------------------
class HateSpeechMLP(nn.Module):
    def __init__(self, input_dim):
        super(HateSpeechMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.out = nn.Linear(128, 2) # 2 classes: 0 (Neutre/Offensant), 1 (Haineux)
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.out(x)
        return x

input_dim = X_train.shape[1]
model = HateSpeechMLP(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("==================================================")
print("3. ENTRAÎNEMENT DU MODÈLE (10 Epochs)")
print("==================================================")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(train_loader):.4f}")

print("==================================================")
print("4. ÉVALUATION DU MODÈLE")
print("==================================================")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_targets.extend(batch_y.numpy())

acc = accuracy_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)

print(f"Accuracy sur le Test Set : {acc * 100:.2f}%")
print("Matrice de Confusion :")
print(cm)

print("==================================================")
print("5. SAUVEGARDE (Modèle & Vectorizer)")
print("==================================================")
torch.save(model.state_dict(), 'hate_speech_model.pth')
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("[SUCCÈS] Fichiers 'hate_speech_model.pth' et 'tfidf_vectorizer.pkl' générés ! Vous pouvez maintenant lancer main.py.")
