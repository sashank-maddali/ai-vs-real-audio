import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load embeddings
real_embeds = np.load("/home/sashank/Desktop/project/wav_2vec/embeds_attention/real_embeddings.npy")
ai_embeds = np.load("/home/sashank/Desktop/project/wav_2vec/embeds_attention/ai_embeddings.npy")    # shape (N_ai, d)

# Create labels: 0 = real, 1 = AI
X = np.vstack([real_embeds, ai_embeds])
y = np.array([0] * len(real_embeds) + [1] * len(ai_embeds))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

