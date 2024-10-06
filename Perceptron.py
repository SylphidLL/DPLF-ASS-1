import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.style.use('seaborn')

# Load and preprocess data
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

class Perceptron:
    def __init__(self, learn_rate=0.01, epochs=1000):
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def train(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        history = []
        
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict_one(X[i])
                self.w += self.learn_rate * (y[i] - y_pred) * X[i]
                self.b += self.learn_rate * (y[i] - y_pred)
            
            train_loss = np.mean((y - self.predict(X))**2)
            val_acc = np.mean(self.predict(X_val) == y_val)
            
            history.append((epoch+1, train_loss, val_acc))
        
        return history
    
    def predict_one(self, x):
        return 1 if np.dot(x, self.w) + self.b > 0 else 0
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def plot_learning_curves(history):
    epochs, train_loss, val_acc = zip(*history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, train_loss, color='#1f77b4')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    
    ax2.plot(epochs, val_acc, color='#2ca02c')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('learning_curves.pdf')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.pdf')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.pdf')
    plt.close()

def experiment_feature_importance():
    base_model = Perceptron(learn_rate=0.01, epochs=1000)
    base_model.train(X_train, y_train)
    base_acc = np.mean(base_model.predict(X_test) == y_test)
    
    importances = []
    for i in range(X.shape[1]):
        X_test_mod = X_test.copy()
        X_test_mod[:, i] = 0  # Zero out the i-th feature
        acc_without_feature = np.mean(base_model.predict(X_test_mod) == y_test)
        importance = base_acc - acc_without_feature
        importances.append((data.columns[i], importance))
    
    importances.sort(key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 6))
    features, scores = zip(*importances)
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    plt.bar(features, scores, color=colors)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.pdf')
    plt.close()
    
    return importances

if __name__ == "__main__":
    model = Perceptron(learn_rate=0.01, epochs=1000)
    history = model.train(X_train, y_train)
    
    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == y_test)
    
    plot_learning_curves(history)
    plot_confusion_matrix(y_test, test_preds)
    plot_roc_curve(y_test, model.predict(X_test))
    
    feature_importances = experiment_feature_importance()
    
    with open('results.txt', 'w') as f:
        f.write("Experiment Results:\n")
        for epoch, train_loss, val_acc in history:
            f.write(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}\n')
        f.write(f'Test Accuracy: {test_acc:.4f}\n\n')
        
        f.write("Feature Importance Results:\n")
        for feature, importance in feature_importances:
            f.write(f'{feature}: {importance:.4f}\n')
    
    print('Results and visualizations have been saved.')
