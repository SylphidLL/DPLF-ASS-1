import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
            
            history.append(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}')
        
        return history
    
    def predict_one(self, x):
        return 1 if np.dot(x, self.w) + self.b > 0 else 0
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


if __name__ == "__main__":
    model = Perceptron(learn_rate=0.01, epochs=1000)
    history = model.train(X_train, y_train)
    

    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == y_test)

    with open('results.txt', 'w') as f:
        for line in history:
            f.write(line + '\n')
        f.write(f'Test Accuracy: {test_acc:.4f}')
    
    print('results has saved')
