import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.datasets import make_classification, fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class MachineLearningAI:

    def __init__(self, classifier=None, param_grid=None):
        self.classifier = classifier if classifier is not None else MLPClassifier()
        self.param_grid = param_grid
        self.pipeline = None
        self.grid_search = None
        self.best_params = None
        self.best_score = None

    def generate_dataset(self, n_samples=100, n_features=20, test_size=0.25, random_state=42):
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_classes=2, random_state=random_state)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def load_prebuilt_dataset(self, dataset_name):
        if dataset_name == 'KDDCup99':
            data = fetch_openml('KDDCup99', version=1, as_frame=True)
            X, y = data.data, data.target
        elif dataset_name == 'NSL-KDD':
            url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt'
            column_names = ["feature_" + str(i) for i in range(41)] + ["label"]
            data = pd.read_csv(url, names=column_names)
            X, y = data.iloc[:, :-1], data.iloc[:, -1]
        else:
            raise ValueError(f"Dataset {dataset_name} is not recognized.")

        return train_test_split(X, y, test_size=0.25, random_state=42)

    def configure_pipeline(self, steps):
        self.pipeline = Pipeline(steps)

    def generate_and_train_classifier(self, X_train, y_train, cv=5):
        if self.pipeline is None:
            self.configure_pipeline([
                ('scaler', StandardScaler()), 
                ('feature_selection', SelectKBest(score_func=f_classif, k=10)), 
                ('classifier', self.classifier)
            ])
        if self.param_grid is not None:
            self.grid_search = RandomizedSearchCV(self.pipeline, self.param_grid, cv=cv, n_jobs=-1, n_iter=20)
            self.grid_search.fit(X_train, y_train)
            print(f'Best parameters found: {self.grid_search.best_params_}')
            self.best_params = self.grid_search.best_params_
            self.pipeline = self.grid_search.best_estimator_
        else:
            scores = cross_val_score(self.pipeline, X_train, y_train, cv=cv, n_jobs=-1)
            self.pipeline.fit(X_train, y_train)
            return np.mean(scores)

    def evaluate_classifier(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        return accuracy_score(y_test, y_pred)

    def save_model(self, filename):
        try:
            joblib.dump(self.pipeline, filename)
            print(f'Model saved to {filename}')
        except Exception as e:
            print(f'Error saving model: {e}')

    def load_model(self, filename):
        try:
            self.pipeline = joblib.load(filename)
            print(f'Model loaded from {filename}')
        except Exception as e:
            print(f'Error loading model: {e}')

    def save_data(self, X_train, X_test, y_train, y_test, filename_prefix):
        try:
            train_data = pd.DataFrame(X_train)
            train_data['target'] = y_train
            test_data = pd.DataFrame(X_test)
            test_data['target'] = y_test

            train_data.to_csv(f'{filename_prefix}_train.csv', index=False)
            test_data.to_csv(f'{filename_prefix}_test.csv', index=False)
            print(f'Data saved to {filename_prefix}_train.csv and {filename_prefix}_test.csv')
        except Exception as e:
            print(f'Error saving data: {e}')


class DeepLearningAI(MachineLearningAI):

    def __init__(self, classifier=None, param_grid=None):
        super().__init__(classifier, param_grid)

    def configure_deep_learning_model(self, input_dim):
        self.classifier = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        self.classifier.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train_deep_learning_model(self, X_train, y_train, epochs=500, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])

    def evaluate_deep_learning_model(self, X_test, y_test):
        evaluation = self.classifier.evaluate(X_test, y_test)
        return evaluation


def train_and_save_deep_learning_model():
    ai = DeepLearningAI()
    X_train, X_test, y_train, y_test = ai.generate_dataset()
    ai.configure_deep_learning_model(input_dim=20)
    ai.train_deep_learning_model(X_train, y_train, epochs=500, batch_size=32)
    evaluation = ai.evaluate_deep_learning_model(X_test, y_test)
    print(f'Deep Learning Model Accuracy: {evaluation[1]}')

    model_file = 'deep_learning_model.h5'
    ai.classifier.save(model_file)
    print(f'Model saved to {model_file}')

    ai.save_data(X_train, X_test, y_train, y_test, 'deep_learning')


def chat_with_model(model, X_test):
    print("Chat with the model:")
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("Model Chat: Goodbye! See you later!")
            break

        try:
            user_input_features = process_user_input(user_input)
            model_response = model.predict(user_input_features)
            print("Model Chat:", model_response)
        except Exception as e:
            print("Error processing input:", e)


def process_user_input(user_input):
    try:
        return np.array([float(x) for x in user_input.split()]).reshape(1, -1)
    except ValueError:
        print("Invalid input. Please enter numeric values separated by spaces.")
        return np.zeros((1, 20))  # Return default input to avoid breaking execution


# Execute training and saving
train_and_save_deep_learning_model()

# Load the model