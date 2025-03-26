import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator
import pennylane as qml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Create directory for saving results
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Quantum Network
class OptimizedQuantumNetwork:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.dev = qml.device("default.qubit", wires=nqubits)
    
    def apply_grover(self):
        @qml.qnode(self.dev)
        def circuit():
            for i in range(self.nqubits):
                qml.Hadamard(wires=i)
            qml.PauliZ(wires=self.nqubits - 1)
            for i in range(self.nqubits):
                qml.Hadamard(wires=i)
                qml.PauliX(wires=i)
            qml.PauliZ(wires=self.nqubits - 1)
            for i in range(self.nqubits):
                qml.PauliX(wires=i)
                qml.Hadamard(wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.nqubits)]
        return circuit()

# Pathfinding Visualization
class NASAPathFinderVisualizer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
    
    def visualize_solution(self, path, filename):
        grid = np.zeros((self.grid_size, self.grid_size))
        for (x, y) in path:
            grid[x, y] = 1
        plt.imshow(grid, cmap='Greys', interpolation='nearest')
        plt.title("Pathfinding Solution")
        plt.savefig(os.path.join(SAVE_DIR, filename))
        plt.close()

# Movement Sequence Generator
def move_sequence(start_x, start_y, steps=10):
    movements = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    path = [(start_x, start_y)]
    for i in range(steps):
        dx, dy = movements[i % len(movements)]
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    return path

# AI/ML Movement Sequence Analysis
class MovementSequenceAnalyzer:
    def __init__(self, sequence):
        self.sequence = sequence
        self.data = np.array(sequence)
    
    def analyze_with_kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        labels = kmeans.fit_predict(self.data)
        return labels, kmeans

    def visualize_clusters(self, labels, filename):
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(self.data)
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis')
        plt.title("KMeans Clustering of Movement Sequence")
        plt.savefig(os.path.join(SAVE_DIR, filename))
        plt.close()

# Auto-save CSV function
def save_to_csv(filename, data, headers):
    with open(os.path.join(SAVE_DIR, filename), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# Main Execution
if __name__ == "__main__":
    # 1. Quantum Network Execution
    quantum_network = OptimizedQuantumNetwork(3)
    results = quantum_network.apply_grover()
    print("Quantum Results:", results)
    save_to_csv("quantum_results.csv", [results], ["Qubit1", "Qubit2", "Qubit3"])
    
    # 2. Pathfinding Visualization
    path_finder = NASAPathFinderVisualizer(4)
    path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    path_finder.visualize_solution(path, "pathfinding.png")
    
    # 3. Generate and Save Movement Sequence
    sequence = move_sequence(0, 0)
    save_to_csv("movement_sequence.csv", sequence, ["X", "Y"])
    
    # 4. KMeans Clustering & Visualization
    analyzer = MovementSequenceAnalyzer(sequence)
    labels, kmeans_model = analyzer.analyze_with_kmeans(n_clusters=2)
    save_to_csv("movement_clusters.csv", list(zip(sequence, labels)), ["X", "Y", "Cluster"])
    analyzer.visualize_clusters(labels, "clusters.png")
    print("All data saved automatically!")
