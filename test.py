from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Create a simple quantum circuit
qc = QuantumCircuit(1)
qc.h(0)
qc.measure_all()

# Run the circuit on the simulator
sim = AerSimulator()
result = sim.run(qc, shots=10000).result()


print(result.get_counts())