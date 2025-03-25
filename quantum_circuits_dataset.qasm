// BELL_STATE CIRCUIT

// Bell State Circuit
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;


// GROVER_ALGORITHM CIRCUIT

// Grover's Algorithm (3-qubit example)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
h q[2];
x q[0];
x q[1];
x q[2];
h q[2];
cx q[1], q[2];
h q[2];
x q[0];
x q[1];
x q[2];
h q[0];
h q[1];
h q[2];
measure q -> c;


// QUANTUM_FOURIER_TRANSFORM CIRCUIT

// Quantum Fourier Transform (3-qubit)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cp(pi/2) q[1],q[0];
cp(pi/4) q[2],q[0];
h q[1];
cp(pi/2) q[2],q[1];
h q[2];
swap q[0],q[2];
measure q -> c;


// VQE_ANSATZ CIRCUIT

// Variational Quantum Eigensolver (VQE) Ansatz (2-qubit)
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rx(pi/4) q[0];
rx(pi/4) q[1];
cx q[0], q[1];
rx(pi/2) q[0];
rx(pi/2) q[1];
measure q -> c;


