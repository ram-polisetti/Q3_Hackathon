import time
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

# --- CIRCUIT CONSTRUCTION FUNCTIONS (Unchanged from original) ---

def create_reversible_adder(num_bits, a_val, b_val):
    """
    Creates a reversible quantum adder circuit for two n-bit numbers.
    """
    print(f"\n--- Building {num_bits}-bit Adder Circuit ---")
    print(f"Adding: {a_val} (binary: {bin(a_val)[2:].zfill(num_bits)}) + {b_val} (binary: {bin(b_val)[2:].zfill(num_bits)})")

    a_reg = QuantumRegister(num_bits, 'a')
    b_reg = QuantumRegister(num_bits, 'b')
    sum_reg = QuantumRegister(num_bits, 'sum')
    carry_reg = QuantumRegister(num_bits, 'carry')
    ancilla_fa = QuantumRegister(1, 'ancilla_fa')
    qc = QuantumCircuit(a_reg, b_reg, sum_reg, carry_reg, ancilla_fa)

    for i in range(num_bits):
        if (a_val >> i) & 1:
            qc.x(a_reg[i])
    for i in range(num_bits):
        if (b_val >> i) & 1:
            qc.x(b_reg[i])
    qc.barrier(label='Inputs Encoded')

    qc.cx(a_reg[0], sum_reg[0])
    qc.cx(b_reg[0], sum_reg[0])
    qc.ccx(a_reg[0], b_reg[0], carry_reg[0])
    qc.barrier(label='LSB Added')

    for i in range(1, num_bits):
        qc.cx(a_reg[i], sum_reg[i])
        qc.cx(b_reg[i], sum_reg[i])
        qc.cx(carry_reg[i - 1], sum_reg[i])
        qc.cx(a_reg[i], ancilla_fa[0])
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.ccx(carry_reg[i - 1], ancilla_fa[0], carry_reg[i])
        qc.ccx(a_reg[i], b_reg[i], carry_reg[i])
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.cx(a_reg[i], ancilla_fa[0])
    qc.barrier(label='Addition Complete')

    return qc, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa

def uncompute_reversible_adder(qc, num_bits, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa):
    """
    Applies the uncomputation (reverse pass) for the reversible adder circuit.
    """
    print("--- Appending Uncomputation Gates ---")
    qc.barrier(label='Start Uncomputation')
    for i in range(num_bits - 1, 0, -1):
        qc.cx(a_reg[i], ancilla_fa[0])
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.ccx(a_reg[i], b_reg[i], carry_reg[i])
        qc.ccx(carry_reg[i - 1], ancilla_fa[0], carry_reg[i])
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.cx(a_reg[i], ancilla_fa[0])
        qc.cx(carry_reg[i - 1], sum_reg[i])
        qc.cx(b_reg[i], sum_reg[i])
        qc.cx(a_reg[i], sum_reg[i])

    qc.ccx(a_reg[0], b_reg[0], carry_reg[0])
    qc.cx(b_reg[0], sum_reg[0])
    qc.cx(a_reg[0], sum_reg[0])
    qc.barrier(label='Full Uncomputation Complete')

# --- IBMQ CONNECTION AND BACKEND SELECTION ---

def connect_and_select_backend(required_qubits):
    """
    Connects to IBM Quantum and selects the best available backend.
    """
    token_file = "ibmapi.txt"
    if not os.path.exists(token_file):
        print(f"❌ Error: API token file '{token_file}' not found.")
        print("Please create this file and place your IBM Quantum API token in it.")
        return None

    with open(token_file, "r") as f:
        token = f.read().strip()

    try:
        print("\n--- Connecting to IBM Quantum ---")

        QiskitRuntimeService.save_account(instance="Practice",  set_as_default = True,token=token, overwrite=True)
        service = QiskitRuntimeService()
        print("✅ Successfully connected to IBM Quantum.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

    print(f"\n--- Searching for a backend with at least {required_qubits} qubits ---")
    backends = service.backends(simulator=False, operational=True, min_num_qubits=required_qubits)

    if not backends:
        print("⚠️ No suitable real quantum hardware found. Using a local simulator instead.")
        return Aer.get_backend('aer_simulator')

    # Sort by the number of pending jobs to find the least busy machine
    sorted_backends = sorted(backends, key=lambda b: b.status().pending_jobs)
    best_backend = sorted_backends[0]
    print(f"✅ Found best available backend: {best_backend.name} "
          f"(Qubits: {best_backend.num_qubits}, Pending Jobs: {best_backend.status().pending_jobs})")

    return best_backend

# --- UNIFIED EXECUTION AND ANALYSIS FUNCTION ---

def run_on_backend_and_analyze(qc, backend, num_bits, a_val, b_val, purpose):
    """
    Transpiles, runs the circuit on the selected backend, and analyzes the results.
    """
    print(f"\n--- Running Job: {purpose} on {backend.name} ---")

    # Add classical registers for measurement right before execution
    qc.add_register(ClassicalRegister(num_bits, 'a_meas'))
    qc.add_register(ClassicalRegister(num_bits, 'b_meas'))
    qc.add_register(ClassicalRegister(num_bits, 'sum_meas'))
    qc.add_register(ClassicalRegister(num_bits, 'carry_meas'))
    qc.add_register(ClassicalRegister(1, 'ancilla_meas'))
    qc.measure(qc.qregs[0], qc.cregs[0]) # a -> a_meas
    qc.measure(qc.qregs[1], qc.cregs[1]) # b -> b_meas
    qc.measure(qc.qregs[2], qc.cregs[2]) # sum -> sum_meas
    qc.measure(qc.qregs[3], qc.cregs[3]) # carry -> carry_meas
    qc.measure(qc.qregs[4], qc.cregs[4]) # ancilla -> ancilla_meas

    # Transpile the circuit for the specific backend
    print("Transpiling circuit...")
    transpiled_qc = transpile(qc, backend)

    # Use Session for better performance on real hardware
    with Session(backend=backend) as session:
        sampler = Sampler()
        job = sampler.run([transpiled_qc], shots=1024)
        print(f"Job submitted with ID: {job.job_id()}")

        # Monitor the job status
        while not job.done():
            print(f"Job status: {job.status()}. Waiting...")
            time.sleep(5) # Wait 5 seconds before checking again

        print(f"Job finished with status: {job.status()}")
        result = job.result()
        data = result[0].data

    # --- Analysis ---
    print(f"\n--- Analyzing Results for: {purpose} ---")

    # Get counts and find the most frequent outcome for each register
    most_frequent = {
        'a': max(data.a_meas.get_counts(), key=data.a_meas.get_counts().get),
        'b': max(data.b_meas.get_counts(), key=data.b_meas.get_counts().get),
        'sum': max(data.sum_meas.get_counts(), key=data.sum_meas.get_counts().get),
        'carry': max(data.carry_meas.get_counts(), key=data.carry_meas.get_counts().get),
        'ancilla': max(data.ancilla_meas.get_counts(), key=data.ancilla_meas.get_counts().get),
    }

    # Convert binary strings to integers
    measured_vals = {key: int(val, 2) for key, val in most_frequent.items()}

    print("Most Frequent Outcomes:")
    for reg, val in most_frequent.items():
        print(f"  {reg.capitalize()} register: '{val}' (Decimal: {measured_vals[reg]})")

    # --- Verification ---
    print("\nVerification:")
    if purpose == 'Addition':
        expected_sum = a_val + b_val
        final_carry_bit = int(most_frequent['carry'][0], 2) # Get the MSB of the carry register
        total_measured_sum = (final_carry_bit << num_bits) + measured_vals['sum']

        print(f"  Expected A: {a_val}, Measured A: {measured_vals['a']} -> {'✅' if measured_vals['a'] == a_val else '❌'}")
        print(f"  Expected B: {b_val}, Measured B: {measured_vals['b']} -> {'✅' if measured_vals['b'] == b_val else '❌'}")
        print(f"  Expected Total Sum: {expected_sum}, Measured Total Sum: {total_measured_sum} -> {'✅' if total_measured_sum == expected_sum else '❌'}")

    elif purpose == 'Uncomputation':
        a_ok = measured_vals['a'] == a_val
        b_ok = measured_vals['b'] == b_val
        sum_ok = measured_vals['sum'] == 0
        carry_ok = measured_vals['carry'] == 0
        ancilla_ok = measured_vals['ancilla'] == 0

        print(f"  Input A Recovered: {a_ok} {'✅' if a_ok else '❌'}")
        print(f"  Input B Recovered: {b_ok} {'✅' if b_ok else '❌'}")
        print(f"  Sum Register Cleared: {sum_ok} {'✅' if sum_ok else '❌'}")
        print(f"  Carry Register Cleared: {carry_ok} {'✅' if carry_ok else '❌'}")
        print(f"  Ancilla Cleared: {ancilla_ok} {'✅' if ancilla_ok else '❌'}")

        if all([a_ok, b_ok, sum_ok, carry_ok, ancilla_ok]):
            print("\n🎉 Congratulations! Full reversibility successfully demonstrated!")
        else:
            print("\n❌ Reversibility failed. This is expected due to noise on real quantum hardware.")

# --- MAIN EXECUTION LOGIC ---

def main():
    """Main function to configure and run the quantum adder."""
    try:
        print("--- Quantum Reversible Adder on Real Hardware ---")
        print("Note: Jobs on real quantum computers can take several minutes to run.")
        print("For best results, use a small number of bits (e.g., 2) due to hardware noise.")

        num_bits = int(input("Enter the number of bits for the adder (e.g., 2): "))
        max_val = 2**num_bits - 1
        a_val = int(input(f"Enter the first decimal value (0 to {max_val}): "))
        b_val = int(input(f"Enter the second decimal value (0 to {max_val}): "))

        if not (0 <= a_val <= max_val and 0 <= b_val <= max_val):
            print(f"Error: Input values must be between 0 and {max_val}.")
            return

        required_qubits = 4 * num_bits + 1

        # Connect and select the backend
        backend = connect_and_select_backend(required_qubits)
        if backend is None:
            return # Exit if connection failed

        # 1. FORWARD COMPUTATION: Run the adder circuit
        qc_adder, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa = create_reversible_adder(num_bits, a_val, b_val)
        run_on_backend_and_analyze(qc_adder, backend, num_bits, a_val, b_val, 'Addition')

        # 2. REVERSE COMPUTATION: Run the full uncomputation circuit
        qc_uncompute, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa = create_reversible_adder(num_bits, a_val, b_val)
        uncompute_reversible_adder(qc_uncompute, num_bits, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa)
        run_on_backend_and_analyze(qc_uncompute, backend, num_bits, a_val, b_val, 'Uncomputation')

    except ValueError:
        print("Invalid input. Please enter valid integers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()