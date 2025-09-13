from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
from qiskit_ibm_runtime import Sampler, Session
import time

def create_reversible_adder(num_bits, a_val, b_val):
    """
    Creates a reversible quantum adder circuit for two n-bit numbers.

    This function builds the circuit using a half-adder for the least
    significant bit (LSB) and a series of full-adders for the remaining bits,
    which is a standard approach for ripple-carry adders. The ancilla qubit
    is uncomputed after each full-adder stage to return it to the |0> state,
    ensuring the overall circuit is reversible.

    Args:
        num_bits (int): The number of bits for the adder.
        a_val (int): The first decimal value to add.
        b_val (int): The second decimal value to add.

    Returns:
        tuple[QuantumCircuit, QuantumRegister, QuantumRegister, QuantumRegister, QuantumRegister, QuantumRegister]:
        A tuple containing the constructed quantum adder circuit and its quantum registers.
    """
    # Print header for clarity
    print(f"--- {num_bits}-bit Reversible Adder: Encoding Inputs ---")
    print(f"Adding: {a_val} (binary: {bin(a_val)[2:].zfill(num_bits)}) + {b_val} (binary: {bin(b_val)[2:].zfill(num_bits)})")

    # Define quantum and classical registers
    a_reg = QuantumRegister(num_bits, 'a')
    b_reg = QuantumRegister(num_bits, 'b')
    sum_reg = QuantumRegister(num_bits, 'sum')
    carry_reg = QuantumRegister(num_bits, 'carry')
    # A single ancilla qubit is sufficient for the entire circuit
    ancilla_fa = QuantumRegister(1, 'ancilla_fa')

    # Create the main quantum circuit (without classical registers for now)
    qc = QuantumCircuit(
        a_reg, b_reg, sum_reg, carry_reg, ancilla_fa
    )

    # Encode the decimal inputs into the quantum registers
    for i in range(num_bits):
        if (a_val >> i) & 1:
            qc.x(a_reg[i])
    print(f"Encoded A: {bin(a_val)[2:].zfill(num_bits)}")

    for i in range(num_bits):
        if (b_val >> i) & 1:
            qc.x(b_reg[i])
    print(f"Encoded B: {bin(b_val)[2:].zfill(num_bits)}")
    qc.barrier(label='Inputs Encoded')

    # Apply a Half-Adder for the LSB (bit 0)
    # Sum_0 = A_0 XOR B_0
    qc.cx(a_reg[0], sum_reg[0])
    qc.cx(b_reg[0], sum_reg[0])
    # Carry_out_0 = A_0 AND B_0
    qc.ccx(a_reg[0], b_reg[0], carry_reg[0])
    qc.barrier(label='LSB Added')

    # Loop for subsequent bits (Full-Adder logic)
    for i in range(1, num_bits):
        # Sum_i = A_i XOR B_i XOR Carry_in_{i-1}
        qc.cx(a_reg[i], sum_reg[i])
        qc.cx(b_reg[i], sum_reg[i])
        qc.cx(carry_reg[i-1], sum_reg[i])

        # Calculate Carry_out_i = (A_i XOR B_i) AND Carry_in_{i-1} OR (A_i AND B_i)
        # Use ancilla to temporary store A_i XOR B_i
        qc.cx(a_reg[i], ancilla_fa[0])
        qc.cx(b_reg[i], ancilla_fa[0])

        # Part 1 of carry-out: (A_i XOR B_i) AND Carry_in_{i-1}
        qc.ccx(carry_reg[i-1], ancilla_fa[0], carry_reg[i])
        # Part 2 of carry-out: A_i AND B_i
        qc.ccx(a_reg[i], b_reg[i], carry_reg[i])

        # Uncompute ancilla_fa[0] to return it to |0> for the next iteration
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.cx(a_reg[i], ancilla_fa[0])

    qc.barrier(label='Addition Complete')

    return qc, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa

def uncompute_reversible_adder(qc, num_bits, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa):
    """
    Applies the uncomputation (reverse pass) for the reversible adder circuit.

    This function reverses every gate from the create_reversible_adder function
    in the exact reverse order. Since CNOT (cx) and Toffoli (ccx) gates are
    their own inverses, we simply re-apply them.

    Args:
        qc (QuantumCircuit): The circuit to uncompute.
        num_bits (int): The number of bits for the adder.
        a_reg (QuantumRegister): The quantum register for 'a'.
        b_reg (QuantumRegister): The quantum register for 'b'.
        sum_reg (QuantumRegister): The quantum register for the sum.
        carry_reg (QuantumRegister): The quantum register for the carry.
        ancilla_fa (QuantumRegister): The quantum register for the ancilla.
    """
    qc.barrier(label='Start Uncomputation')

    # Uncompute the Full-Adder loop by reversing the gate order
    for i in range(num_bits - 1, 0, -1):
        # Reverse the final ancilla cleanup gates
        qc.cx(a_reg[i], ancilla_fa[0])
        qc.cx(b_reg[i], ancilla_fa[0])

        # Reverse the carry computation gates
        qc.ccx(a_reg[i], b_reg[i], carry_reg[i])
        qc.ccx(carry_reg[i-1], ancilla_fa[0], carry_reg[i])

        # Reverse the intermediate ancilla calculation
        qc.cx(b_reg[i], ancilla_fa[0])
        qc.cx(a_reg[i], ancilla_fa[0])

        # Reverse the sum computation gates
        qc.cx(carry_reg[i-1], sum_reg[i])
        qc.cx(b_reg[i], sum_reg[i])
        qc.cx(a_reg[i], sum_reg[i])

    # Reverse the Half-Adder gates
    qc.ccx(a_reg[0], b_reg[0], carry_reg[0])
    qc.cx(b_reg[0], sum_reg[0])
    qc.cx(a_reg[0], sum_reg[0])

    qc.barrier(label='Full Uncomputation Complete')

def simulate_and_analyze(qc, num_bits, a_val, b_val, purpose):
    """
    Simulates the quantum circuit using a local Aer simulator and analyzes the results.

    Args:
        qc (QuantumCircuit): The circuit to simulate.
        num_bits (int): The number of bits used for the adder.
        a_val (int): The first decimal value.
        b_val (int): The second decimal value.
        purpose (str): A string indicating the purpose of the simulation
                       ('Addition' or 'Uncomputation').
    """
    print(f"\n--- {num_bits}-bit Reversible Adder: Simulating {purpose} (Aer Simulator) ---")

    # The expected outcome depends on the purpose of the simulation
    if purpose == 'Addition':
        expected_a = a_val
        expected_b = b_val
        expected_sum = a_val + b_val
        expected_ancilla = 0
    else: # Uncomputation
        expected_a = a_val
        expected_b = b_val
        expected_sum = 0
        expected_ancilla = 0

    # Set up the simulator
    simulator = Aer.get_backend('aer_simulator')

    # Transpile the circuit for the simulator
    transpiled_qc_adder = transpile(qc, simulator)

    # Run the simulation
    job_adder = simulator.run(transpiled_qc_adder, shots=1024)
    result_adder = job_adder.result()
    counts_adder = result_adder.get_counts()

    # Find the most frequent outcome
    most_frequent_adder = max(counts_adder.keys(), key=counts_adder.get)
    print(f"\nMost Frequent Outcome: {most_frequent_adder}")

    # Decode the measured bitstring
    bits_split = most_frequent_adder.split(' ')

    if len(bits_split) != 5:
        print("Error: The number of measured classical registers is not as expected (5).")
        return

    measured_ancilla_fa_str = bits_split[0]
    measured_carry_str = bits_split[1]
    measured_sum_str = bits_split[2]
    measured_b_str = bits_split[3]
    measured_a_str = bits_split[4]

    # Convert binary strings to integers
    measured_a = int(measured_a_str, 2)
    measured_b = int(measured_b_str, 2)
    measured_sum_lower = int(measured_sum_str, 2)

    final_carry_bit_str = measured_carry_str[0] if measured_carry_str else '0'
    final_carry_bit = int(final_carry_bit_str, 2)

    total_measured_sum = (final_carry_bit << num_bits) + measured_sum_lower

    # Print decoded values
    print(f"  Decoded Measured A: {measured_a} (Binary: {measured_a_str})")
    print(f"  Decoded Measured B: {measured_b} (Binary: {measured_b_str})")
    if purpose == 'Addition':
        print(f"  Decoded Measured Sum (Sum Register): {measured_sum_lower} (Binary: {measured_sum_str})")
        print(f"  Total Measured Sum (Decimal): {total_measured_sum}")
        print(f"  Total Measured Sum (Binary, {num_bits+1} bits): {bin(total_measured_sum)[2:].zfill(num_bits + 1)}")
    else: # Uncomputation
        print(f"  Decoded Measured Sum (Sum Register): {measured_sum_lower} (Expected: {expected_sum})")
    print(f"  Decoded Measured Ancilla_FA: {int(measured_ancilla_fa_str, 2)}")

    # Verification
    print(f"\nVerification:")
    print(f"  Inputs Preserved (A): {measured_a == expected_a} {'✅' if measured_a == expected_a else '❌'}")
    print(f"  Inputs Preserved (B): {measured_b == expected_b} {'✅' if measured_b == expected_b else '❌'}")
    if purpose == 'Addition':
        print(f"  Addition Result Correct: {total_measured_sum == expected_sum} {'✅' if total_measured_sum == expected_sum else '❌'}")
    else: # Uncomputation
        print(f"  Sum Register Cleared: {measured_sum_lower == expected_sum} {'✅' if measured_sum_lower == expected_sum else '❌'}")
        print(f"  Carry Register Cleared: {int(measured_carry_str, 2) == 0} {'✅' if int(measured_carry_str, 2) == 0 else '❌'}")
    print(f"  Ancilla Cleared: {int(measured_ancilla_fa_str, 2) == 0} {'✅' if measured_ancilla_fa_str == '0' else '❌'}")

def run_with_sampler_and_analyze(qc, num_bits, a_val, b_val):
    """
    Runs the quantum circuit using Sampler and analyzes the results.

    Args:
        qc (QuantumCircuit): The circuit to simulate.
        num_bits (int): The number of bits used for the adder.
        a_val (int): The first decimal value.
        b_val (int): The second decimal value.
    """
    print(f"\n--- {num_bits}-bit Reversible Adder: Simulating Uncomputation (Sampler) ---")

    # Use the AerSimulator for a local test within a Session
    with Session(backend=Aer.get_backend('aer_simulator')) as session:
        # Sampler is now instantiated with no arguments
        sampler = Sampler()

        # Sampler automatically handles transpilation for the backend in the session
        job = sampler.run([qc], shots=1024)
        print(f"Sampler job ID: {job.job_id()}")

        # The job for a local simulator completes immediately, no need to wait
        print(f"Job completed with status: {job.status()}")
        result = job.result()

        data = result[0].data

        # --- Analyzing the Uncomputation Result (MODIFIED) ---
        print(f"\n--- {num_bits}-bit Reversible Adder: Analyzing Uncomputation Results (Explicit C_out) ---")
        print(f"Original Input A: {a_val} (binary: {bin(a_val)[2:].zfill(num_bits)})")
        print(f"Original Input B: {b_val} (binary: {bin(b_val)[2:].zfill(num_bits)})")

        # Get counts from the individual classical registers
        a_counts = data.a_meas.get_counts()
        b_counts = data.b_meas.get_counts()
        sum_counts = data.sum_meas.get_counts()
        carry_counts = data.carry_meas.get_counts()
        ancilla_counts = data.ancilla_meas.get_counts()

        # Find the most frequent outcome for each register
        most_frequent_a_rev = max(a_counts.keys(), key=a_counts.get)
        most_frequent_b_rev = max(b_counts.keys(), key=b_counts.get)
        most_frequent_sum_rev = max(sum_counts.keys(), key=sum_counts.get)
        most_frequent_carry_rev = max(carry_counts.keys(), key=carry_counts.get)
        most_frequent_ancilla_rev = max(ancilla_counts.keys(), key=ancilla_counts.get)

        # Convert to integers
        measured_a_rev = int(most_frequent_a_rev, 2)
        measured_b_rev = int(most_frequent_b_rev, 2)
        measured_sum_rev = int(most_frequent_sum_rev, 2) # Should be 0
        measured_carry_rev = int(most_frequent_carry_rev, 2) # Should be 0 (C1C0 cleared)
        measured_ancilla_rev = int(most_frequent_ancilla_rev, 2) # Should be 0

        print(f"\nMost Frequent Outcome after Uncomputation:")
        print(f"  A register: '{most_frequent_a_rev}'")
        print(f"  B register: '{most_frequent_b_rev}'")
        print(f"  Sum register: '{most_frequent_sum_rev}'")
        print(f"  Carry register: '{most_frequent_carry_rev}'")
        print(f"  Ancilla register: '{most_frequent_ancilla_rev}'")

        print(f"  Measured A (recovered): {measured_a_rev} (Binary: {most_frequent_a_rev})")
        print(f"  Measured B (recovered): {measured_b_rev} (Binary: {most_frequent_b_rev})")
        print(f"  Measured Sum_reg (cleared): {measured_sum_rev} (Binary: {most_frequent_sum_rev})")
        print(f"  Measured Carry_reg (cleared): {measured_carry_rev} (Binary: {most_frequent_carry_rev})")
        print(f"  Measured Ancilla_FA (cleared): {measured_ancilla_rev}")

        # Final Verification
        print(f"\nFinal Reversibility Verification:")
        print(f"  Inputs A Recovered: {measured_a_rev == a_val} {'✅' if measured_a_rev == a_val else '❌'}")
        print(f"  Inputs B Recovered: {measured_b_rev == b_val} {'✅' if measured_b_rev == b_val else '❌'}")
        print(f"  Sum Register Cleared: {measured_sum_rev == 0} {'✅' if measured_sum_rev == 0 else '❌'}")
        print(f"  Carry Registers Cleared: {measured_carry_rev == 0} {'✅' if measured_carry_rev == 0 else '❌'}")
        print(f"  Ancilla Cleared: {measured_ancilla_rev == 0} {'✅' if measured_ancilla_rev == 0 else '❌'}")

        if (measured_a_rev == a_val and measured_b_rev == b_val and
            measured_sum_rev == 0 and measured_carry_rev == 0 and
            measured_ancilla_rev == 0):
            print("\n🎉 Congratulations! The reversible binary adder successfully demonstrated full reversibility!")
        else:
            print("\n❌ Reversibility demonstration failed. Check circuit logic and uncomputation steps.")

def main():
    """Main function to run the quantum adder simulation."""
    try:
        # --- Configure your inputs here via user input ---
        num_bits = int(input("Enter the number of bits for the adder (e.g., 5): "))
        a_val = int(input(f"Enter the first decimal value (0 to {2**num_bits - 1}): "))
        b_val = int(input(f"Enter the second decimal value (0 to {2**num_bits - 1}): "))

        if a_val < 0 or b_val < 0 or a_val >= 2**num_bits or b_val >= 2**num_bits:
            print(f"Error: Input values must be between 0 and {2**num_bits - 1} for a {num_bits}-bit adder.")
            return

        # 1. FORWARD COMPUTATION: Build the adder circuit
        qc_adder, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa = create_reversible_adder(num_bits, a_val, b_val)

        # Add classical registers and measurement for the forward pass
        qc_adder.add_register(ClassicalRegister(num_bits, 'a_meas'))
        qc_adder.add_register(ClassicalRegister(num_bits, 'b_meas'))
        qc_adder.add_register(ClassicalRegister(num_bits, 'sum_meas'))
        qc_adder.add_register(ClassicalRegister(num_bits, 'carry_meas'))
        qc_adder.add_register(ClassicalRegister(1, 'ancilla_meas'))
        qc_adder.measure(a_reg, qc_adder.cregs[0])
        qc_adder.measure(b_reg, qc_adder.cregs[1])
        qc_adder.measure(sum_reg, qc_adder.cregs[2])
        qc_adder.measure(carry_reg, qc_adder.cregs[3])
        qc_adder.measure(ancilla_fa, qc_adder.cregs[4])

        # Visualize the forward pass
        print("\n--- Visualizing the Forward Pass Circuit (Requires matplotlib) ---")
        try:
            qc_adder.draw(output='mpl', fold=-1, filename='reversible_adder_circuit.png')
            print("Circuit diagram saved to 'reversible_adder_circuit.png'")
        except ImportError:
            print("Matplotlib not found. Skipping circuit visualization.")
            print("You can install it using: pip install matplotlib")

        # Simulate the forward pass and analyze results
        simulate_and_analyze(qc_adder, num_bits, a_val, b_val, 'Addition')

        # 2. REVERSE COMPUTATION: Create a new circuit for uncomputation
        qc_uncompute, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa = create_reversible_adder(num_bits, a_val, b_val)
        uncompute_reversible_adder(qc_uncompute, num_bits, a_reg, b_reg, sum_reg, carry_reg, ancilla_fa)

        # Add classical registers and measurement for the uncomputation pass
        qc_uncompute.add_register(ClassicalRegister(num_bits, 'a_meas'))
        qc_uncompute.add_register(ClassicalRegister(num_bits, 'b_meas'))
        qc_uncompute.add_register(ClassicalRegister(num_bits, 'sum_meas'))
        qc_uncompute.add_register(ClassicalRegister(num_bits, 'carry_meas'))
        qc_uncompute.add_register(ClassicalRegister(1, 'ancilla_meas'))
        qc_uncompute.measure(a_reg, qc_uncompute.cregs[0])
        qc_uncompute.measure(b_reg, qc_uncompute.cregs[1])
        qc_uncompute.measure(sum_reg, qc_uncompute.cregs[2])
        qc_uncompute.measure(carry_reg, qc_uncompute.cregs[3])
        qc_uncompute.measure(ancilla_fa, qc_uncompute.cregs[4])

        # Visualize the full circuit (addition + uncomputation)
        print("\n--- Visualizing the Full Circuit (Addition + Uncomputation) ---")
        try:
            qc_uncompute.draw(output='mpl', fold=-1, filename='full_reversible_circuit.png')
            print("Full circuit diagram saved to 'full_reversible_circuit.png'")
        except ImportError:
            print("Matplotlib not found. Skipping full circuit visualization.")

        # Simulate the full circuit with Sampler and analyze results
        run_with_sampler_and_analyze(qc_uncompute, num_bits, a_val, b_val)

    except ValueError:
        print("Invalid input. Please enter valid integers.")

if __name__ == "__main__":
    main()
