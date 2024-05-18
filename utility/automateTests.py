import os
import subprocess

def compile_c_file(file_name):
    """
    Compile a C file using gcc.
    
    Parameters:
        file_name (str): Name of the C file to compile.
        
    Returns:
        bool: True if compilation succeeds, False otherwise.
    """
    try:
        subprocess.check_call(['gcc', '-o', 'executable', file_name])
        return True
    except subprocess.CalledProcessError:
        print("Compilation failed.")
        return False

def run_test_case(input_file):
    """
    Run a test case by providing input from a file to the compiled C program.
    
    Parameters:
        input_file (str): Name of the input file containing test case data.
    """
    try:
        with open(input_file, 'r') as f:
            subprocess.check_call(['./executable'], stdin=f)
    except FileNotFoundError:
        print(f"Input file '{input_file}' not found.")

def main():
    c_file = 'sequentialImplementation.c'

    # Compile the C file
    if compile_c_file(c_file):
        print("Compilation successful.")
    else:
        return

    # Directory containing test case input files
    test_case_directory = 'test_cases/'

    # Get list of input files in the test case directory
    input_files = [file for file in os.listdir(test_case_directory) if file.endswith('.txt')]

    # Run each test case
    for input_file in input_files:
        print(f"Running test case: {input_file}")
        run_test_case(os.path.join(test_case_directory, input_file))

if __name__ == "__main__":
    main()
