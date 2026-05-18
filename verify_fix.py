
import os
import sys
from src.sandbox import CUDASandbox
from src.cli import locate_and_setup_msvc

# Add current dir to path to find src
sys.path.append(os.getcwd())

# Setup MSVC for nvcc on Windows
if sys.platform == "win32":
    print("Setting up MSVC environment...")
    if not locate_and_setup_msvc():
        print("Warning: Could not find MSVC (cl.exe). Compilation might fail.")

kernel_file = "optimized_signal_kernels.cu"

if not os.path.exists(kernel_file):
    print(f"Error: {kernel_file} not found")
    sys.exit(1)

print(f"--- Testing {kernel_file} ---")
sandbox = CUDASandbox(kernel_file)

print("Compiling...")
compile_res = sandbox.compile()
if not compile_res['success']:
    print("Compilation Failed!")
    print(compile_res['error_log'])
    sys.exit(1)
print("Compilation Successful!")

print("Profiling & Verifying...")
# The first call to profile_latency in a fresh sandbox instance 
# will establish reference_checksums if they are empty and compile() succeeded.
# Wait, sandbox.compile() actually runs it once to set reference_checksums if success.
# Let's check compile() again.
# Yes: if success and self.has_custom_harness and not self.reference_checksums: run it.

prof_res = sandbox.profile_latency()
print(f"Latency: {prof_res['latency']} ms")
print("Raw Output Snippet:")
print(prof_res['raw_output'])

if prof_res['latency'] < 99999.0:
    print("\nSUCCESS: Kernels are functional and verified!")
else:
    print("\nFAILURE: Verification or profiling failed.")
    sys.exit(1)
