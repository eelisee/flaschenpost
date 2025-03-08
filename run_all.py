import subprocess

def run_script(script_name):
    print(f"Running {script_name} ...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error while running {script_name}:")
        print(result.stderr)

# Liste der Skripte in der gewünschten Reihenfolge
scripts = [
    "_8_LinexXGB.py",
    "_9_LightGBM.py",
    "_7_BART.py",
    "_5_BayesRidgeRegression.py"
]

for script in scripts:
    run_script(script)

print("Alle Skripte wurden ausgeführt.")