import ast
import json
from pathlib import Path
import nbformat

root = Path(__file__).parent
script = (root / 'recover_smollm2.py').read_text(encoding='utf-8')
ast.parse(script)
notebook = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_markdown_cell('# Recover SmolLM2 Shahnameh model\nEvaluate and export the preserved 51-step checkpoint from the two-T4 training run.'),
    nbformat.v4.new_code_cell("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.56.2', 'accelerate==1.10.1'], check=True)"),
    nbformat.v4.new_code_cell("from pathlib import Path\nscript = " + repr(script) + "\nPath('/kaggle/working/recover_smollm2.py').write_text(script, encoding='utf-8')\nsubprocess.run([sys.executable, '-u', '/kaggle/working/recover_smollm2.py'], check=True)"),
])
notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.validate(notebook)
nbformat.write(notebook, root / 'recover_smollm2.ipynb')
print('Recovery notebook validated')
