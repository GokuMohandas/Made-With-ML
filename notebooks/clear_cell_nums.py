from pathlib import Path

import nbformat


def clear_execution_numbers(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            for output in cell["outputs"]:
                if "execution_count" in output:
                    output["execution_count"] = None
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    NOTEBOOK_DIR = Path(__file__).parent
    notebook_fps = list(NOTEBOOK_DIR.glob("**/*.ipynb"))
    for fp in notebook_fps:
        clear_execution_numbers(fp)
