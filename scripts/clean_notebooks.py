"""Keep notebook code in Git and preserve executed copies under local reports/."""
from pathlib import Path
import shutil

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def clean_notebooks():
    """Save each notebook with outputs before clearing its source copy."""
    for folder, group in ((ROOT / "notebooks", "analysis"),
                          (ROOT / "poster/notebooks", "poster")):
        for path in sorted(folder.glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            has_outputs = any(cell.get("outputs") or cell.get("execution_count") is not None
                              for cell in notebook.cells if cell.cell_type == "code")
            if not has_outputs and "widgets" not in notebook.metadata:
                continue
            snapshot = ROOT / "reports/notebooks" / group / path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
                    cell.metadata.pop("execution", None)
            notebook.metadata.pop("widgets", None)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"Saved outputs: {snapshot.relative_to(ROOT)}; cleared {path.relative_to(ROOT)}")


if __name__ == "__main__":
    clean_notebooks()
