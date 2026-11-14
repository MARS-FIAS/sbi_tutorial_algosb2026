"""Build the student notebooks and the solution notebooks from one source each.

    python reference/build_notebooks.py                    # write the four notebooks, without outputs
    python reference/build_notebooks.py --run-solutions    # also execute the two solutions notebooks

The student notebooks ship without outputs. The solutions notebooks ship executed,
so that they can be read on GitHub without running anything.

The sources are plain Python files in reference/notebooks/, in the "percent" cell
format that VS Code and Jupytext also understand:

    # %% [markdown]
    # Text of a Markdown cell, every line behind "# ".

    # %%
    code_of_a_cell()

    # %% tags=["student"]      only in the student notebook (the open challenge)
    # %% tags=["solution"]     only in the solutions notebook (the worked answer)

Two extras: a code line that starts with "#!! " becomes a notebook magic (the Colab
install lines), and {{COLAB_BADGE}} is replaced by the "Open in Colab" button of
the notebook being written.

Edit the sources, not the .ipynb files: the notebooks are overwritten on every build.
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import nbformat

# The repository on GitHub: the Colab buttons and the setup cells point at it. Rebuild after a change.
GITHUB_REPO = "covinolab/sbi_tutorial_algosb2026"
SBI_VERSION = "0.27.0"

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "01_sbi_toy_2d": ROOT / "reference/notebooks/01_sbi_toy_2d.py",
    "02_sbi_double_well": ROOT / "reference/notebooks/02_sbi_double_well.py",
}
CELL_MARKER = re.compile(r"^# %%(?P<markdown> \[markdown\])?(?: tags=\[(?P<tags>[^\]]*)\])?\s*$")


def parse(source):
    """Split a percent-format source into (cell_type, tags, text) triples."""
    cells, current = [], None
    for line in source.splitlines():
        marker = CELL_MARKER.match(line)
        if marker:
            tags = re.findall(r'"([^"]+)"', marker.group("tags") or "")
            current = ["markdown" if marker.group("markdown") else "code", tags, []]
            cells.append(current)
        elif current is not None:
            current[2].append(line)
    return [(kind, tags, _clean(kind, lines)) for kind, tags, lines in cells]


def _clean(kind, lines):
    if kind == "markdown":
        lines = [re.sub(r"^# ?", "", line) for line in lines]
    else:
        lines = [re.sub(r"^(\s*)#!! ", r"\1", line) for line in lines]
    return "\n".join(lines).strip("\n")


def colab_badge(path_in_repo):
    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/main/{path_in_repo}"
    return (f'<a href="{url}" target="_parent"><img '
            'src="https://colab.research.google.com/assets/colab-badge.svg" '
            'alt="Open In Colab"/></a>')


def build(name, variant, write=True):
    """Build one notebook and return (path in the repo, notebook). variant: "student" or "solution"."""
    path_in_repo = f"{name}.ipynb" if variant == "student" else f"solutions/{name}_solutions.ipynb"
    other = "solution" if variant == "student" else "student"
    replacements = {
        "{{COLAB_BADGE}}": colab_badge(path_in_repo),
        "{{GITHUB_REPO}}": GITHUB_REPO,
        "{{REPO_NAME}}": GITHUB_REPO.split("/")[-1],
        "{{SBI_VERSION}}": SBI_VERSION,
    }
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    })
    for index, (kind, tags, text) in enumerate(parse(SOURCES[name].read_text())):
        if other in tags:
            continue
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        make = nbformat.v4.new_markdown_cell if kind == "markdown" else nbformat.v4.new_code_cell
        cell = make(text)
        cell["id"] = f"cell-{index:03d}"  # stable ids keep the diffs small
        notebook.cells.append(cell)
    nbformat.validate(notebook)
    if write:
        nbformat.write(notebook, ROOT / path_in_repo)
    return path_in_repo, notebook


def execute(notebook, working_directory):
    """Run a notebook in place, from top to bottom, on the CPU."""
    from nbclient import NotebookClient

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the kernel inherits this: no GPU, no driver warnings
    NotebookClient(notebook, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(working_directory)}}).execute()
    return notebook


def tidy_outputs(notebook):
    """Make the saved outputs readable: keep only the last state of every progress line."""
    for cell in notebook.cells:
        if "outputs" not in cell:
            continue
        merged = []
        for output in cell["outputs"]:
            same_stream = (output.output_type == "stream" and merged
                           and merged[-1].output_type == "stream" and merged[-1].name == output.name)
            if same_stream:
                merged[-1].text += output.text  # a stream arrives in chunks: put them back together
            else:
                merged.append(output)
        for output in merged:
            if output.output_type == "stream":  # "\r" rewinds the line: what follows overwrites it
                output.text = "\n".join(line.split("\r")[-1] for line in output.text.split("\n"))
        cell["outputs"] = [o for o in merged if not (o.output_type == "stream" and not o.text.strip())]
        cell.get("metadata", {}).pop("execution", None)  # run times would change the file on every build
    return notebook


def cell_seconds(notebook):
    """(cell index, run time in seconds) of every executed code cell."""
    times = []
    for index, cell in enumerate(notebook.cells):
        timing = cell.get("metadata", {}).get("execution", {})
        if "iopub.execute_input" in timing and "shell.execute_reply" in timing:
            start, end = (datetime.fromisoformat(timing[k].replace("Z", "+00:00"))
                          for k in ("iopub.execute_input", "shell.execute_reply"))
            times.append((index, (end - start).total_seconds()))
    return times


if __name__ == "__main__":
    for name in SOURCES:
        for variant in ("student", "solution"):
            path, notebook = build(name, variant)
            if variant == "solution" and "--run-solutions" in sys.argv:
                execute(notebook, ROOT / "solutions")
                slowest = max(seconds for _, seconds in cell_seconds(notebook))
                nbformat.write(tidy_outputs(notebook), ROOT / path)
                print(f"wrote {path} (executed, slowest cell {slowest:.0f} s)")
            else:
                print("wrote", path)
