import nbformat as nbf
import os
import argparse
import pathlib


def create_header(header_name: str, n_markdown_cells: int, n_code_cells: int, nb: nbf.NotebookNode) -> nbf.NotebookNode:
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {header_name}"))
    for i in range(n_markdown_cells):
        nb.cells.append(nbf.v4.new_markdown_cell())
    for i in range(n_code_cells):
        nb.cells.append(nbf.v4.new_code_cell())
    return nb


def create_section(curr_cell: int, n_markdown_cells: int, n_code_cells: int, nb: nbf.NotebookNode) -> nbf.NotebookNode:
    nb.cells.append(nbf.v4.new_markdown_cell(f"## {curr_cell}."))
    for i in range(n_markdown_cells):
        nb.cells.append(nbf.v4.new_markdown_cell())
    for i in range(n_code_cells):
        nb.cells.append(nbf.v4.new_code_cell())
    return nb


def create_template_project(n_sections: int, header_name: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb = create_header(header_name, 1, 1, nb)
    for i in range(1, n_sections + 1):
        nb = create_section(i, 1, 1, nb)
    return nb


def create_template_lesson(n_sections: int, header_name: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb = create_header(header_name, 0, 1, nb)
    for i in range(1, n_sections + 1):
        nb = create_section(i, 1, 2, nb)
    return nb


def create_file(path: pathlib.Path, nb: nbf.NotebookNode) -> None:
    if path.suffix != ".ipynb":
        path = path.with_suffix(".ipynb")
    
    with open(path, "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Jupyter Notebook Templates")
    parser.add_argument("-n", type=int, help="Number of sections", required=True)
    parser.add_argument("-o", type=str, help="Output Path + Filename", required=True)
    args = parser.parse_args()
    
    path = pathlib.Path(args.o)  
    splitted_filename = path.stem.split("_")
    file_type = splitted_filename[0].lower()
    header_name = " ".join(splitted_filename[1:]).title()
    
    if file_type == "project":
        nb = create_template_project(args.n, header_name)
        create_file(path, nb)
    elif file_type == "lesson":
        nb = create_template_lesson(args.n, header_name)
        create_file(path, nb)
    else:
        raise ValueError(f"Notebook type must be either 'Project' or 'Lesson', you entered: {file_type}")