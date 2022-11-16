# empkins-micro

A Python package to analyze EmpkinS Micro data.


## Description

A longer description of the project goes here... (*will follow soon*)


## Installation

To run the code in this repository first install a compatible version of Python (>= 3.8). 
Then, install [poetry](https://python-poetry.org) which is used to manage dependencies and packaging.

Once you installed poetry, run the following commands to clone the repository, initialize a virtual env and install 
all development dependencies:

With ssh access:

```bash
git clone git@gitlab.rrze.fau.de:empkins/empkins-micro.git
cd empkins_micro
poetry install
```

With https access:

```bash
git clone https://gitlab.rrze.fau.de/empkins/empkins-micro.git
cd empkins_micro
poetry install
```

### Working with the code

**Note**: In order to use the virtual environment associated with this project you need to register a new IPython 
kernel (`poe register_ipykernel` - see below). When creating a notebook, make to sure to select this kernel 
(top right corner of the notebook).

To run any of the tools required for the development workflow, use the `poe` commands of the 
[poethepoet](https://github.com/nat-n/poethepoet) task runner:

```bash
$ poe
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and empkins_io.__init__ .
register_ipykernel      Register a new IPython kernel named `empkins-micro` linked to the virtual environment.
```

**Note**: The `poe` commands are only available if you are in the virtual environment associated with this project. 
You can either activate the virtual environment manually (e.g., `source .venv/bin/activate`) or use the `poetry shell` 
command to spawn a new shell with the virtual environment activated.

To add new dependencies you need for this repository:
```bash
poetry add <package_name>
```

To update dependencies after the `pyproject.toml` file was changed (It is a good idea to run this after a `git pull`):
```bash
poetry update
```

For more commands see the [official documentation](https://python-poetry.org/docs/cli/).



## Running the Experiments

In order to use this package make sure to perform the following steps before:
1. Install the required EmpkinS dependency [`empkins-io`](https://mad-srv.informatik.uni-erlangen.de/empkins/empkins-io).
1. Install this package.
1. Get the dataset you want to perform your validation on (e.g., the [EmpkinS Micro Prestudy Dataset](https://mad-srv.informatik.uni-erlangen.de/MadLab/data/empkins/d03/empkins-micro-prestudy)).
1. Save the dataset somewhere on your local computer.
1. In this project:  
    1. Navigate to `experiments/<name-of-experiment-folder>` and create a file named `config.json`.
    1. Add the following content to `config.json`:
        ```json
        {
            "base_path": "<path-to-dataset>"
        }
        ```
        where the path to the dataset is either provided as relative or absolute path.  
        **WARNING:** This path is specific for your local machine, so don't add this file to git! (by default, it is ignored by `.gitignore`, don't change that).
