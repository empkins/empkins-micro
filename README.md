# empkins-micro

A Python package to analyze EmpkinS micro data.


## Description

A longer description of the project goes here... (*will follow soon*)


## Installation

To work with the project you need to install Python >=3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

Then run the commands below to install [poethepoet](`https://poethepoet.natn.io`), get the latest source,
and install the dependencies:

With ssh access:

```bash
git clone git@github.com:empkins/empkins-micro.git
uv tool install poethepoet
uv sync --all-extras --dev
```

With https access:

```bash
git clone https://github.com/empkins/empkins-micro.git
uv tool install poethepoet
uv sync --all-extras --dev
```

All dependencies are specified in the main `pyproject.toml` when running `uv sync`.

### Working with the code

**Note**: In order to use jupyter notebooks with the project you need to register a new IPython 
kernel associated with the venv of the project (`poe register_ipykernel` - see below). 
When creating a notebook, make to sure to select this kernel (top right corner of the notebook).

To run any of the tools required for the development workflow, use the `poe` commands of the 
[poethepoet](https://github.com/nat-n/poethepoet) task runner:

```bash
$ poe
docs                 Build the html docs using Sphinx.
format               Reformat all files using black.
format_check         Check, but not change, formatting using black.
lint                 Lint all files with Prospector.
test                 Run Pytest with coverage.
update_version       Bump the version in pyproject.toml and empkins_micro.__init__ .
register_ipykernel   Register a new IPython kernel named `empkins-micro` linked to the virtual environment.
remove_ipykernel     Remove the associated IPython kernel.
```

**Note**: The `poe` commands are only available if you are in the virtual environment associated with this project. 
You can either activate the virtual environment manually (e.g., `source .venv/bin/activate`) or use the `uv run <task>` 
command to run a task in the virtual environment.

To add new dependencies you need for this repository:
```bash
uv add add <package_name>
```

To update dependencies after the `pyproject.toml` file was changed (It is a good idea to run this after a `git pull`):
```bash
uv sync
```

For more commands see the [official documentation](https://docs.astral.sh/uv/).

### Format and Linting

To ensure consistent code structure this project uses black and ruff to automatically check (and fix) the code format.

```
poe format  # runs ruff format and ruff lint with the autofix flag
poe lint # runs ruff without autofix (will show issues that can not automatically be fixed)
```

If you want to check if all code follows the code guidelines, run `poe ci_check`.
This can be useful in the CI context


### Tests

All tests are located in the `tests` folder and can be executed by using `poe test`.