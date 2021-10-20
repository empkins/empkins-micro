# empkins-micro

A Python package to analyze EmpkinS Micro data.


## Description

A longer description of the project goes here... (*will follow soon*)


## Installation
In this package we use [poetry](https://python-poetry.org/) to manage dependencies and packaging. In order to use this project for development, first install poetry, then run the following commands to get the latest source, initialize a virtual env and install all development dependencies:

With ssh access:
```bash
git clone git@mad-srv.informatik.uni-erlangen.de:empkins/empkins-micro.git
cd empkins-micro
poetry install
```

With https access:
```bash
git clone https://mad-srv.informatik.uni-erlangen.de/empkins/empkins-micro.git
cd empkins-micro
poetry install
```


## Usage

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