# MCP-FS

## Installing 

You need to setup anaconda to run this repo.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

You also need to create a virtual environment.

```bash
conda env create -n MCP-FS
conda env update -f environement.yml
```

To contribute to the project, you can install dev dependencies.

```bash
conda env update -f environment-dev.yml
```
