# Setting up the notebook
- Activate virtual environment nn_env:
    - python -m venv nn_venv
    - nn_venv/Scripts/activate
- Run pip install -r nn_requirements.txt
- Register environment as Jupyter kernel
    - python -m ipykernel install --user --name nn_env --display-name "Python (nn_env)"