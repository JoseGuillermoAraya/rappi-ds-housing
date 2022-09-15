This repository was made with Poetry to manage dependencies, here https://python-poetry.org/docs/#installation are detailed instructions on how to install it.
Once installed run 'poetry install' to install all needed dependencies.
The bulk of the analysis is made in './notebooks/housing.ipynb', the final estimator (with the full pipeline) can be found in the last cell of the notebook as fp, or it can be obtained with housing.final_pipeline.get_final_estimator()
