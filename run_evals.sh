
source ./venv/bin/activate
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=my-venv-kernel evals_catalanqa.ipynb
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=my-venv-kernel evals_xquad.ipynb
