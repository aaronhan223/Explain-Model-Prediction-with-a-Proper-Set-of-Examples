# Explain Model Prediction with a Proper Set of Examples
Black Box interpretable machine learning

Steps to run:

1. Create conda environment via environment.yml
2. Change the writing directory in "run_counterfactual.py", "run_influence_function.py".
3. Run counterfactual experiment via "python run_counterfactual.py".
4. Run influence function experiment via "python run_influence_function.py".

In the first run, the script will automatically download full CIFAR_10 data set (50000 training, 10000 testing), and randomly select 1000 training and 200 testing data from the first two class (50% each) to calculate the values for transfer learning and store in the cache.

It uses SLSQP method for constrained optimization:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html#scipy.optimize.fmin_slsqp