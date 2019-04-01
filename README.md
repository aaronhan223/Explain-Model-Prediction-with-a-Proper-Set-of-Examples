# Explain Model Prediction with a Proper Set of Examples
Black Box interpretable machine learning (cifar-10 dataset)

**Steps to run**:

1. Create conda environment via environment.yml
2. Change the writing directory in "run_counterfactual.py", "run_influence_function.py".
3. Run counterfactual experiment via "python run_counterfactual.py".
4. Run influence function experiment via "python run_influence_function.py".

In the first run, the script will automatically download the Inception model and full CIFAR_10 data set (50000 training, 10000 testing), and then randomly select 1000 training and 200 testing data from the first two classes (50% each) to calculate the values for transfer learning and store in the cache.

You can try different target test example by changing the "test_idx" variable in "run_counterfactual.py" and "run_influence_function.py". If you are adding noised test image to the training data, be sure to delete the "data/CIFAR-10/inception_cifar10_noise.pkl" and "./embedding.npz" files before running the program of changed test index.

Our algorithm uses SLSQP method for constrained optimization:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html#scipy.optimize.fmin_slsqp
