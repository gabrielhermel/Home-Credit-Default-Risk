import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize_scalar


def business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    """
    Calculate the business cost based on the number of false negatives (fn)
    and false positives (fp) in the confusion matrix.

    Parameters:
    y_true (array-like): Ground truth labels
    y_pred (array-like): Predicted labels
    fn_cost (int, optional): The cost associated with false negatives
    fp_cost (int, optional): The cost associated with false positives

    Returns:
    int: The total business cost
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Extract the true negatives (tn), false positives (fp),
    # false negatives (fn), and true positives (tp) from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    # Calculate the cost based on the given costs for false negatives and false positives
    cost = fn_cost * fn + fp_cost * fp

    return cost


def threshold_optimization(y_true, y_proba, fn_cost=10, fp_cost=1):
    """
    Find the optimal threshold for classifying predicted probabilities into binary
    classes (0 or 1) that minimizes the business cost associated with false negatives
    and false positives.

    Parameters:
    y_true (array-like): Ground truth labels
    y_proba (array-like): Predicted probabilities
    fn_cost (int, optional): The cost associated with false negatives
    fp_cost (int, optional): The cost associated with false positives

    Returns:
    float: The optimal threshold for minimizing the business cost
    """

    # Define a cost_function that computes the business cost for a given threshold
    def cost_function(threshold):
        # Convert the predicted probabilities into binary class predictions
        # based on the current threshold
        y_pred = np.where(y_proba >= threshold, 1, 0)
        # Calculate the business cost using the business_cost function
        return business_cost(y_true, y_pred, fn_cost, fp_cost)

    # Use minimize_scalar from scipy.optimize to find the threshold value
    # that minimizes the cost_function
    result = minimize_scalar(cost_function, bounds=(0, 1), method="bounded")

    return result.x


def business_cost_metric(y_true, y_proba, fn_cost=10, fp_cost=1):
    """
    Calculate the normalized business cost metric using the optimal threshold
    for classifying predicted probabilities into binary classes (0 or 1) that
    minimizes the business cost associated with false negatives and false positives.
    The cost is inverted so higher values indicate better performance

    Parameters:
    y_true (array-like): Ground truth labels
    y_proba (array-like): Predicted probabilities
    fn_cost (int, optional): The cost associated with false negatives
    fp_cost (int, optional): The cost associated with false positives

    Returns:
    float: The normalized business cost metric
    """
    # Find the optimal threshold using the threshold_optimization function
    optimal_threshold = threshold_optimization(y_true, y_proba, fn_cost, fp_cost)
    # Convert the predicted probabilities into binary class predictions
    # based on the optimal threshold
    y_pred = np.where(y_proba >= optimal_threshold, 1, 0)
    # Calculate the business cost using the business_cost function
    cost = business_cost(y_true, y_pred, fn_cost, fp_cost)
    # Normalize the cost by dividing by the total number of predictions
    normalized_cost = cost / len(y_true)
    # The cost is inverted so higher values indicate better performance, to be more
    # intuitively interpreted when compared to roc_auc and accuracy
    inverted_cost = 1 - normalized_cost

    return inverted_cost
