import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch

def pixel_distance(outputs, targets):
    total_distance = 0
    k=0
    for output, target in zip(outputs, targets):
        distance = ((target[0] - output[0]) ** 2 + (target[1] - output[1]) ** 2) ** 0.5
        total_distance += distance
        if distance <=7:
            k+=1
    return total_distance / len(outputs),k

def error_calc(outputs, targets):
    # Compute the Euclidean distance between outputs and targets
    d,_ = pixel_distance(outputs, targets)
    
    # Compute the magnitude of the targets vector
    target_magnitude = torch.mean(torch.sqrt(torch.sum(targets ** 2, dim=1)))
    if(target_magnitude<=0):
        print("error in calculating target_magnitude")
        
    # Compute the error for each pair in the batch
    error = 100.0 * ((d / target_magnitude))
    
    # Return the mean error over the batch
    return error.item()

def denormalize(image_tensor, mean=0.5, std=0.5):
    """Denormalizes the image tensor using the provided mean and standard deviation."""
    image_tensor = image_tensor * std + mean
    return image_tensor



def plot_circles_on_image(image_tensor, output, target):
    """
    Plots a 10-pixel circle centering in the locations of output and target on the image.

    Parameters:
    - image: 2D or 3D array representing the image.
    - output: Tuple (x, y) representing the predicted location.
    - target: Tuple (x, y) representing the true location.
    """
    plt.rcParams.update({'font.size': 14})

    image_tensor = denormalize(image_tensor)
    image = image_tensor.numpy().transpose((1, 2, 0))
    # Ensure the image is a 2D or 3D numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be a 2D (grayscale) or 3D (color) array")
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    image = np.clip(image, 0, 1)

    # Show the image
    if image.ndim == 2:  # Grayscale image
        image = np.stack((image,)*3, axis=-1)         # Convert grayscale image to RGB if needed
        ax.imshow(image)
    else:  # Color image
        ax.imshow(image)

    # Plot the circles
    circle_output = plt.Circle(output, 5, color='blue', fill=False, linewidth=2, label='Output')
    circle_target = plt.Circle(target, 5, color='green', fill=False, linewidth=2, label='Target')

    ax.add_patch(circle_output)
    ax.add_patch(circle_target)

    # Add legend
    ax.legend()
  # Remove axis numbers
    ax.axis('off')
    # Show the plot
    plt.show()
    
def plot_circles_on_test_image(image, output, target):
    """
    Plots a 10-pixel circle centering in the locations of output and target on the image.

    Parameters:
    - image: 2D or 3D array representing the image.
    - output: Tuple (x, y) representing the predicted location.
    - target: Tuple (x, y) representing the true location.
    """
    plt.rcParams.update({'font.size': 14})
    # image = np.transpose(image, (1,2,0))

    # image_tensor = denormalize(image_tensor)
    image = image.numpy().transpose((0, 1, 2))
    # Ensure the image is a 2D or 3D numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be a 2D (grayscale) or 3D (color) array")
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    # image = np.clip(image, 0, 1)

    # Show the image
    if image.ndim == 2:  # Grayscale image
        image = np.stack((image,)*3, axis=-1)         # Convert grayscale image to RGB if needed
        ax.imshow(image)
    else:  # Color image
        ax.imshow(image)

    # Plot the circles
    circle_output = plt.Circle(output, 5, color='blue', fill=False, linewidth=2, label='Output')
    circle_target = plt.Circle(target, 5, color='green', fill=False, linewidth=2, label='Target')

    ax.add_patch(circle_output)
    ax.add_patch(circle_target)

    # Add legend
    ax.legend()
  # Remove axis numbers
    ax.axis('off')
    # Show the plot
    plt.show()

def mean_absolute_error(output, target):
    return torch.sum(torch.abs(output - target)), output.numel()

def mean_squared_error(output, target):
    return torch.sum((output - target) ** 2), output.numel()

def root_mean_squared_error(output, target):
    mse_sum, n = mean_squared_error(output, target)
    return torch.sqrt(mse_sum / n), n


def r_squared(output, target):
    ss_res = torch.sum((target - output) ** 2)
    ss_tot = torch.sum((target - torch.mean(target, dim=0)) ** 2)
    return 1 - ss_res / ss_tot, output.numel()



def mean_squared_logarithmic_error(output, target):
    return torch.sum((torch.log1p(output) - torch.log1p(target)) ** 2), output.numel()

def Metrics(metrics_accum, output, target):
    mae_sum, mae_count = mean_absolute_error(output, target)
    mse_sum, mse_count = mean_squared_error(output, target)
    rmse_sum, rmse_count = root_mean_squared_error(output, target)
    r2_sum, r2_count = r_squared(output, target)
    # msle_sum, msle_count = mean_squared_logarithmic_error(output, target)
    
    metrics_accum['MAE'] += mae_sum
    metrics_accum['MSE'] += mse_sum
    metrics_accum['RMSE'] += rmse_sum * rmse_count  # RMSE should be calculated at the end
    metrics_accum['R-squared'] += r2_sum * r2_count
    # metrics_accum['MSLE'] += msle_sum
    metrics_accum['count'] += mae_count

def epoch_metrics(metrics_accum):
    count = metrics_accum['count']
    mae = metrics_accum['MAE'] / count
    mse = metrics_accum['MSE']/ count
    rmse = torch.sqrt(metrics_accum['RMSE'] / count)
    r2 = metrics_accum['R-squared'] / count
    # msle = metrics_accum['MSLE'] / count

    # Print the accumulated sums and counts for debugging
    print(f"##Metrics##:\n"
          f"MAE : {mae}\n"
          f"MSE : {mse}\n"
          f"RMSE : {rmse}\n"
          f"R-squared : {r2}\n"
        #   f"MSLE : {msle}\n"
          f"Count: {count}")

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'R-squared': r2,
        # 'MSLE': msle
    }

# # Example usage
# # Assuming you have an image array, and output and target locations
# image = np.random.rand(100, 100, 3)  # Replace with your actual image
# output = (50, 50)  # Replace with your actual output location
# target = (60, 60)  # Replace with your actual target location

# plot_circles_on_image(image, output, target)
