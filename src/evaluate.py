import os
import csv
import numpy as np
import pybboxes as pyb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image
from config import HOME
import json


def evaluate_predictions(predictions_folder, ground_truth_folder, class_names):
    """
    Evaluate predictions by generating a confusion matrix, classification report, accuracy, and mAP.
    """
    y_true, y_pred = [], []

    # Loop through ground truth label files
    for label_file in os.listdir(ground_truth_folder):
        if not label_file.endswith('.txt'):
            continue

        # Load ground truth labels
        with open(os.path.join(ground_truth_folder, label_file), 'r') as f:
            ground_truths = [int(line.split()[0]) for line in f.readlines()]

        # Load predicted labels if they exist
        pred_file_path = os.path.join(predictions_folder, label_file)
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r') as f:
                predictions = [int(line.split()[0]) for line in f.readlines()]

            # Match the lengths by padding with -1 if predictions are fewer, or truncating if more
            if len(predictions) < len(ground_truths):
                predictions.extend([-1] * (len(ground_truths) - len(predictions)))
            elif len(predictions) > len(ground_truths):
                predictions = predictions[:len(ground_truths)]
        else:
            # No detections for these ground truth labels
            predictions = [-1] * len(ground_truths)

        # Extend true and predicted lists with matching pairs
        y_true.extend(ground_truths)
        y_pred.extend(predictions)
    # Now we calculate precision, recall, and AP for each class using scikit-learn
    num_classes = len(class_names)  # The number of unique classes
    precision_dict = {}
    recall_dict = {}
    ap_dict = {}

    # Calculate precision, recall, and AP for each class
    for class_id in range(num_classes):
        # Precision and recall for each class
        precision = precision_score(y_true, y_pred, labels=[class_id], average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=[class_id], average='macro', zero_division=0)
        precision_dict[class_id] = precision
        recall_dict[class_id] = recall

        # Compute Average Precision (AP) for each class
        # Convert y_true and y_pred to binary labels for each class (one-vs-rest)
        y_true_class = (np.array(y_true) == class_id).astype(int)
        y_pred_class = (np.array(y_pred) == class_id).astype(int)

        try:
            ap = average_precision_score(y_true_class, y_pred_class)
        except ValueError:
            ap = 0.0  # If there's an error (e.g., no positive predictions), set AP to 0
        ap_dict[class_id] = ap

    # Calculate mean Average Precision (mAP)
    mean_ap = np.mean(list(ap_dict.values()))

    print("\nAverage Precision (AP) for each class:")
    for class_id, ap in ap_dict.items():
        print(f"{class_names[class_id]}: {ap:.4f}")

    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Calculate precision, recall, F1-score, accuracy
    print("Generating classification report and calculating accuracy")
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0)
    mean_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    mean_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    mean_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Print metrics
    print("Classification Report:\n", report)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
                annot=True,       # Annotate each cell with the numerical value
                fmt='g',           # Format the annotation text
                cmap='Blues',      # Color map for the heatmap
                xticklabels=class_names,  # Set x-axis labels
                yticklabels=class_names,  # Set y-axis labels
                cbar_kws={'label': 'Count'})  # Add a color bar

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.show()

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "f1_score": mean_f1
    }

def save_evaluation_results(iteration, results_text, cm_plot):
    """Save evaluation results and confusion matrix plot."""

    eval_dir = os.path.join(HOME, f"evaluation_results/iteration_{iteration}")
    os.makedirs(eval_dir, exist_ok=True)

    # Save text results
    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(results_text)

    # Save confusion matrix plot
    cm_plot.savefig(os.path.join(eval_dir, "confusion_matrix.png"))
    plt.close(cm_plot)

def evaluate_final_model(model,dataset, img_size):
    # Path to YOLO settings file
    settings_path = "/root/.config/Ultralytics/settings.json"

    # Load the existing settings
    with open(settings_path, "r") as file:
        settings = json.load(file)

    # Update the dataset download directory
    settings["dataset_download_dir"] = "/content/SSOD/datasets"

    # Save the updated settings
    with open(settings_path, "w") as file:
        json.dump(settings, file, indent=4)
        
    model = YOLO(model)
    dataset_path = f"{HOME}/datasets/VOC1"
    result = model.val(data=f'{dataset_path}/data.yaml', split='test')

    # Extract metrics
    res = result.mean_results()
    metrics = {
          "precision": res[0],   # Mean precision over all classes
          "recall": res[1],      # Mean recall over all classes
          "mAP50": res[2],           # mAP at 0.5 IoU
          "mAP50-95": res[3]          # mAP over range 0.5:0.95 IoU
      }
    return metrics
    
def visualize_predictions(image_path, predictions, image_width, image_height):
    """
    Visualize the image with bounding boxes.
    image_path: Path to the image file
    predictions: List of tuples (class_id, [xmin, ymin, xmax, ymax])
    """
    # Open the image
    image = Image.open(image_path)
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)


    # Add bounding boxes to the image
    for class_id, (xmin, ymin, xmax, ymax) in predictions:

        #change format from yolo to orginal
        W, H = image_width, image_height
        YoloBox1 = ( xmin, ymin, xmax, ymax)
        OrginalBox = pyb.convert_bbox(YoloBox1, from_type="yolo", to_type="voc", image_size=(W,H))
        xmin, ymin, xmax, ymax = OrginalBox

        # Create a rectangle patch
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add class ID text
        ax.text(xmin, ymin, f'Class {class_id}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Display the image
    plt.show()
