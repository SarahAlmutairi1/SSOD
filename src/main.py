import os
import time
import dataset
import train
import auto_labeling
import evaluate
import csv
from config import HOME
from ultralytics import settings
import subprocess
def log_results(num_instances, threshold_val, ScoreBased, ScoreThreshold, processing_time, Train_time, metrics, Labels_quality, save_path, filename="results.csv"):
    """
    Logs experiment results into a CSV file at a specified path.

    Parameters:
    - num_instances: Number of instances used
    - threshold_val: Threshold value used
    - ScoreBased: Score-based method used
    - ScoreThreshold: Score threshold applied
    - processing_time: Processing time in minutes
    - Train_time: Training time in hours
    - metrics: Dictionary containing precision, recall, mAP50, and mAP50-95
    - Labels_quality: Dictionary containing precision, recall, and F1-score for labels
    - save_path: Directory where results.csv should be saved
    - filename: Name of the results file (default: "results.csv")
    """
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Full path to the results file
    file_path = os.path.join(save_path, filename)

    # Check if the file exists to decide whether to write the header
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file is new
        if not file_exists:
            writer.writerow([
                "num_instances", "threshold_val", "ScoreBased", "ScoreThreshold", 
                "processing_time(min)", "Train_Time(hr)", "Precision", "Recall", 
                "mAP50", "mAP50-95", "Labels_quality:P", "Labels_quality:R", "Labels_quality:F1-score"
            ])

        # Append new results
        writer.writerow([
            num_instances, threshold_val, ScoreBased, ScoreThreshold, 
            processing_time, Train_time, metrics["precision"], metrics["recall"], 
            metrics["mAP50"], metrics["mAP50-95"], Labels_quality["precision"], 
            Labels_quality["recall"], Labels_quality["f1_score"]
        ])

    print(f"Results logged in: {file_path}")

def main(iteration ,main_dataset_dir, class_names, img_size, num_instances, epochs_per_iteration, threshold_val, ScoreBased, ScoreThreshold, save_path):
    """
    Main function that drives the iterative auto-labeling process and model training
    with auto-annotated labels.

    Parameters:
    - main_dataset_dir: Directory where the dataset is stored.
    - class_names: List of class names for the dataset.
    - img_size: Image size to be used for training.
    - num_instances: Number of YOLO instances to train in parallel.
    - epochs_per_iteration: Number of epochs to train each instance in each iteration.
    - threshold_val: Threshold value to consider boxes as overlapping.
    - ScoreBased: Flag to indicate if confidence score should be used for predictions.
    - ScoreThreshold: Minimum score threshold for predictions.
    - save_path: Directory where results will be saved.

    Returns:
    - all_final_predictions: Final predictions after the auto-labeling process.
    """

    # Define the number of images each YOLO instance should be trained on per iteration
    num_images_per_instance = int(len(os.listdir(main_dataset_dir + '/train/images')) / num_instances)
    print(f'Number of images per instance: {num_images_per_instance}')

    # Execute the iterative auto-labeling process
    start_time = time.time()

    final_predictions = auto_labeling.iterative_auto_labeling(
        main_dataset_dir, num_images_per_instance, num_instances,
        epochs_per_iteration, img_size,class_names, threshold_val=0.5,
        ScoreBased=True, ScoreThreshold=0.6)

    end_time = time.time()
    print("Auto-labeling process completed.")

    # Calculate the processing_time in minutes
    processing_time = (end_time - start_time) / 60
    print(f"processing_time: {processing_time:.2f} minutes")

    # Define the path for the final auto-annotated dataset
    Final_auto_annotated_dataset = f'{HOME}/Final_auto_annotated_dataset'

    # Train the YOLO model using the auto-annotated labels
    model, Train_time = train.train_final_model(iteration, Final_auto_annotated_dataset, img_size, 1)

    # Evaluate the model
    metrics = evaluate.evaluate_final_model(model, main_dataset_dir, img_size)

    # Combine all final Pseudo-predictions from each instance
    all_final_predictions = {}
    for d in final_predictions:
        all_final_predictions.update(d)

    # Evaluate the predictions generated during the auto-labeling process
    print("Evaluating labels produced by the ETSR model")
    output_folder = "/content/final_pred"
    auto_labeling.save_predictions(all_final_predictions, output_folder, img_size, img_size)
    ground_truth_folder = '/content/datasets/VOC1/valid/labels'  # Folder containing ground truth labels
    Labels_quality = evaluate.evaluate_predictions(output_folder, ground_truth_folder, class_names)

    # Log results
    log_results(num_instances, threshold_val, ScoreBased, ScoreThreshold, processing_time, Train_time,metrics, Labels_quality, save_path)


# Main execution
if __name__ == "__main__":
    
    # Call the main
    main_dataset_dir = f'{HOME}/datasets/VOC1'  # dataset path
    output_path = f'{HOME}/output'  # output folder path

    class_names =[
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    img_size = 640
    epochs_per_iteration = 1
    num_instances_list = [2]
    threshold_values = [0.5]
    score_based_options = [False, True]
    score_thresholds = [0.5]

    # run the experiment
    iteration = 0
    for num_instances in num_instances_list:
        for threshold_val in threshold_values:
            for ScoreBased in score_based_options:
                if ScoreBased:
                  for ScoreThreshold in score_thresholds:
                      print(f"Running with: num_instances={num_instances}, threshold_val={threshold_val}, ScoreBased={ScoreBased}, ScoreThreshold={ScoreThreshold}")
                      main(iteration+1, main_dataset_dir, class_names, img_size, num_instances,epochs_per_iteration, threshold_val, ScoreBased, ScoreThreshold, output_path)
                else:
                  ScoreThreshold = 0    # neglected since scoreBased is False
                  print(f"Running with: num_instances={num_instances}, threshold_val={threshold_val}, ScoreBased={ScoreBased}, ScoreThreshold neglected")
                  main( iteration+1, main_dataset_dir, class_names, img_size, num_instances,epochs_per_iteration, threshold_val, ScoreBased, ScoreThreshold, output_path)
