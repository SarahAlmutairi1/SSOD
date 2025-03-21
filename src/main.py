import os
import time
import dataset
import train
import auto_labeling
import evaluate

def main(main_dataset_dir, class_names, img_size, num_instances, epochs_per_iteration, threshold_val, ScoreBased, ScoreThreshold):
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

    Returns:
    - all_final_predictions: Final predictions after the auto-labeling process.
    """
    global HOME
    HOME = os.getcwd()  # Get the current working directory
    print(f"Home Directory: {HOME}")

    # Define the number of images each YOLO instance should be trained on per iteration
    num_images_per_instance = int(len(os.listdir(main_dataset_dir + '/train/images')) / num_instances)
    print(f'Number of images per instance: {num_images_per_instance}')

    # Execute the iterative auto-labeling process
    start_time = time.time()
    final_predictions = auto_labeling.iterative_auto_labeling(
        main_dataset_dir, num_images_per_instance, num_instances,
        epochs_per_iteration, img_size, class_names,threshold_val=0.5,
        ScoreBased=True, ScoreThreshold=0.6
    )
    end_time = time.time()
    runtime = (end_time - start_time) / 60  # Calculate the runtime in minutes
    print(f"Runtime: {runtime:.2f} minutes")
    print("Auto-labeling process completed.")

    # Combine all final predictions from each instance
    all_final_predictions = {}
    for d in final_predictions:
        all_final_predictions.update(d)

    return all_final_predictions

# Main execution
if __name__ == "__main__":
    # Call the main function with appropriate parameters
    main_dataset_dir = 'dataset/VOC1'  # Replace with the actual dataset path
    class_names = ['class1', 'class2', 'class3']  # Replace with actual class names
    img_size = 640  # Example image size
    num_instances = 5  # Example number of instances
    epochs_per_iteration = 10  # Example number of epochs
    threshold_val = 0.5  # Example threshold value for overlapping boxes
    ScoreBased = True  # Whether to use confidence scores for predictions
    ScoreThreshold = 0.6  # Minimum score threshold for predictions

    # Get all final predictions after the auto-labeling process
    all_final_predictions = main(
        main_dataset_dir, class_names, img_size, num_instances,
        epochs_per_iteration, threshold_val, ScoreBased, ScoreThreshold
    )

    # Define the path for the final auto-annotated dataset
    Final_auto_annotated_dataset = f'{HOME}/Final_auto_annotated_dataset'

    # Train the YOLO model using the auto-annotated labels
    model = train.train_final_model(Final_auto_annotated_dataset, img_size, 200)

    # Evaluate the final predictions generated during the auto-labeling process
    print("Evaluating final predictions...")
    output_folder = "/content/final_pred"
    train.save_predictions(all_final_predictions, output_folder, img_size, img_size)
    ground_truth_folder = '/content/datasets/VOC1/valid/labels'  # Folder containing ground truth labels
    evaluate.evaluate_predictions(output_folder, ground_truth_folder, class_names)

    # Evaluate the model trained using the auto-annotated labels
    evaluate.evaluate_final_model(model, main_dataset_dir, img_size)
