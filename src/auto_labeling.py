import os
import sys
import pybboxes as pyb
import numpy as np
import torch
import shutil
from collections import Counter
from config import HOME
import dataset
import preprocess
import train 
import evaluate
import glob
from IPython.display import clear_output
from collections import Counter

def read_predictions_from_file(file_path, image_width, image_height, ScoreBased, ScoreThreshold):
    """
    Read predictions from a text file in YOLO format and return as a list of tuples.
    Each tuple contains (class_id, [xmin, ymin, xmax, ymax], confidence score).
    """
    predictions = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:         # 6 parts including the confidence score in the predictions
                    class_id = int(parts[0])
                    score = float(parts[5])
                    if (ScoreBased == True and score >= ScoreThreshold):  # Filter by score threshold
                        predictions.append((class_id, [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], score))
                    elif (ScoreBased == False):
                        predictions.append((class_id, [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])], score))
                elif len(parts) == 5:  # No confidence score in the actual labels
                    class_id = int(parts[0])
                    predictions.append((class_id, [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]))
                else:
                    print(f"Warning: Skipping malformed line in {file_path}: {line}")
    except FileNotFoundError:
        print(f"File not found, skipping: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return predictions

def read_all_predictions(models_folders, image_width, image_height, ScoreBased, ScoreThreshold):
    """
    Read all predictions from multiple model folders.
    models_folders: List of folder paths for each model
    image_width: Width of the images
    image_height: Height of the images
    Returns a dictionary where keys are image names and values are lists of predictions from each model.
    """
    all_predictions = {}
    try:
        image_files = os.listdir(models_folders[0])
    except FileNotFoundError:
        print(f"Error: Directory not found - {models_folders[0]}")
        return all_predictions
    except Exception as e:
        print(f"Error listing files in {models_folders[0]}: {e}")
        return all_predictions

    for image_file in image_files:
        if image_file.endswith(".txt"):  # Check if the file is a text file
            image_predictions = []
            for model_folder in models_folders:
                file_path = os.path.join(model_folder, image_file)
                if os.path.exists(file_path):
                    predictions = read_predictions_from_file(file_path, image_width, image_height, ScoreBased, ScoreThreshold)
                    image_predictions.extend(predictions)
                else:
                    print(f"No detection file found, skipping: {file_path}")
            if image_predictions:  # Only add if there are any predictions
                print("predictions added !")
                all_predictions[image_file] = image_predictions

    return all_predictions

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is represented by a list [xmin, ymin, xmax, ymax].
    """
    X1a, Y1a, X2a, Y2a = box1
    X1b, Y1b, X2b, Y2b = box2

    W, H = 640, 640
    VocBox1 = pyb.convert_bbox((X1a, Y1a, X2a, Y2a), from_type="yolo", to_type="voc", image_size=(W, H))
    VocBox2 = pyb.convert_bbox((X1b, Y1b, X2b, Y2b), from_type="yolo", to_type="voc", image_size=(W, H))

    intersection_width = max(0, min(VocBox1[2], VocBox2[2]) - max(VocBox1[0], VocBox2[0]))
    intersection_height = max(0, min(VocBox1[3], VocBox2[3]) - max(VocBox1[1], VocBox2[1]))
    intersection_area = intersection_width * intersection_height

    box1_area = (VocBox1[2] - VocBox1[0]) * (VocBox1[3] - VocBox1[1])
    box2_area = (VocBox2[2] - VocBox2[0]) * (VocBox2[3] - VocBox2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def non_max_suppression_with_majority(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) across all boxes regardless of class,
    then use majority voting to assign a final class to each selected box.

    A box is only considered if the number of overlapping boxes is at least
    half of the total number of predicted boxes.

    predictions: List of tuples (class_id, [xmin, ymin, xmax, ymax], score)
    iou_threshold: Threshold for IoU to consider boxes as overlapping.
    """
    # Validate predictions
    predictions = [pred for pred in predictions if isinstance(pred, tuple) and len(pred) == 3 and isinstance(pred[2], (int, float))]

    final_predictions = []
    total_predictions = len(predictions)  # Store total count for filtering

    while predictions:
        # Sort predictions by confidence score (highest first)
        predictions.sort(key=lambda x: x[2], reverse=True)
        best_pred = predictions.pop(0)

        # Group overlapping boxes
        overlapping_boxes = [best_pred]
        remaining_predictions = []

        for pred in predictions:
            iou = compute_iou(best_pred[1], pred[1])
            if iou >= iou_threshold:
                overlapping_boxes.append(pred)
            else:
                remaining_predictions.append(pred)

        # Update the list of remaining predictions
        predictions = remaining_predictions

        # Discard boxes if the group size is one box discard it
        if len(overlapping_boxes) == 1:
            continue  # Skip adding this box to final predictions

        # Majority voting for class assignment
        class_ids = [box[0] for box in overlapping_boxes]
        majority_class = Counter(class_ids).most_common(1)[0][0]

        # Calculate the average bounding box
        boxes = np.array([box[1] for box in overlapping_boxes])
        avg_box = np.mean(boxes, axis=0)

        # Keep the highest confidence score
        final_score = best_pred[2]

        final_predictions.append((majority_class, avg_box, final_score))

    return final_predictions


def process_predictions(models_folders, image_width, image_height, iou_threshold=0.5, ScoreBased = True, ScoreThreshold = 0.6):
    """
    Process predictions from multiple model folders using Non-Maximum Suppression (NMS).
    models_folders: List of folder paths for each model
    image_width: Width of the images
    image_height: Height of the images
    iou_threshold: Threshold for IoU to consider boxes as overlapping
    Returns a dictionary where keys are image names and values are the final aggregated predictions.
    """
    final_predictions = {}
    # Read and aggregate predictions for each image
    all_predictions = read_all_predictions(models_folders, image_width, image_height, ScoreBased, ScoreThreshold)
    
    for image_file, predictions_list in all_predictions.items():
        # Apply NMS to filter out redundant boxes
        aggregated_predictions = non_max_suppression_with_majority(predictions_list, iou_threshold)
        final_predictions[image_file] = aggregated_predictions

    return final_predictions

def save_predictions(predictions, output_folder, image_width, image_height):
    """
    Save predictions to text files in the specified output folder in YOLO format.
    predictions: Dictionary of predictions where keys are image names and values are lists of predictions.
    output_folder: Folder to save the prediction files.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)
    else:
        os.makedirs(output_folder)

    for image_file, model_predictions in predictions.items():
        output_file_path = os.path.join(output_folder, image_file)
        with open(output_file_path, 'w') as f:
            for class_id, bbox, _  in model_predictions:
              bbox_str = ' '.join(map(str, bbox))
              f.write(f"{class_id} {bbox_str}\n")

def iterative_auto_labeling(main_dataset_dir, num_images_per_instance, num_instances, epochs_per_iteration, img_size, class_names ,threshold_val, ScoreBased, ScoreThreshold):
    iteration = 0
    images_per_iteration = num_images_per_instance*num_instances

    # Prepare the datasets folder
    if not os.path.exists(main_dataset_dir):
        print("Error: No dataset folder found!")
        sys.exit(1)

    # Make sure the destination root folder exists
    dest_root_folder = os.path.join(HOME, "distributed-dataset")
    if not os.path.exists(dest_root_folder):
      print(f"Creating destination root folder: {dest_root_folder}")
      os.makedirs(dest_root_folder)

    # distribute dataset in seperate folders for training
    remaining_unlabeled_images, distributed_datasets = preprocess.distribute_dataset(main_dataset_dir, dest_root_folder, num_images_per_instance*0.9, num_images_per_instance*0.1, num_instances, False)

    # Save the initial split to merge it later with the final dataset
    manually_labeled_folder = os.path.join(HOME, "manually_labeled_folder")
    shutil.copytree(distributed_datasets, manually_labeled_folder)

    # Remove the 'unlabeled_data' directory if it exists
    unlabeled_data_path = os.path.join(manually_labeled_folder, 'unlabeled_data')
    if os.path.exists(unlabeled_data_path):
        shutil.rmtree(unlabeled_data_path)

    # Create a folder to merge datasets into
    manually_labeled_folder_merged = os.path.join(HOME, "manually_labeled_folder_merged")
    os.makedirs(manually_labeled_folder_merged, exist_ok=True)

    # Get list of all subfolders in manually_labeled_folder
    folders_in_manually_labeled_folder = [
        os.path.join(manually_labeled_folder, folder_name)
        for folder_name in os.listdir(manually_labeled_folder)
        if os.path.isdir(os.path.join(manually_labeled_folder, folder_name))
    ]

    # Collect the train and val folders to merge
    source_folders_to_merge = []

    for folder in folders_in_manually_labeled_folder:
        # Add train and val folders
        train_folder = os.path.join(folder, 'train')
        val_folder = os.path.join(folder, 'valid')

        if os.path.isdir(train_folder):
            source_folders_to_merge.append(train_folder)
        if os.path.isdir(val_folder):
            source_folders_to_merge.append(val_folder)

    # Merge the datasets from the collected folders
    preprocess.merge_datasets(source_folders_to_merge, manually_labeled_folder_merged)
    shutil.rmtree(manually_labeled_folder)

    print(f'distributed_datasets {distributed_datasets}')

    #move remaining_unlabeled_images to HOME
    if os.path.exists(remaining_unlabeled_images):
      print(f"Moving remaining_unlabeled_images to {HOME}")
      remaining_unlabeled_images = shutil.move(remaining_unlabeled_images, HOME)

    # List all unlabeled images
    unlabeled_images_folder = os.path.join(remaining_unlabeled_images, 'images')
    num_remaining_unlabeled_images = sum(os.path.isfile(os.path.join(unlabeled_images_folder, f)) for f in os.listdir(unlabeled_images_folder))
    print(f"Number of unlabeled images: {num_remaining_unlabeled_images}")

    # Get a list of the distributed datasets directories
    distributed_datasets_folders = [
        os.path.join(distributed_datasets, folder_name)
        for folder_name in os.listdir(distributed_datasets)
        if os.path.isdir(os.path.join(distributed_datasets, folder_name))
        ]

    # prepare the auto annotated folder
    auto_annotated_folders = f'{HOME}/auto_annotated_folders'
    os.makedirs(auto_annotated_folders, exist_ok=True)

    # Initialize the best models list and the final predictions list
    best_models = [None] * num_instances
    final_predictions_list = []

    num_remaining_unlabeled_images = sum(os.path.isfile(os.path.join(unlabeled_images_folder, f)) for f in os.listdir(unlabeled_images_folder))


    # Iterate until there are no more unlabeled images
    #============================================================================================================================================
    while num_remaining_unlabeled_images >= images_per_iteration:
        print(f"Starting iteration {iteration}...")
        print(f"Remaining unlabeled images: {num_remaining_unlabeled_images}")

        # Train each YOLO model using the distributed datasets
        print("Training YOLO models...")
        best_model_paths, model_performance = train.train_multiple_instances(distributed_datasets_folders, epochs_per_iteration, best_models, img_size)
        print(f'best model paths: {best_model_paths}')
        print(f'Training iteration #{iteration}')
        print(f'model_performance: {model_performance}')
        train.check_model_paths(best_model_paths)
        best_models = best_model_paths

        # Check models' performance, if all have mAP >= 0.8, no need to split the data and train the models again.
        # Stop and auto-annotate all remaining unlabeled data
        exit = [False] * num_instances

        for i, performance in enumerate(model_performance):
          if performance >= 0.8:
            print(f'Model no.{i}: performance {performance}')
            exit[i] = True

        # Check if all models meet condition (all have mAP >= 0.9)
        if all(exit):
          print("All models meet condition. EXIT.")
          break

        # Define the unlabeled folder name for the current iteration
        iteration_folder_name = f'unlabeled_dataset_{iteration}'
        iteration_folder_path = os.path.join(HOME, iteration_folder_name)

        unlabeled_dataset = preprocess.move_images(remaining_unlabeled_images, iteration_folder_path, images_per_iteration)
        del iteration_folder_name
        del iteration_folder_path

        # Auto-annotate the next set of unlabeled images
        Pseudo_labels_folders = train.Pseudo_Labeling(best_model_paths, unlabeled_dataset, iteration)
        
        # Process the predictions from the Pseudo Labeling
        final_predictions = process_predictions(Pseudo_labels_folders, image_width = img_size, image_height = img_size, iou_threshold = threshold_val, ScoreBased = ScoreBased, ScoreThreshold = ScoreThreshold )
        final_predictions_list.append(final_predictions)

        # Save final predictions
        output_folder = f"{HOME}/final_labels"
        save_predictions(final_predictions, output_folder, image_width = img_size, image_height = img_size)
        num_images_per_instance = int(len(os.listdir(output_folder))/num_instances)
        print("Final predictions saved.")

        # if the final predicted pseudo labels are less than needed no. of images_per_iteration, stop
        if len(os.listdir(output_folder)) < images_per_iteration :
          iteration += 1
          print("Not enough auto-annotated labels to continue.")
          break

        # Define the auto-annotated folder name for the current iteration
        iteration_auto_annotated_folder_name = f'auto_annotated_{iteration}'
        iteration_auto_annotated_folder_path = os.path.join(HOME, auto_annotated_folders ,iteration_auto_annotated_folder_name)

        #update the unlabeled data
        print("Updating unlabeled data")
        Auto_annotated_folder = preprocess.update_unlabeled_folder(unlabeled_dataset, unlabeled_images_folder, output_folder, iteration_auto_annotated_folder_path)
        num_remaining_unlabeled_images = sum(os.path.isfile(os.path.join(unlabeled_images_folder, f)) for f in os.listdir(unlabeled_images_folder))

        del unlabeled_dataset
        images_folder = os.path.join(Auto_annotated_folder, 'images')
        num_auto_annotated_labels = sum(os.path.isfile(os.path.join(Auto_annotated_folder, f)) for f in os.listdir(Auto_annotated_folder))

        print('enough auto-annotated labels : Prepare data for training')
        #delete the distributed dataset folder and create new one
        print("Deleting distributed dataset folder...")
        shutil.rmtree(distributed_datasets)

        dest_root_folder = os.path.join(HOME, "distributed-dataset")
        if not os.path.exists(dest_root_folder):
          os.makedirs(dest_root_folder)

        #update the distributed folder using the auto-annotated data
        print("Creating distributed folder")
        print(f'num_images_per_instance: {num_images_per_instance}')
        distributed_datasets = preprocess.distribute_dataset(Auto_annotated_folder, dest_root_folder, num_images_per_instance*0.9, num_images_per_instance*0.1, num_instances, True)
        print(f'distributed_datasets {distributed_datasets}')

        # Get a list of the distributed datasets directories
        distributed_datasets_folders = [
            os.path.join(distributed_datasets, folder_name)
            for folder_name in os.listdir(distributed_datasets)
            if os.path.isdir(os.path.join(distributed_datasets, folder_name))
            ]
        print(f"Auto-labeling complete for iteration:{iteration}")
        iteration += 1
        num_remaining_unlabeled_images = sum(os.path.isfile(os.path.join(unlabeled_images_folder, f)) for f in os.listdir(unlabeled_images_folder))
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA


    # after exiting the while loop, annotate the left images
    if num_remaining_unlabeled_images > 0:
      # Auto-annotate the next set of unlabeled images
      Pseudo_labels_folders = train.Pseudo_Labeling(best_model_paths, unlabeled_images_folder, iteration)
        
      # Process the predictions from the Pseudo Labeling
      final_predictions = process_predictions(Pseudo_labels_folders, image_width = img_size, image_height = img_size, iou_threshold = threshold_val, ScoreBased = ScoreBased, ScoreThreshold = ScoreThreshold )

      final_predictions_list.append(final_predictions)
      print(f'number of final predictions is {len(final_predictions)}')

      # Save final predictions
      output_folder = f"{HOME}/final_labels"
      save_predictions(final_predictions, output_folder, image_width = img_size, image_height = img_size)
      print("Final predictions saved.")

      # Define the auto-annotated folder name for the current iteration
      iteration_auto_annotated_folder_name = f'auto_annotated_{iteration}'
      iteration_auto_annotated_folder_path = os.path.join(HOME, auto_annotated_folders, iteration_auto_annotated_folder_name)

      #update the unlabeled data
      print("Updating unlabeled data ...")
      Auto_annotated_folder = preprocess.update_unlabeled_folder( unlabeled_images_folder, unlabeled_images_folder, output_folder, iteration_auto_annotated_folder_path, True)
      num_remaining_unlabeled_images = sum(os.path.isfile(os.path.join(unlabeled_images_folder, f)) for f in os.listdir(unlabeled_images_folder))
      num_auto_annotated_labels = sum(os.path.isfile(os.path.join(Auto_annotated_folder, f)) for f in os.listdir(Auto_annotated_folder))
      print(f'list of auto_annotated labels is {num_auto_annotated_labels}')

      # add the initial split to the auto annotated folders
      shutil.copytree(manually_labeled_folder_merged, os.path.join(manually_labeled_folder_merged, 'manually_labeled'), dirs_exist_ok=True)
      shutil.rmtree(os.path.join(manually_labeled_folder_merged, 'images'))
      shutil.rmtree(os.path.join(manually_labeled_folder_merged, 'labels'))
      shutil.copytree(manually_labeled_folder_merged, auto_annotated_folders, dirs_exist_ok=True)

      # Get a list of the auto-annotated datasets directories
      auto_annotated_datasets_folders = [
        os.path.join(auto_annotated_folders, folder_name)
        for folder_name in os.listdir(auto_annotated_folders)
        if os.path.isdir(os.path.join(auto_annotated_folders, folder_name))
      ]

      #Prepare the auto-annotated dataset
      print("Preparing auto-annotated dataset...")
      merged_folder = f"{HOME}/merged_folder"
      os.makedirs(merged_folder, exist_ok=True)

      preprocess.merge_datasets(auto_annotated_datasets_folders, merged_folder)
      print("Auto-annotated dataset prepared.")

      # Preparing the final auto annotated datasets
      Final_output_folder = f"{HOME}/Final_auto_annotated_dataset_{iteration}"
      if os.path.exists(Final_output_folder):
        shutil.rmtree(Final_output_folder)
      os.makedirs(Final_output_folder, exist_ok=True)

      preprocess.resplit_dataset(merged_folder, Final_output_folder, train_ratio = 0.9, test_ratio = 0.0, val_ratio= 0.1, seed=42)
      print("Dataset Ready")

      # remove all folders creates except Final output folder
      shutil.rmtree(distributed_datasets)
      shutil.rmtree(remaining_unlabeled_images)
      shutil.rmtree(auto_annotated_folders)
      shutil.rmtree(manually_labeled_folder_merged)
      shutil.rmtree(merged_folder)
      shutil.rmtree(output_folder)
      shutil.rmtree(os.path.join(HOME, 'runs'))
      patterns = [os.path.join(HOME, "pseudo-labels-*"), os.path.join(HOME, "unlabeled_dataset_*")]
      for pattern in patterns:
        for folder in glob.glob(pattern):
          if os.path.isdir(folder):
            shutil.rmtree(folder)
      clear_output()
      return final_predictions_list

    else:

      # add the initial split to the auto annotated folders
      shutil.copytree(manually_labeled_folder_merged, os.path.join(manually_labeled_folder_merged, 'manually_labeled'), dirs_exist_ok=True)
      shutil.rmtree(os.path.join(manually_labeled_folder_merged, 'images'))
      shutil.rmtree(os.path.join(manually_labeled_folder_merged, 'labels'))
      shutil.copytree(manually_labeled_folder_merged, auto_annotated_folders, dirs_exist_ok=True)

      # Get a list of the auto-annotated datasets directories
      auto_annotated_datasets_folders = [
        os.path.join(auto_annotated_folders, folder_name)
        for folder_name in os.listdir(auto_annotated_folders)
        if os.path.isdir(os.path.join(auto_annotated_folders, folder_name))
      ]

      # Prepare the auto-annotated dataset
      print("Preparing auto-annotated dataset")
      merged_folder = f"{HOME}/merged_folder"
      os.makedirs(merged_folder, exist_ok=True)

      # merge the auto-annotated datasets
      preprocess.merge_datasets(auto_annotated_datasets_folders, merged_folder)
      print("Auto-annotated dataset prepared.")

      # Preparing the final auto annotated datasets
      Final_output_folder = f"{HOME}/Final_auto_annotated_dataset"
      if os.path.exists(Final_output_folder):
        shutil.rmtree(Final_output_folder)
      os.makedirs(Final_output_folder, exist_ok=True)

      # split the output folder
      preprocess.resplit_dataset(merged_folder, Final_output_folder, train_ratio = 0.9, test_ratio = 0.0, val_ratio= 0.1, seed=42)
      print("Dataset Ready")

      #remove all folders creates except Final output folder
      shutil.rmtree(distributed_datasets)
      shutil.rmtree(remaining_unlabeled_images)
      shutil.rmtree(auto_annotated_folders)
      shutil.rmtree(manually_labeled_folder_merged)
      shutil.rmtree(merged_folder)
      shutil.rmtree(output_folder)
      shutil.rmtree(os.path.join(HOME, 'runs'))
      patterns = [os.path.join(HOME, "pseudo-labels-*"), os.path.join(HOME, "unlabeled_dataset_*")]
      for pattern in patterns:
        for folder in glob.glob(pattern):
          if os.path.isdir(folder):
            shutil.rmtree(folder)
      clear_output()
      return final_predictions_list
