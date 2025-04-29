import time
from ultralytics import YOLO
import os
import shutil
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import torch
import torch.multiprocessing as mp
from config import HOME

def train_single_instance(dataset_path, epochs_per_iteration, model, img_size, iteration, RetrainAll):
    """Train a single YOLO instance and return the best model path and its performance."""

    try:
        if RetrainAll and iteration % 3 == 0:
          print(f"Loading yolov8n.yaml")
          model = YOLO('yolo11n.pt')

        elif model is not None:
          print(f"Loading teacher model")
          model = YOLO(model)

        else:
          print(f"Loading yolov8n.yaml")
          model = YOLO('yolo11n.pt')

        # Train the model
        results = model.train(data=f'{dataset_path}/data.yaml', epochs=epochs_per_iteration, imgsz=img_size, plots=True, patience = 10, augment=True, project=f'{HOME}/runs') 
        save_dir = results.save_dir  # Get the save directory

        print(f"Model saved to: {save_dir}")

        # Path to the best model
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        print(f"Best model saved to: {best_model_path}")

        # Performance metrics
        metrics = results.maps.mean()
        return best_model_path, metrics
        
    except Exception as e:
        print(f"Error training model on {dataset_path}: {e}")
        traceback.print_exc()
        return None, None
'''
def train_multiple_instances(datasets, epochs_per_iteration, trained_models_paths, img_size, iteration, RetrainAll):
    """Train multiple YOLO models on different datasets in parallel using ProcessPoolExecutor."""
    best_model_paths = []
    model_performance = []

    # Create a process pool excutor
    with ProcessPoolExecutor(max_workers=2) as executor:

        print(f"process pool executor")
        # Prepare futures for parallel training tasks
        futures = [
            executor.submit(train_single_instance, dataset, epochs_per_iteration, model, img_size,iteration, RetrainAll)
            for dataset, model in zip(datasets, trained_models_paths)
        ]

        # Process results as each task completes
        for future in as_completed(futures):
            print(f"future completed")
            best_model_path, metrics = future.result()
            if best_model_path:
                best_model_paths.append(best_model_path)
                model_performance.append(metrics)

    torch.cuda.empty_cache()  # Clear GPU memory after training
    return best_model_paths, model_performance

'''
def train_multiple_instances(datasets, epochs_per_iteration, trained_models_paths, img_size, iteration, RetrainAll):
    """Train multiple YOLO models on different datasets and return best model paths and performance metrics."""
    best_model_paths = []
    model_performance = []

    # Train each model sequentially
    for dataset, model in zip(datasets, trained_models_paths):
        best_model_path, metrics = train_single_instance(dataset, epochs_per_iteration, model, img_size, iteration, RetrainAll)
        best_model_paths.append(best_model_path)
        model_performance.append(metrics)

    #torch.cuda.empty_cache()  # Clear GPU memory after training
    return best_model_paths, model_performance

def check_model_paths(paths):
  '''
  function to delete any extra folders that could be created during parallel training
  '''
  # Iterate over each path in the list
  for path in paths:
    print(f"Checking path: {path}")
    # Check if the best.pt file exist
    if not os.path.isfile(path):
      print(f"The file 'best.pt' does not exist in {path}.")
      # If best.pt does not exist, delete the parent 'train' folder
      train_folder = os.path.dirname(os.path.dirname(path))
      shutil.rmtree(train_folder)
      print(f"The folder '{train_folder}' has been deleted because best.pt was not found.")
    else:
      print(f"The file 'best.pt' exists in {path}.")

def Pseudo_Labeling(best_model_paths, unlabeled_dataset, iteration):
    """
    Use best YOLO models to predict the pseudo labels.
    Create a new directory for each iteration, move the labels folder, and clean up.
    """

    # Define the home directory and iteration-specific directory
    model_id = 0
    iteration_folder_name = f'pseudo-labels-{iteration}'
    iteration_folder_path = os.path.join(HOME, iteration_folder_name)

    # Create the iteration-specific directory if it does not exist
    os.makedirs(iteration_folder_path, exist_ok=True)

    print("Pseudo-Labeling the next set of unlabeled images...")

    for i, model_path in enumerate(best_model_paths):
        print(f"Model {i+1}:{model_path} predicting unlabeled images...")
        model = YOLO(model_path)
        results = model.predict(unlabeled_dataset, project=f'{HOME}/runs',save=False, save_txt=True, save_conf=True)
        print(f"Predictions for model {i+1} saved.")

        save_dir = Path(results[0].save_dir)
        labels_source = save_dir / 'labels'
        print(f"Labels saved to: {labels_source}")
        
        # Define the path for labels destination
        labels_name = f'labels-model-{model_id}'
        labels_path = os.path.join(iteration_folder_path, labels_name)
        os.makedirs(labels_path, exist_ok=True)

        # Move label files to the new directory
        if labels_source.exists():
            for item in labels_source.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(labels_path))

        # Delete all files created during current iteration
        if labels_source.exists():
          shutil.rmtree(labels_source)

        model_id += 1

    entries = os.listdir(iteration_folder_path)
    # Create a list of full paths
    list_iteration_folder_path = [os.path.join(iteration_folder_path, entry) for entry in entries]
    print(f'list of pseudo_labels is: {list_iteration_folder_path}')

    torch.cuda.empty_cache()  # Clear GPU memory after prediction
    return list_iteration_folder_path

def train_final_model(iteration ,dataset, img_size, train_epochs):
  #Start Timer
  start_time = time.time()

  #Train the model
  print("Training final model...")
  model = YOLO('yolo11n.pt')
  results = model.train(data=f'{dataset}/data.yaml', epochs=train_epochs, imgsz=img_size, plots=True, augment=True, patience = 20, project=f'{HOME}/Final_runs', name=f'Final_model_runs_{iteration}')
  save_dir = results.save_dir  # Get the save directory

  # Path to the best model
  best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
  print(f"Best model saved to: {best_model_path}")

  #End Timer
  end_time = time.time()
  runtime = (end_time - start_time) / 3600
  print(f"Runtime: {runtime:.2f} hr")

  return best_model_path, runtime
