import os
import sys
import shutil
import random
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import zipfile
import yaml

def resplit_dataset(dataset_folder, dest_root_folder, train_ratio, test_ratio, val_ratio, seed=42):
    """
    Resplit the dataset into new training, testing, and validation sets.

    Parameters:
    - dataset_folder: Path to the source dataset folder containing 'images' and 'labels' subfolders.
    - dest_root_folder: Path to the root folder where the new splits will be saved.
    - train_ratio: Ratio of the dataset to be used for training.
    - test_ratio: Ratio of the dataset to be used for testing.
    - val_ratio: Ratio of the dataset to be used for validation.
    - seed: Random seed for reproducibility (optional).
    """
    # Set the random seed for reproducibility, if provided
    if seed is not None:
        random.seed(seed)

    total_ratio = train_ratio + test_ratio + val_ratio
    if not 0.999 <= total_ratio <= 1.001:
        raise ValueError("The sum of train_ratio, test_ratio, and val_ratio must be approximately 1.0.")

    # Define the paths for images and labels
    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')

    # Define destination subfolders for train, test, and val splits
    dest_subfolders = ['train', 'test', 'val']
    for subfolder in dest_subfolders:
        os.makedirs(os.path.join(dest_root_folder, subfolder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_root_folder, subfolder, 'labels'), exist_ok=True)

    # Gather all image-label pairs
    all_files = []
    for file_name in os.listdir(images_folder):
        image_file = os.path.join(images_folder, file_name)
        label_file = os.path.join(labels_folder, file_name.replace('.jpg', '.txt'))
        if os.path.isfile(image_file) and os.path.isfile(label_file):
            all_files.append((image_file, label_file))

    # Shuffle the list of files
    random.shuffle(all_files)
    total_files = len(all_files)
    train_count = int(train_ratio * total_files)
    test_count = int(test_ratio * total_files)
    val_count = total_files - train_count - test_count

    # Helper function to copy files to the destination folders
    def copy_files(file_list, dest_subfolder):
        for image_file, label_file in file_list:
            dest_image_folder = os.path.join(dest_root_folder, dest_subfolder, 'images')
            dest_label_folder = os.path.join(dest_root_folder, dest_subfolder, 'labels')

            shutil.copy(image_file, os.path.join(dest_image_folder, os.path.basename(image_file)))
            shutil.copy(label_file, os.path.join(dest_label_folder, os.path.basename(label_file)))

    # Split and copy files into train, test, and val folders
    copy_files(all_files[:train_count], 'train')
    copy_files(all_files[train_count:train_count + test_count], 'test')
    copy_files(all_files[train_count + test_count:], 'val')
    output_dir = Path(dest_root_folder)

    # Create data.yaml
    yaml_content = f"""
    path: {output_dir}
    train: train/images
    val: val/images
    test: test/images

    names:
      0: aeroplane
      1: bicycle
      2: bird
      3: boat
      4: bottle
      5: bus
      6: car
      7: cat
      8: chair
      9: cow
      10: diningtable
      11: dog
      12: horse
      13: motorbike
      14: person
      15: pottedplant
      16: sheep
      17: sofa
      18: train
      19: tvmonitor
    """

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"Resplit dataset into {train_ratio*100}% train, {test_ratio*100}% test, {val_ratio*100}% val.")
    return dest_root_folder

def distribute_dataset( source_folder, dest_root_folder, num_img_train, num_img_val, num_dest_folders, class_names ,ignore_Remaining_images):
    """
    Distribute dataset images and labels into multiple destination folders with `data.yaml`.

    Parameters:
    - source_folder: Path to the source data.
    - dest_root_folder: Path to the root folder where the distributed folders will be created.
    - num_img_train: Maximum number of images per 'train' folder in each destination folder.
    - num_img_val: Maximum number of images per 'valid' folder in each destination folder.
    - num_dest_folders: Number of destination folders to create.
    - ignore_Remaining_images: If True, ignore any remaining images that cannot be distributed. If False, move them to a separate folder.
    """

    # Verify dataset structure
    subfolders = ['train', 'valid']
    is_auto_folder = False
    if os.path.exists(os.path.join(source_folder, 'images')):
        print(f"Source folder '{os.path.join(source_folder, 'images')}' exist.")
        total_images = len(os.listdir(os.path.join(source_folder, 'images')))
        is_auto_folder = True
    elif os.path.exists(os.path.join(source_folder, subfolders[0], 'images')):
        print(f"Source folder '{os.path.join(source_folder, subfolders[0], 'images')}' exist.")
        total_images = (len(os.listdir(os.path.join(source_folder, subfolders[0], 'images'))) + len(os.listdir(os.path.join(source_folder, subfolders[1], 'images'))))
        print(f"Total images in the dataset: {total_images}")
    else:
        print(f"Source folder '{source_folder}' does not exist.")
        return

    images_per_folder = num_img_train + num_img_val
    dest_folder_idx = 1
    train_img_count = 0
    val_img_count = 0
    remaining_images = []

    # Distribute images
    if is_auto_folder:
        images_path = os.path.join(source_folder, 'images')
        labels_path = os.path.join(source_folder, 'labels')

        for img_file in os.listdir(images_path):
            if train_img_count + val_img_count >= images_per_folder:
                if dest_folder_idx >= num_dest_folders:
                    remaining_images.append((images_path, img_file))
                    continue
                dest_folder_idx += 1
                train_img_count = 0
                val_img_count = 0

            dest_folder = os.path.join(dest_root_folder, f'folder_{dest_folder_idx}')
            os.makedirs(os.path.join(dest_folder, 'train', 'images'), exist_ok=True)
            os.makedirs(os.path.join(dest_folder, 'train', 'labels'), exist_ok=True)
            os.makedirs(os.path.join(dest_folder, 'valid', 'images'), exist_ok=True)
            os.makedirs(os.path.join(dest_folder, 'valid', 'labels'), exist_ok=True)

            img_src_path = os.path.join(images_path, img_file)
            label_src_path = os.path.join(labels_path, os.path.splitext(img_file)[0] + '.txt')

            if train_img_count < num_img_train:
                img_dest_path = os.path.join(dest_folder, 'train', 'images', img_file)
                label_dest_path = os.path.join(dest_folder, 'train', 'labels', os.path.splitext(img_file)[0] + '.txt')
                train_img_count += 1
                shutil.copy(img_src_path, img_dest_path)
                if os.path.exists(label_src_path):
                    shutil.copy(label_src_path, label_dest_path)

            elif val_img_count < num_img_val:
                img_dest_path = os.path.join(dest_folder, 'valid', 'images', img_file)
                label_dest_path = os.path.join(dest_folder, 'valid', 'labels', os.path.splitext(img_file)[0] + '.txt')
                val_img_count += 1
                shutil.copy(img_src_path, img_dest_path)
                if os.path.exists(label_src_path):
                    shutil.copy(label_src_path, label_dest_path)

    else:
        for subfolder in subfolders:
            images_path = os.path.join(source_folder, subfolder, 'images')
            labels_path = os.path.join(source_folder, subfolder, 'labels')

            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                print(f"Subfolder '{subfolder}' does not exist in the source folder.")
                continue

            for img_file in os.listdir(images_path):
                if train_img_count + val_img_count >= images_per_folder:
                    if ignore_Remaining_images == False:
                        if dest_folder_idx >= num_dest_folders:
                            remaining_images.append((images_path, img_file))
                            continue
                    dest_folder_idx += 1
                    train_img_count = 0
                    val_img_count = 0

                dest_folder = os.path.join(dest_root_folder, f'folder_{dest_folder_idx}')
                os.makedirs(os.path.join(dest_folder, 'train', 'images'), exist_ok=True)
                os.makedirs(os.path.join(dest_folder, 'train', 'labels'), exist_ok=True)
                os.makedirs(os.path.join(dest_folder, 'valid', 'images'), exist_ok=True)
                os.makedirs(os.path.join(dest_folder, 'valid', 'labels'), exist_ok=True)

                img_src_path = os.path.join(images_path, img_file)
                label_src_path = os.path.join(labels_path, os.path.splitext(img_file)[0] + '.txt')

                if train_img_count < num_img_train:
                    img_dest_path = os.path.join(dest_folder, 'train', 'images', img_file)
                    label_dest_path = os.path.join(dest_folder, 'train', 'labels', os.path.splitext(img_file)[0] + '.txt')
                    train_img_count += 1
                    shutil.copy(img_src_path, img_dest_path)
                    if os.path.exists(label_src_path):
                        shutil.copy(label_src_path, label_dest_path)

                elif val_img_count < num_img_val:
                    img_dest_path = os.path.join(dest_folder, 'valid', 'images', img_file)
                    label_dest_path = os.path.join(dest_folder, 'valid', 'labels', os.path.splitext(img_file)[0] + '.txt')
                    val_img_count += 1
                    shutil.copy(img_src_path, img_dest_path)
                    if os.path.exists(label_src_path):
                        shutil.copy(label_src_path, label_dest_path)

    print(f"Distributed images into {dest_folder_idx} folders.")

    # Load class information from the main data.yaml in the source folder
    main_data_yaml_path = os.path.join(source_folder, 'data.yaml')

    if not os.path.exists(main_data_yaml_path):
        print(f"No data.yaml found in {source_folder}, creating a default one.") 
        classes = class_names
        num_classes = len(class_names)
    else:
        with open(main_data_yaml_path, 'r') as file:
            main_data_yaml = yaml.safe_load(file)
            classes = main_data_yaml.get('names', [])
            num_classes = main_data_yaml.get('nc', len(classes))
            
    # Create dynamic data.yaml files
    for i in range(1, dest_folder_idx + 1):
      dest_folder = os.path.join(dest_root_folder, f'folder_{i}')

      # Ensure the destination folder exists
      os.makedirs(dest_folder, exist_ok=True)

      data_yaml_content = {
          "nc": num_classes,
          "names": classes,
          "train": os.path.join(dest_folder, 'train'),
          "val": os.path.join(dest_folder, 'valid')
      }

      data_yaml_path = os.path.join(dest_folder, 'data.yaml')

      try:
          # Write the data.yaml file
          with open(data_yaml_path, 'w') as yaml_file:
              yaml.dump(data_yaml_content, yaml_file, default_flow_style=False)
          print(f"Successfully created data.yaml at {data_yaml_path}")
      except Exception as e:
          print(f"Failed to write data.yaml at {data_yaml_path}: {e}")
          raise


    if ignore_Remaining_images:
        return dest_root_folder

    # Move remaining images to a separate folder
    unlabeled_folder = os.path.join(dest_root_folder, 'unlabeled_data')
    os.makedirs(os.path.join(unlabeled_folder, 'images'), exist_ok=True)

    for images_path, img_file in remaining_images:
        img_src_path = os.path.join(images_path, img_file)
        img_dest_path = os.path.join(unlabeled_folder, 'images', img_file)
        shutil.copy(img_src_path, img_dest_path)

    print(f"Remaining images copied to {unlabeled_folder}.")

    return unlabeled_folder, dest_root_folder


def update_unlabeled_folder(iteration_unlabeled_folder , main_unlabeled_folder, final_labels_folder, Auto_annotated_folder, final_iteration = False):
    """
    Update the main unlabeled folder with the final set of auto-annotated labels.
    Move images and auto-annotated labels to Auto_annotated_folder.
    If a label does not exist for an image in the final set, move this image to the main_unlabeled_folder.

    Parameters:
    - iteration_unlabeled_folder: The folder containing current iteration images
    - main_unlabeled_folder: The main unlabeled folder containing images without labels.
    - final_labels_folder: The folder containing the final set of auto-annotated labels.
    - Auto_annotated_folder: The folder where images and auto-annotated labels will be moved.
    """

    # Ensure the Auto_annotated_folder exists
    os.makedirs(Auto_annotated_folder, exist_ok=True)

    # Define paths to 'images' and 'labels' directories within Auto_annotated_folder
    auto_annotated_images_folder = os.path.join(Auto_annotated_folder, 'images')
    auto_annotated_labels_folder = os.path.join(Auto_annotated_folder, 'labels')

    os.makedirs(auto_annotated_images_folder, exist_ok=True)
    os.makedirs(auto_annotated_labels_folder, exist_ok=True)

    # List all images in the main_unlabeled_folder
    for image_file in os.listdir(iteration_unlabeled_folder):
      image_path = os.path.join(iteration_unlabeled_folder, image_file)

      # labels have the same filename as images, but with a different extension (.txt)
      label_file = Path(image_file).stem + ".txt"
      label_path = os.path.join(final_labels_folder, label_file)

      if os.path.exists(label_path):
        # Move the image and its label to the Auto_annotated_folder
        shutil.move(image_path, os.path.join(auto_annotated_images_folder, image_file))
        shutil.move(label_path, os.path.join(auto_annotated_labels_folder, label_file))
      else:
        # Move the image to the main_unlabeled_folder
        if final_iteration == False:
          shutil.move(image_path, os.path.join(main_unlabeled_folder, image_file))

    print(f"Moved images and auto-annotated labels to {Auto_annotated_folder}.")
    return Auto_annotated_folder

def merge_datasets(source_folders, output_folder):
    """
    Merge images and labels from multiple source folders into one dataset folder.

    Parameters:
    - source_folders: List of paths to source folders containing 'images' and 'labels' subfolders.
    - output_folder: Path to the output dataset folder where all images and labels will be moved.
    """

    # Create output directories for images and labels
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # Iterate through each source folder
    print(f'source_folders: {source_folders}')
    for source_folder in source_folders:
        print(f"Processing folder: {source_folder}")

        # Skip hidden folders like '.ipynb_checkpoints'
        if os.path.basename(source_folder).startswith('.'):
            continue

        images_folder = os.path.join(source_folder, 'images')
        labels_folder = os.path.join(source_folder, 'labels')

        # Check if the images and labels folders exist before processing
        if not os.path.exists(images_folder):
            print(f"No images folder found in {source_folder}, skipping.")
            continue

        if not os.path.exists(labels_folder):
            print(f"No labels folder found in {source_folder}, skipping.")
            continue

        # Move images
        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            if os.path.isfile(image_path):
                target_image_path = os.path.join(output_images_folder, image_file)

                # Handle file name conflicts by skipping the file
                if os.path.exists(target_image_path):
                    continue

                shutil.move(image_path, target_image_path)

        # Move labels
        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            if os.path.isfile(label_path):
                target_label_path = os.path.join(output_labels_folder, label_file)

                # Handle file name conflicts by skipping the file
                if os.path.exists(target_label_path):
                    continue

                shutil.move(label_path, target_label_path)

    print(f"All images and labels have been moved to {output_folder}.")
def move_images(src_folder, dest_folder, num_images):
    """
    Move a specified number of images from the source folder to the destination folder.
    """
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # List all images in the source folder
    images_folder = os.path.join(src_folder, 'images')
    all_images = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    # Randomly sample the specified number of images
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    # Move selected images to the destination folder
    for image in selected_images:
        shutil.move(os.path.join(images_folder, image), os.path.join(dest_folder, image))

    return dest_folder
