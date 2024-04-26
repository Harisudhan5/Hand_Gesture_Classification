import os

def count_files_in_directories(parent_directory):
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    for directory in directories:
        directory_path = os.path.join(parent_directory, directory)
        files_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        print(f"Number of files in {directory}: {files_count}")

# Replace 'parent_directory' with the path to your parent directory
parent_directory = 'Dataset/'
count_files_in_directories(parent_directory)
