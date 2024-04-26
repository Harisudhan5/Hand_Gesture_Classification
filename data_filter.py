import os

def keep_only_200_files(parent_directory, max_files=2000):
    directories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    for directory in directories:
        directory_path = os.path.join(parent_directory, directory)
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        if len(files) > max_files:
            files_to_delete = sorted(files)[max_files:]
            for file_to_delete in files_to_delete:
                os.remove(os.path.join(directory_path, file_to_delete))
            print(f"Deleted {len(files_to_delete)} files in {directory} directory to keep only {max_files}.")
        else:
            print(f"{directory} directory has {len(files)} files and does not need to be modified.")

# Replace 'parent_directory' with the path to your parent directory
parent_directory = 'Dataset'
keep_only_200_files(parent_directory)
