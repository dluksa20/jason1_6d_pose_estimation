import os

'''Method to remove specific data from dataset if needed'''
'''------------------------------------------------------------------------------------------------------'''
def remove_files_with_substring(directory_path, substring):
    for filename in os.listdir(directory_path):
        if substring in filename:
            file_path = os.path.join(directory_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except OSError as e:
                print(f"Error deleting {filename}: {e}")


ranges = [20,30,40] # ranges
FL =15 # indicator for the directory 

# initiate file delete
for range in ranges:
    directory_path = 'database/y-axis_FL{}_r{}/dmaps/'.format(FL, range)
    substring_to_remove = "0002.exr"# file id to remove or substring to remove multiple files

    remove_files_with_substring(directory_path, substring_to_remove)
