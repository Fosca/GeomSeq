"""
Author: Fosca Al Roumi <fosca.al.roumi@gmail.com>
"""

import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
