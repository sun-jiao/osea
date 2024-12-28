import sys
import os
from os import listdir

from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm


def find_duplicate_images(directory, hash_size=8, hash_threshold=5):
    image_hashes = defaultdict(list)
    duplicates = []

    for root, _, files in tqdm(os.walk(directory)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)

                try:
                    image = Image.open(file_path)
                    image_hash = imagehash.phash(image, hash_size=hash_size)
                    image_hashes[str(image_hash)].append(file_path)
                except Exception as e:
                    print(f"Cannot process file: {file_path}: {e}")
                    continue

    # calculating duplicate images by phash
    for hash_val, paths in image_hashes.items():
        if len(paths) > 1:
            group = []
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    hash1 = imagehash.phash(Image.open(paths[i]), hash_size=hash_size)
                    hash2 = imagehash.phash(Image.open(paths[j]), hash_size=hash_size)
                    hash_distance = hash1 - hash2
                    if hash_distance <= hash_threshold:
                        if paths[i] not in group:
                            group.append(paths[i])
                        if paths[j] not in group:
                            group.append(paths[j])
            if group:
                duplicates.append(group)

    return duplicates


def clear_intra_class_duplicates(parent):
    for directory in listdir(parent):
        print(f'Processing directory: {directory}')

        duplicates = find_duplicate_images(os.path.join(parent, directory))

        if duplicates:
            print(f"Found {len(duplicates)} duplicates: ")
            for group in duplicates:
                for index, path in enumerate(group):
                    if index != 0:
                        os.remove(path)
                        print(f'Duplicate file deleted: {path}')
        else:
            print("No duplicate images found.")


def clear_inter_class_duplicates(directory):
    print(f'Processing directory: {directory}')

    duplicates = find_duplicate_images(os.path.join(parent, directory))

    if duplicates:
        print(f"Found {len(duplicates)} duplicates: ")
        for group in duplicates:
            for index, path in enumerate(group):
                os.remove(path)
                print(f'Duplicate file deleted: {path}')
    else:
        print("No duplicate images found.")


if __name__ == "__main__":
    parent = str(sys.argv[1])
    clear_intra_class_duplicates(parent)
    clear_inter_class_duplicates(parent)

