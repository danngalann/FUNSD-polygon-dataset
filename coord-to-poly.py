import os
import json
import threading
import shutil

def box_to_polygon(data):
    if "box" in data:
        xleft, ytop, xright, ybottom = data["box"]
        polygon = [xleft, ytop, xright, ytop, xright, ybottom, xleft, ybottom]
        data["polygon"] = polygon
        del data["box"]

    for key, value in data.items():
        if isinstance(value, dict):
            box_to_polygon(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    box_to_polygon(item)

    return data

def process_json_files(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    image_source_dir = os.path.join(source_dir.rsplit('/', 1)[0], "images")
    image_dest_dir = os.path.join(dest_dir.rsplit('/', 1)[0], "images")
    if not os.path.exists(image_dest_dir):
        os.makedirs(image_dest_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            # Process JSON files
            with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            data_with_polygon = box_to_polygon(data)

            with open(os.path.join(dest_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(data_with_polygon, f, indent=4)

            # Copy corresponding images
            image_name = filename.rsplit('.', 1)[0] + ".png"  # Assuming the images are in PNG format
            shutil.copy2(os.path.join(image_source_dir, image_name), os.path.join(image_dest_dir, image_name))

source_dirs = [
    'datasets/FUNSD/dataset/testing_data/annotations',
    'datasets/FUNSD/dataset/training_data/annotations'
]

dest_dirs = [
    'datasets/FUNSD_polygon/dataset/testing_data/annotations',
    'datasets/FUNSD_polygon/dataset/training_data/annotations'
]

threads = []
for source_dir, dest_dir in zip(source_dirs, dest_dirs):
    t = threading.Thread(target=process_json_files, args=(source_dir, dest_dir))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Processing completed!")
