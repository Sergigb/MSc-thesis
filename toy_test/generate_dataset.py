import json
import os
import random
from shutil import copyfile

max_images = 5000
dataset_images_path = '../../datasets/ali/all_images'

image_folders = os.listdir(dataset_images_path)

topic_1_words = 'plane', 'boeing', 'airplane', 'aviation', 'civil', 'aircraft', 'commercial', 'airline', 'airbus', \
                'flight'
topic_2_words = 'war', 'world_war', 'battle', 'weapon', 'invasion', 'tank', 'squadron', 'bomb', 'infantry', 'missile', \
                'fighter'
topic_12_words = 'aviation', 'flight', 'plane', 'airplane', 'fighter', 'missile', 'bomb', 'war'

if not os.path.exists('images'):
    os.mkdir('images')

images_topic1 = []
images_topic2 = []
images_topic12 = []
for image_folder in image_folders:
    image_filenames = os.listdir(os.path.join(dataset_images_path, image_folder))
    print("Searching in folder " + image_folder)
    for filename in image_filenames:
        if (filename.lower().find("boeing") != -1 or filename.lower().find("airbus") != -1) \
                and len(images_topic1) < max_images:
            images_topic1.append(os.path.join(image_folder, filename))
        if filename.lower().find("world_war") != -1 or \
                filename.lower().find("missile") != -1 or \
                filename.lower().find("_tank_") != -1 \
                    and len(images_topic2) < max_images:
            images_topic2.append(os.path.join(image_folder, filename))
        if filename.lower().find("eurofighter") != -1 or \
            filename.lower().find("f-16") != -1 or \
            filename.lower().find("f-15") != -1 or \
            filename.lower().find("f-22") != -1 or \
            filename.lower().find("_mig_") != -1 or \
            filename.lower().find("sukhoi") != -1 or \
            filename.lower().find("su-35") != -1 or \
            filename.lower().find("su-22") != -1 or \
            filename.lower().find("f-35") != -1 \
            and len(images_topic12) < max_images:
            images_topic12.append(os.path.join(image_folder, filename))


print(len(images_topic1))
print(len(images_topic2))
print(len(images_topic12))

data_pairs = []
for image_path in images_topic1:
    filename = image_path.split("/")[-1]
    words = [random.choice(topic_1_words) for _ in range(5)]
    text = ""
    for word in words:
        text += " " + word
    contents = {'img': image_path,
                'text': text}
    copyfile(os.path.join(dataset_images_path, image_path), os.path.join('images', filename))
    data_pairs.append(contents)

for image_path in images_topic2:
    filename = image_path.split("/")[-1]
    words = [random.choice(topic_2_words) for _ in range(5)]
    text = ""
    for word in words:
        text += " " + word
        contents = {'img': image_path,
                'text': text}
    copyfile(os.path.join(dataset_images_path, image_path), os.path.join('images', filename))
    data_pairs.append(contents)

for image_path in images_topic12:
    filename = image_path.split("/")[-1]
    words = [random.choice(topic_12_words) for _ in range(5)]
    text = ""
    for word in words:
        text += " " + word
    contents = {'img': image_path,
                'text': text}
    copyfile(os.path.join(dataset_images_path, image_path), os.path.join('images', filename))
    data_pairs.append(contents)

with open('data_pairs.json', 'w') as f:
    json.dump(data_pairs, f)

