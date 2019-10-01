import json
import sys
import os


dataset_path = "../../datasets/ali"
images_path = os.path.join(dataset_path, 'all_images')
texts_path = os.path.join(dataset_path, 'all')
json_files = ['wiki_all_0.json',  'wiki_all_2.json',  'wiki_all_4.json',  'wiki_all_6.json',  'wiki_all_8.json', 
              'wiki_all_1.json',  'wiki_all_3.json',  'wiki_all_5.json',  'wiki_all_7.json',  'wiki_all_9.json']

counter = 0
key_errors = 0
missing_images = 0

num_pairs = 1000000
i = 0

for file in json_files:
    f = open(os.path.join(texts_path, file))
    data = json.load(f)
    keys = data.keys()

    data_pairs = []
    for key in keys:
        if data[key] is not None:
            section_keys = data[key]['sections'].keys()
            for section_key in section_keys:
                section = data[key]['sections'][section_key]
                if section.has_key('imgs'):
                    imgs_keys = section['imgs'].keys()
                    for img_key in imgs_keys:
                        try:
                            if section['imgs'][img_key]['filepath'] is not None:
                                if counter > num_pairs:
                                    break
                                filename = (section['imgs'][img_key]['img_url'].split("/")[-1])
                                filename = "File:" + filename
                                path_to_image = os.path.join(dataset_path, section['imgs'][img_key]['filepath'].replace("./", ""), filename)
                                if os.path.exists(path_to_image):
                                    data_pairs.append({'text':section['text'], 'img':path_to_image})
                                    sys.stdout.write("\rNum pairs found: " + str(counter))
                                    sys.stdout.flush()
                                    counter += 1
                                else:
                                    missing_images += 1 

                        except:
                            key_errors += 1

    del data
    f.close()
    i += 1
    with open('data_pairs_' + str(i) + '.json', 'w') as f:
        json.dump(data_pairs, f)
    if counter > num_pairs:
        break


print("")
print("total images found", counter)
print("key errors", key_errors)
print("missing images", missing_images)
print("total num pairs", len(data_pairs))

