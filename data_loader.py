import os
import json

from PIL import Image
import torch
import torch.utils.data as data


class CLEF_Wikipedia_dataset(data.Dataset):
    def __init__(self, dataset_path, json_path, transform, return_ids=False):
        """
        dataset_path: path to the dataset
        json_path: path to the json with the train labels
        transform: image transform, at least to tensor 
        """
        self.dataset_path = dataset_path
        self.json_path = json_path
        self.transform = transform
        self.return_ids = return_ids

        with open(json_path) as f:
            self.train_labels = json.load(f)

        self.ids = list(self.train_labels.keys())

    def __getitem__(self, index):
        t = torch.tensor(self.train_labels[self.ids[index]])
        image_path = self.ids[index]
        image = Image.open(os.path.join(self.dataset_path, image_path))
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.return_ids:
            return image, t, self.ids[index]
        else:
            return image, t

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    if len(data[0]) == 2:
        images, labels = zip(*data)

        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        return images, labels
    else:
        images, labels, ids = zip(*data)

        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        return images, labels, ids



def get_wiki_data_loader(dataset_path, json_path, transform, 
                         batch_size, shuffle=True, num_workers=4, return_ids=False):

    wiki_dataset = CLEF_Wikipedia_dataset(dataset_path, json_path, transform,
                                          return_ids=return_ids)

    data_loader = data.DataLoader(dataset=wiki_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=collate_fn)

    return data_loader

