import os

from dataset import Dataset


def main():
    # path_to_dataset = input("Podaj sciezke do katalogu zawierajacego zbior: ")
    dataset = 'F:\\imdb_crop\\'
    clean_dataset = 'F:\\clean_dataset\\'
    os.makedirs(dataset + 'clean', exist_ok=True)

    # faces_dataset = Dataset(dataset_path=dataset)
    # faces_dataset.prepare_dataset()
    # faces_dataset.histogram(blocking=True)

    faces_dataset_2 = Dataset(dataset_path=clean_dataset)
    faces_dataset_2.load_dataset()
    # faces_dataset_2.augment_rare_classes()
    # faces_dataset_2.update_df_with_augmented()
    faces_dataset_2.histogram(blocking=True)


if __name__ == '__main__':
    main()
