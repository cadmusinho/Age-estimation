import os
import shutil

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from random import choice

UNIX_START_TIME = 719529


class Dataset:
    df_clean = None
    dataset_path = None

    def __init__(self, dataset_path, ):
        self.dataset_path = dataset_path

    def load_dataset(self):
        try:
            data = scipy.io.loadmat(self.dataset_path + 'imdb.mat')
        except FileNotFoundError:
            print('Nie ma takiego pliku')
            return

        name = data['name'][0]
        age = data['age'][0]
        face_score = data['face_score'][0]
        second_face_score = data['second_face_score'][0]
        path = [str(f[0]) for f in data['path'][0]]

        print(f'Total records from dataset: {len(name)}\n')

        self.df_clean = pd.DataFrame({
            'name': name,
            'path': path,
            'age': age,
            'face_score': face_score,
            'second_face_score': second_face_score
        })

    def prepare_dataset(self):
        try:
            data = scipy.io.loadmat(self.dataset_path + 'imdb.mat')
        except FileNotFoundError:
            print('Nie ma takiego pliku')
            return

        meta = data['imdb'][0, 0]

        name = meta['name'][0]
        photo_taken = meta['photo_taken'][0]
        dob = meta['dob'][0]
        face_score = meta['face_score'][0]
        second_face_score = meta['second_face_score'][0]
        full_path = [str(f[0]) for f in meta['full_path'][0]]

        bad_date_counter = 0
        age = np.full(len(dob), np.nan)

        print(f'Total records from dataset: {len(dob)}\n')

        for i in range(len(dob)):
            if name[i][0] == '' or full_path[i] == '':
                continue

            try:
                dob_date = pd.to_datetime(dob[i] - UNIX_START_TIME, unit='D')
                age[i] = photo_taken[i] - dob_date.year
            except:
                bad_date_counter += 1

        print(f'Counted ages: {len(age)}')
        print(f'Records with bad date format: {bad_date_counter}')

        df = pd.DataFrame({
            'name': name,
            'path': full_path,
            'age': age,
            'face_score': face_score,
            'second_face_score': second_face_score
        })

        df_clean = df[
            (df['age'] >= 18) & (df['age'] <= 65) &
            (~np.isinf(df['face_score'])) & (df['face_score'] > 0.7) &
            (np.isnan(df['second_face_score']))
            ]
        df_clean.reset_index(drop=True, inplace=True)

        print(f'Images left after cleaning the dataset and selecting from the specified age range: {len(df_clean)}')

        self.df_clean = df_clean

    def histogram(self, blocking):
        if self.df_clean.empty:
            print('Dataframe is empty')
            return

        plt.show()

        plt.hist(self.df_clean['age'], bins=48)
        plt.xticks(np.arange(15, 75, 1))
        plt.show(block=blocking)

    def augment_rare_classes(self):
        target_count = 5500

        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        age_counts = self.df_clean['age'].value_counts()

        for age, count in age_counts.items():
            if count >= target_count:
                continue

            age_dir = self.dataset_path + f'augmented\\{str(int(age))}'
            os.makedirs(age_dir, exist_ok=True)

            age_df = self.df_clean[self.df_clean['age'] == age]

            current_total = count
            augment_index = 0

            while current_total < target_count:
                row = age_df.sample(n=1).iloc[0]
                img_path = row['path']

                try:
                    img = load_img(self.dataset_path + img_path)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, 0)

                    for batch in datagen.flow(img_array, batch_size=1):
                        new_img = array_to_img(batch[0])
                        new_name = f'aug_{augment_index}_{os.path.basename(img_path)}'
                        save_path = self.dataset_path + f'augmented\\{str(int(age))}\\' + new_name
                        new_img.save(save_path)

                        current_total += 1
                        augment_index += 1
                        break
                except Exception as e:
                    print(repr(e))
                    return

    def update_df_with_augmented(self):
        new_rows = []

        for age_folder in os.listdir(self.dataset_path + 'augmented'):
            age_dir = self.dataset_path + f'augmented\\{age_folder}'
            if not os.path.isdir(age_dir):
                continue

            for file_name in os.listdir(age_dir):
                if not file_name.startswith('aug_'):
                    continue

                original_name = '_'.join(file_name.split('_')[2:])

                matches = self.df_clean[self.df_clean['path'].str.contains(original_name)]
                if matches.empty:
                    continue

                orig = matches.iloc[0]

                new_path = f'{age_folder}\\{file_name}'

                new_row = {
                    'name': orig['name'],
                    'path': new_path,
                    'age': orig['age'],
                    'face_score': orig['face_score'],
                    'second_face_score': orig['second_face_score']
                }

                new_rows.append(new_row)

        df_aug = pd.DataFrame(new_rows)
        self.df_clean = pd.concat([self.df_clean, df_aug], ignore_index=True)

        self.save_to_mat('F:\\clean_dataset', 'imdb_aug.mat')

    def save_to_mat(self, path, file):
        def to_cell_str(series):
            return np.array([series.astype(str).tolist()], dtype=object)

        mat_data = {
            "name": self.df_clean['name'],
            "path": to_cell_str(self.df_clean['path']),
            "age": self.df_clean['age'].astype(float).to_numpy(),
            "face_score": self.df_clean['face_score'].astype(float).to_numpy(),
            "second_face_score": self.df_clean['second_face_score'].astype(float).to_numpy(),
        }

        scipy.io.savemat(f'{path}\\{file}', mat_data)