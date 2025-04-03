import sys

import scipy.io
import pandas as pd
import numpy as np

UNIX_START_TIME = 719529


def prepare_dataset(path_to_dataset):
    try:
        data = scipy.io.loadmat(path_to_dataset)
    except FileNotFoundError:
        print('Nie ma takiego pliku')
        sys.exit()

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
        (df['age'] >= 10) & (df['age'] <= 95) &
        (~np.isinf(df['face_score'])) & (df['face_score'] > 0.7) &
        (np.isnan(df['second_face_score']))
        ]
    df_clean.reset_index()

    print(f'Images left after cleaning the dataset and selecting from the specified age range: {len(df_clean)}')

    return df_clean
