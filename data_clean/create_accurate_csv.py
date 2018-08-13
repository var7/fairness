import sys
import pandas as pd
import imghdr
import hashlib
import os
import numpy as np
# %%

def remove_missing_files(df, data_path):
    count = 0
    indices = []
    curropt = []
    for index, row in df.iterrows():
        name = row['name'].replace(' ', '_')
        img_id = str(row['image_id'])

        # img_name = name + '_' + img_id + '.jpeg'
        img_name = hashlib.sha1(
            row['url'].encode('utf-8')).hexdigest() + '.jpg'
        # print(img_name)
        # img_path = os.path.join(data_path, 'images', name, img_name)
        img_path = os.path.join(data_path, name, 'face', img_name)
        if os.path.isfile(img_path):
            if imghdr.what(img_path) is not None:
                # print(img_path)
                continue
            else:
                print('Image is corrupt {}'.format(img_path))
                count += 1
                if "actor" in data_path:
                    folder = "actor"
                else:
                    folder = "actress"
                curropt.append(os.path.join(folder, name, img_name))
                curropt.append(os.path.join(folder, name, 'face', img_name))
                indices.append(index)
        else:
            count += 1
            indices.append(index)
            

    with open('corrupt_files.txt', 'a') as output:
        for fname in curropt:
            output.write(fname)
            output.write('\n')
    print('corrupt file names written to file')
    return df.drop(actors_frame.index[indices]), count


def add_name_id(df):
    df['person_id'] = pd.Categorical(pd.factorize(df.name)[0] + 1)
    return df


def data_split(low=1, class_size=265, holdout_frac=0.2, val_test_split=0.5, seed=1791387):
    random_state = np.random.RandomState(seed)
    all_pids = np.arange(low, low+class_size+1)
    holdout_pids = random_state.randint(
        low, low+class_size+1, size=int(class_size*holdout_frac))
    val_pids = random_state.choice(holdout_pids, size=int(
        class_size*holdout_frac*val_test_split), replace=False)
    test_pids = np.setdiff1d(holdout_pids, val_pids)
    train_pids = np.setdiff1d(all_pids, holdout_pids)
    return train_pids, val_pids, test_pids

# def data_split_verification():
#     pid_to_num = 


ANNOT_ACTORS_PATH = "/home/var/facescrub/facescrub_actors.txt"
ANNOT_ACTRESS_PATH = "/home/var/facescrub/facescrub_actresses.txt"

DATA_PATH = "/home/var/final-fs-data/"
DATA_ACTORS_PATH = "/home/var/final-fs-data/actor/"
DATA_ACTRESS_PATH = "/home/var/final-fs-data/actress/"

SAVE_PATH = "/home/var/final-fs-data/"

actors_frame = pd.read_csv(ANNOT_ACTORS_PATH, delimiter='\t')
actors_frame['gender'] = 'male'

print('Before deletion, actors frame')
print(actors_frame.head())
print('Shape:', actors_frame.shape)
print('Number of entries: ', len(actors_frame))

updated_actors_frame, actors_count = remove_missing_files(
    actors_frame, DATA_ACTORS_PATH)

print('After deletion, actors frame')
print(updated_actors_frame.head())
print('Shape:', updated_actors_frame.shape)
print('{} lines were deleted'.format(actors_count))
print('Number of remaining entries: ', len(updated_actors_frame))

# %%
actress_frame = pd.read_csv(ANNOT_ACTRESS_PATH, delimiter='\t')
actress_frame['gender'] = 'female'

print('Before deletion, actress frame')
print(actress_frame.head())
print('Shape: ', actress_frame.shape)
print('Number of entries: ', len(actress_frame))

updated_actress_frame, actress_count = remove_missing_files(
    actress_frame, DATA_ACTRESS_PATH)

print('After deletion, actress frame')
print(updated_actress_frame.head())
print('Shape:', updated_actress_frame.shape)
print('{} lines were deleted'.format(actress_count))
print('Number of remaining entries: ', len(updated_actress_frame))

new_full_frame = updated_actors_frame.append(
    updated_actress_frame, ignore_index=True)
print(new_full_frame.keys())
full_frame_with_ids = add_name_id(new_full_frame)

print(full_frame_with_ids.head())
print(full_frame_with_ids.tail())
print('Shape:', full_frame_with_ids.shape)
print('Total number of images downloaded: ', len(full_frame_with_ids))

full_frame_with_ids.to_csv(
    SAVE_PATH+'full_facescrub_with_ids.txt', sep='\t', index=False)
print('Successfully saved new dataframe to {}'.format(
    SAVE_PATH+'full_facescrub_with_ids.txt'))

male_train, male_val, male_test = data_split()
female_train, female_val, female_test = data_split(low=265)

train_full = full_frame_with_ids[full_frame_with_ids.person_id.isin(
    np.concatenate((male_train, female_train)))]
val_full = full_frame_with_ids[full_frame_with_ids.person_id.isin(
    np.concatenate((male_val, female_val)))]
test_full = full_frame_with_ids[full_frame_with_ids.person_id.isin(
    np.concatenate((male_test, female_test)))]

print('Training set')
print(train_full.head())
print(train_full.shape)
train_full.to_csv(SAVE_PATH+'train_full_with_ids.txt', sep='\t', index=False)

print('Validation set')
print(val_full.head())
print(val_full.shape)
print('Valid set names', val_full.name.unique())
val_full.to_csv(SAVE_PATH+'val_full_with_ids.txt', sep='\t', index=False)

print('Test set')
print(test_full.head())
print(test_full.shape)
print('Test set names', test_full.name.unique())
test_full.to_csv(SAVE_PATH+'test_full_with_ids.txt', sep='\t', index=False)