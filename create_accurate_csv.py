#%%
import pandas as pd
import os
print('Imported')
#%%
def remove_missing_files(df, data_path):
    count = 0
    indices = []
    for index, row in df.iterrows():
        name = row['name'].replace(' ', '_')
        img_id = str(row['image_id'])

        img_name = name + '_' + img_id + '.jpeg'
        img_path = os.path.join(data_path, 'images', name, img_name)
        if os.path.isfile(img_path):
            continue
        else:
            count += 1
            indices.append(index)

    return df.drop(actors_frame.index[indices]), count

ANNOT_ACTORS_PATH = "data/facescrub_actors.txt"
ANNOT_ACTRESS_PATH = "data/facescrub_actresses.txt"

DATA_ACTORS_PATH = "data/actor/"
DATA_ACTRESS_PATH = "data/actress/"

SAVE_PATH = "data/"

actors_frame = pd.read_csv(ANNOT_ACTORS_PATH, delimiter='\t')
actors_frame['gender'] = 'male'

print('Before deletion, actors frame')
print(actors_frame.head())

updated_actors_frame, actors_count = remove_missing_files(actors_frame, DATA_ACTORS_PATH)

print('After deletion, actors frame')
print(updated_actors_frame.head())
print(updated_actors_frame.shape)
print('{} lines were deleted'.format(actors_count))

#%%
actress_frame = pd.read_csv(ANNOT_ACTRESS_PATH, delimiter='\t')
actress_frame['gender'] = 'female'

print('Before deletion, actress frame')
print(actress_frame.head())

updated_actress_frame, actress_count = remove_missing_files(actress_frame, DATA_ACTRESS_PATH)

print('After deletion, actress frame')
print(updated_actress_frame.head())
print(updated_actress_frame.shape)
print('{} lines were deleted'.format(actress_count))

new_full_frame = updated_actors_frame.append(updated_actress_frame, ignore_index=True)

print(new_full_frame.head())
print(new_full_frame.tail())
print(new_full_frame.shape)

new_full_frame.to_csv(SAVE_PATH+'full_facescrub.txt', sep='\t', index=False)
print('Successfully saved new dataframe to {}'.format(SAVE_PATH+'full_facescrub.txt'))