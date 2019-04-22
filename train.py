import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
# from keras import models
# from keras import layers
import sys
import json

current_dir = Path(__file__).resolve().parent

class Model:
    def train(self, X, Y):
        X = X.values
        Y = Y.values
        seed = 7
        np.random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

        model = models.Sequential()
        # Input Layer
        model.add(layers.Dense(50, activation='relu', input_shape=(X.shape[1],)))

        # Hidden Layers
        model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
        model.add(layers.Dense(100,activation='relu'))
        model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
        model.add(layers.Dense(100, activation='relu'))

        # Output Layer
        model.add(layers.Dense(Y.shape[1],  activation='sigmoid'))
        model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

        results = model.fit(X_train, y_train, epochs = 2, batch_size = 100, validation_data=(X_test, y_test))
        print("Test Accuracy: ", np.mean(results.history["val_acc"]))
        return model

class Encoder:
    def create_max_df(self, df, ignore_columns):
        columns = df.columns.values
        columns = columns.tolist()
        for col in ignore_columns:
            columns.remove(col)
        max_df = pd.DataFrame(0, index=['Max','Min'], columns=columns)
        return max_df

    def encode_repertoire(self, grouped_data, pitch_types):
        pitch_types = np.insert(pitch_types, 0, 'pitcherId', axis=0)
        repertoire_df = pd.DataFrame(0, index=np.arange(len(grouped_data)), columns=pitch_types)
        repertoire_df['pitcherId'] = grouped_data.groups
        repertoire_df.set_index('pitcherId', inplace=True)
        repertoire_df.dropna()
     
        for pitcher_id, group in grouped_data:
            pitcher_repertoire = group['pitchType'].unique()
            pitcher_repertoire = pitcher_repertoire[~pd.isnull(pitcher_repertoire)]
            for pitch in pitcher_repertoire:
                repertoire_df.at[pitcher_id,pitch] = 1
                


        repertoire_df.to_csv('out_data/out_repertoire.csv')
        return repertoire_df

    def normalize_column(self, dataframe, column, max_df):
        max_value = dataframe[column].max()
        max_df.at['Max',column] = max_value

        min_value = dataframe[column].min()
        max_df.at['Min',column] = min_value

        dataframe[column] = (dataframe[column] - min_value) / (max_value - min_value)
        return dataframe

    def build_map(self, column_array, dataframe, file_name):
        outcome_map = games_df[column_array]
        outcome_map = outcome_map.drop_duplicates()
        outcome_map.set_index(column_array[0], inplace=True)
        outcome_map.dropna(inplace=True)
        outcome_map.to_csv('out_data/{}'.format(file_name))

    def encode_game_events(self, game_data, repertoire_df, pitch_types, post_season_split_index):
        
        x_columns = ['pitcherId','balls','strikes',
        'outs','inningHalfEventSequenceNumber','atBatEventSequenceNumber',
        'hitterBatHand','hitterHeight','hitterWeight',
        'pitcherThrowHand','pitcherPitchCount','hitterPitchCount',
        'homeCurrentTotalRuns', 'awayCurrentTotalRuns','inningNumber','inningHalf']
        y_columns = ['outcomeId', 'pitchType', 'pitchZone']

        encode_columns = ['pitcherThrowHand', 'hitterBatHand', 'inningHalf']

        # encoding(0, 1): batterHand, pitcherHand: L, R;
        all_columns = np.insert(x_columns, 0, y_columns, axis=0)
        collapsed = game_data[all_columns]
        collapsed = collapsed.dropna()

        X = collapsed[x_columns]
        for_dummies = X[encode_columns]
        max_df = self.create_max_df(X, encode_columns)
        dummies = pd.get_dummies(for_dummies)
        X.drop(columns=encode_columns, inplace=True)
        X = pd.DataFrame(X.join(dummies))
        # Crear DataFrame con Máximos y Mínimos
    
        for col in x_columns:
            if col not in encode_columns and col != 'pitcherId':
                self.normalize_column(X, col, max_df)

        max_df.dropna(inplace=True)
        max_df.to_csv('out_data/min_max.csv')
        Y = collapsed[y_columns]
        Y['pitchZone'] = Y['pitchZone'].astype(str)
        Y = pd.get_dummies(Y)
        split_index = Y.shape[0] - post_season_split_index
        Y, post_season_y = np.split(Y, [split_index], axis = 0)

        repertoire_cols = pd.DataFrame(0, index=np.arange(len(X)), columns=list(repertoire_df.columns.values))
        repertoire_cols['pitcherId'] = X['pitcherId']
        repertoire_cols.set_index('pitcherId', inplace=True)
    
        with_rep = pd.merge(X, repertoire_df, on='pitcherId')
        with_rep, post_season_x = np.split(with_rep, [split_index], axis = 0)
        with_rep.drop(columns='pitcherId', inplace=True)
        with_rep.to_csv('out_data/game_events_x.csv')
        post_season_x.to_csv('post_season/events_x.csv')
        post_season_y.to_csv('post_season/events_y.csv')
        Y.to_csv('out_data/game_events_y.csv')

        return {'X': with_rep, 'Y': Y, 'PostX': post_season_x, 'PostY': post_season_y}

start_time = datetime.datetime.now()
print(start_time)

columns = ['gameId']

csv_cols = {
    'gameId': str,
    'startTime': str,
    'attendance': int,
    'dayNight': str,
    'durationMinutes':int,
    'awayTeamName': str,
    'homeTeamName': str,
    'venueName':str,
    'venueSurface':str,
    'venueCity':str,
    'venueState':str,
    'homeFinalRuns':int,
    'homeFinalHits':int,
    'homeFinalErrors':int,
    'awayFinalRuns':int,
    'awayFinalHits':int,
    'awayFinalErrors':int,
    'inningNumber':int,
    'inningHalf':str,
    'inningEventType':str,
    'inningHalfEventSequenceNumber':int,
    'atBatEventType':str,
    'atBatEventSequenceNumber':int,
    'outcomeId':str,
    'outcomeDescription': str,
    'hitterId':str,
    'hitterWeight':int,
    'hitterHeight':int,
    'hitterBatHand':str,
    'pitcherId':str,
    'pitcherThrowHand':str,
    'pitchType':str,
    'pitchTypeDescription':str,
    'pitchSpeed':int,
    'pitchZone':int,
    'pitcherPitchCount':int,
    'hitterPitchCount':int,
    'hitLocation':int,
    'hitType':str,
    'pitcherFirstName':str,
    'pitcherLastName':str,
    'startingBalls':int,
    'startingStrikes':int,
    'startingOuts':int,
    'balls':int,
    'strikes':int,
    'outs':int,
    'rob0_start':str,
    'rob0_end':int,
    'rob1_start':str,
    'rob1_end':int,
    'rob2_start':str,
    'rob2_end':int,
    'rob3_start':str,
    'rob3_end':int,
    'is_ab':int,
    'is_ab_over':int,
    'is_hit':int,
    'is_on_base':int,
    'is_bunt':int,
    'is_bunt_shown':int,
    'is_double_play':int,
    'is_triple_play':int,
    'is_wild_pitch':int,
    'is_passed_ball':int,
    'homeCurrentTotalRuns':int,
    'awayCurrentTotalRuns':int
}

file_path = 'season_data/mydata000000000000.csv'

games_df = pd.read_csv(file_path, usecols=csv_cols, dtype=csv_cols)
# Populate CSV

for i in range(2):
    if i > 0:
        data_path = 'season_data/mydata00000000000{}.csv'.format(i)
        current_df = pd.read_csv(data_path, usecols=csv_cols, dtype=csv_cols)
        games_df = games_df.append(current_df, ignore_index=True)

post_games_data_path = 'season_data/post_season_data.csv'
post_games_df = pd.read_csv(post_games_data_path, usecols=csv_cols, dtype=csv_cols)
post_games_df = games_df.sort_values(
    by=['gameId','inningNumber','inningHalf','inningHalfEventSequenceNumber','atBatEventSequenceNumber'],
    ascending=[1,1,0,1,1])

cut_y = post_games_df.shape[0]

games_df = games_df.sort_values(
    by=['gameId','inningNumber','inningHalf','inningHalfEventSequenceNumber','atBatEventSequenceNumber'],
    ascending=[1,1,0,1,1])


games_df = games_df.append(post_games_df, ignore_index = True)

encoder = Encoder()
encoder.build_map(['outcomeId', 'outcomeDescription'], games_df, 'outcome_map.csv')
encoder.build_map(['pitcherId', 'pitcherFirstName', 'pitcherLastName'], games_df, 'pitcher_name_map.csv')
encoder.build_map(['pitchType', 'pitchTypeDescription'], games_df, 'pitch_type_map.csv')

curr_time = datetime.datetime.now()
total_time = curr_time - start_time
print(total_time)

pitchers_groups = games_df.groupby('pitcherId')
pitch_types = games_df['pitchType'].unique()
pitch_types = pitch_types[~pd.isnull(pitch_types)]

exp_data_path = 'season_data/out_data.csv'

# repertoire_df = encoder.encode_repertoire(pitchers_groups, pitch_types)
repertoire_df = pd.read_csv('out_data/out_repertoire.csv')
repertoire_df.set_index('pitcherId', inplace=True)
encoded_events_data = encoder.encode_game_events(games_df, repertoire_df, pitch_types, cut_y)
sys.exit()

# Build Model
# model = Model()
# model = model.train(encoded_events_data['X'], encoded_events_data['Y'])
# model_json = model.to_json()
# model.save_weights('trained_mlb.h5')
# model_file = open('model.json', 'w')
# model_file.write(model_json)
# model_file.close()