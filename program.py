import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras import models
from keras.models import model_from_json
from keras import layers
import sys
import json

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

class InputEncoder:
    def decode_row(self,row):
        cols = ['outcomeId', 'pitchType', 'pitchZone']
        returnObj = []
        

class Program:
    def __init__(self):
        self.post_season_x = pd.read_csv('post_season/events_x.csv')
        self.post_season_y = pd.read_csv('post_season/events_y.csv')
        model_json = open('model.json', 'r')
        loaded_json = model_json.read()
        model_json.close()
        self.loaded_model = model_from_json(loaded_json)
        self.loaded_model.load_weights('trained_mlb.h5')
        self.pitcher_repertoires = pd.read_csv('out_data/out_repertoire.csv')
        self.max_df = pd.read_csv('out_data/min_max.csv')

    def predict_some(self, n_predictions):
        encoder = InputEncoder()
        x = self.post_season_x
        y = self.post_season_y
        condensed=x.append(y,axis=1, ignore_index=True)
        samples = condensed.sample(n=n_predictions)
        x_pred = samples[x.columns]
        x_pred.drop('pitcherId')
        x_pred = x_pred.values
        y_pred = samples[y.columns]
        prediction = self.loaded_model.predict(x_pred)
        return {'X': x_pred, 'Y': y_pred, 'Pred': prediction}

    def predict_all(self):
        return 0 

    def predict_row(self, row_data):
        encoder = InputEncoder()
        x, y = encoder.encode_for_prediction(row_data)
        pred = self.model.predict(x)
        x, y, pred = encoder.encode_for_presentation(x, y ,pred)
        return {'X': x, 'Y':y, 'Pred': p}

program = Program()
program.predict_some(1)