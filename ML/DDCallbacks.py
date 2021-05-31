"""
James Gleave
v1.1.0
"""

from tensorflow.keras.callbacks import Callback
import pandas as pd
import time
import os


class DDLogger(Callback):
    """
    Logs the important data regarding model training
    """

    def __init__(self, log_path,
                 max_time=36000,
                 max_epochs=500,
                 monitoring='val_loss', ):
        super(Callback, self).__init__()
        # Params
        self.max_time = max_time
        self.max_epochs = max_epochs
        self.monitoring = monitoring

        # Stats
        self.epoch_start_time = 0
        self.current_epoch = 0

        # File
        self.log_path = log_path
        self.model_history = {}

    def on_train_begin(self, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        # Store the data
        current_time = time.time()
        epoch_duration = current_time - self.epoch_start_time
        logs['time_per_epoch'] = epoch_duration
        self.model_history["epoch_" + str(epoch + 1)] = logs

        # Estimate time to completion
        estimate, elapsed, (s, p, x) = self.estimate_training_time()
        logs['estimate_time'] = estimate
        logs['time_elapsed'] = elapsed
        self.model_history["epoch_" + str(epoch + 1)] = logs

        # Save the data to a csv
        df = pd.DataFrame(self.model_history)
        df.to_csv(self.log_path)

        print("Time taken calculating callbacks:", time.time()-current_time)

    def estimate_training_time(self):
        max_allotted_time = self.max_time
        max_allotted_epochs = self.max_epochs

        # Grab the info about the model
        model_loss = []
        time_per_epoch = []
        for epoch in self.model_history:
            model_loss.append(self.model_history[epoch]['val_loss'])
            time_per_epoch.append(self.model_history[epoch]['time_per_epoch'])

        time_elapsed = sum(time_per_epoch)
        average_time_per_epoch = sum(time_per_epoch) / len(time_per_epoch)
        current_epoch = len(time_per_epoch)

        # Find out if the model is approaching an early stop
        epochs_until_early_stop = 10
        stopping_vector = []
        prev_loss = model_loss[0]
        for loss in model_loss:
            improved = loss < prev_loss
            stopping_vector.append(improved)
            if improved:
                prev_loss = loss

        # Check how close we are to an early stop
        longest_failure = 0
        for improved in stopping_vector:
            if not improved:
                longest_failure += 1
            else:
                longest_failure = 0

        max_time = max_allotted_epochs * average_time_per_epoch if max_allotted_epochs * average_time_per_epoch < max_allotted_time else max_allotted_time
        time_if_early_stop = (epochs_until_early_stop - longest_failure) * average_time_per_epoch

        # Estimate a completion time
        loss_drops = stopping_vector.count(True)
        loss_gains = len(stopping_vector) - loss_drops
        try:
            gain_drop_ratio = loss_gains / loss_drops
        except ZeroDivisionError:
            gain_drop_ratio = 0

        # Created a function to estimate training time
        power = 1 - (gain_drop_ratio ** 3 / 5)
        time_estimate = (max_time ** power) / (1 + longest_failure)

        # Smooth out the estimate
        if current_epoch > 1:
            last = self.model_history['epoch_{}'.format(current_epoch - 1)]['estimate_time']
            time_estimate = (time_estimate + last) / 2

        # If the time estimate surpasses the max time then just show the max time
        time_for_remaining_epochs = (self.max_epochs - current_epoch) * average_time_per_epoch
        if time_for_remaining_epochs < time_estimate:
            time_estimate = time_for_remaining_epochs

        return time_estimate, time_elapsed, (longest_failure, gain_drop_ratio, max_time)



