"""
Version 1.1.2

Used to read .DDSS files which are useful for monitoring model progress
"""
import pandas as pd
import numpy as np


class Parser:

    @staticmethod
    def parse_ddss(path):

        architecture = {}
        hyperparameters = {}
        history = {}
        time = {}
        info = {'time': time, 'history': history, 'hyperparameters': hyperparameters, 'architecture': architecture}

        with open(path, 'r') as ddss_file:
            lines = ddss_file.readlines()
            lines.remove('\n')

            for i, line in enumerate(lines):
                line = line.strip('\n')

                # Get the model name
                if 'Model mode' in line:
                    info['name'] = line.split()[-1]

                # Get the model timings
                if 'training_time' in line:
                    split_line = line.split()
                    time['training_time'] = float(split_line[-1])
                if 'prediction_time' in line:
                    split_line = line.split()
                    time['prediction_time'] = float(split_line[-1])

                # Get the history stats
                if 'History Stats' in line:
                    #  Grab everything under the history set
                    for sub_line in lines[i + 1:]:  # search the sub lines under history
                        if '-' not in sub_line or 'Model has not been trained yet' in sub_line:
                            break
                        else:  # Split up the lines and stores the values
                            split_line = sub_line.split()[1:]
                            history_key = split_line[0].replace(":", "")

                            value = []
                            for v in split_line[1:]:
                                value.append(float(v.strip(",").strip('[').strip(']')))

                            # If the list has one value, it should be closed to a scalar
                            if len(value) == 1:
                                history[history_key] = value[0]
                            else:
                                history[history_key] = value

                # Get the history stats
                if 'Hyperparameter Stats' in line:
                    # search the sub lines under history
                    for sub_line in lines[i + 1:]:
                        if '-' not in sub_line or 'Model has not been trained yet' in sub_line:
                            break
                        else:
                            sub_line = sub_line.strip(" - ").strip("\n").strip(" ").split(":")
                            key = sub_line[0].strip(" ")
                            value = sub_line[1].strip(" ")

                            if '[' in value:
                                value_list = []
                                for char in value:
                                    if char.isnumeric():
                                        value_list.append(int(char))
                                value = value_list
                            else:
                                try:
                                    value = float(value)
                                except ValueError:
                                    # If this value error occurs, it is because it has found the non-decimal
                                    # hyperparameters
                                    value = value

                            hyperparameters[key] = value

                if 'total_params' in line or 'trainable_params' in line or 'total_params' in line:
                    if "Cannot be determined" not in line:
                        sub_line = line.strip(" - ").strip("\n").strip(" ").split(":")
                        architecture[sub_line[0]] = int(sub_line[1].replace(",", ""))

        return info