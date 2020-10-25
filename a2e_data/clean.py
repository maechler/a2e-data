import argparse
import os
from datetime import datetime
from pathlib import Path
import yaml
from argparse import Namespace
import pandas as pd
from pandas import DataFrame
from a2e_data.utility import timestamp_to_date_time, get_recursive_config


class Cleaner:

    def __init__(self, args: Namespace):
        self.args = args
        self.out_folder = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../out',
            self.__class__.__name__.lower(),
            Path(args.config).resolve().stem,
        ))

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        if os.path.isabs(self.args.config):
            config_file_path = self.args.config
        else:
            config_file_path = self.to_absolute_path(self.args.config)

        with open(config_file_path) as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

    def main(self):
        print('main: loading data')
        data_frame = pd.read_csv(
            self.to_absolute_path(self.get_config('data', 'path')),
            parse_dates=[self.get_config('data', 'index_column')],
            date_parser=timestamp_to_date_time,
            quotechar=self.get_config('data', 'quotechar', default='"'),
            delimiter=self.get_config('data', 'delimiter', default=','),
        )

        self.expand_fft(data_frame)

        if self.args.clip:
            self.clip(data_frame)

        if self.args.shift is not None:
            self.shift(data_frame, self.args.shift)

        if self.args.dry:
            pass
        else:
            data_frame.set_index(self.get_config('data', 'index_column'), inplace=True)
            out_path = os.path.join(self.out_folder, Path(self.get_config('data', 'path')).resolve().stem + '.csv')
            print(f'main: writing data to "{str(out_path)}"')
            data_frame.to_csv(out_path, date_format='%s.%f')

        print('main: finished')

    def expand_fft(self, data_frame: DataFrame):
        if 'fft_magnitude' not in data_frame:
            return

        fft_by_frequency = {}

        for index, row in data_frame.iterrows():
            fft_values = map(float, row['fft_magnitude'].split(','))

            for fft_row_index, fft_value in enumerate(fft_values, start=1):
                fft_key = f'fft_{fft_row_index}'

                if fft_key not in fft_by_frequency:
                    fft_by_frequency[fft_key] = []

                fft_by_frequency[fft_key].append(fft_value)

        del data_frame['fft_magnitude']
        del data_frame['fft_start_frequency']
        del data_frame['fft_end_frequency']

        for frequency in fft_by_frequency:
            data_frame[frequency] = fft_by_frequency[frequency]

    def clip(self, data_frame: DataFrame):
        print('clip: start clipping dataset')
        start_datetime = self.get_config('windows', 'train', 'start')
        end_datetime = self.get_config('windows', 'test_anomalous', 'end')
        index_column = self.get_config('data', 'index_column')

        data_frame_too_early = data_frame[data_frame[index_column] < start_datetime]
        data_frame_too_late = data_frame[data_frame[index_column] > end_datetime]

        print(f'clip: clipping {len(data_frame_too_early.index)} rows at the beginning of the dataset.')
        data_frame.drop(data_frame_too_early.index, axis=0, inplace=True)

        print(f'clip: clipping {len(data_frame_too_late.index)} rows at the end of the dataset.')
        data_frame.drop(data_frame_too_late.index, axis=0, inplace=True)

        data_frame.reset_index(drop=True, inplace=True)

        print(f'clip: dataset clipped to {str(data_frame.iloc[0][index_column])} - {str(data_frame.iloc[-1][index_column])}')

    def shift(self, data_frame: DataFrame, target_start):
        print('shift: start dataset shifting')

        index_column = self.get_config('data', 'index_column')
        data_frame_start_timestamp = data_frame[index_column][0]
        target_start_timestamp = datetime.fromisoformat(target_start)
        timestamp_delta = pd.Timedelta(data_frame_start_timestamp - target_start_timestamp)

        if timestamp_delta.seconds > 0:
            data_frame[index_column] = data_frame[index_column].apply(lambda x: x - timestamp_delta)
            print(f'shift: dataset start shifted to {str(data_frame.iloc[0][index_column])}')
        else:
            print('shift: no time shift necessary')

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    def to_absolute_path(self, path):
        return os.path.abspath(os.path.join(
            Path(__file__).resolve().parent,
            '..',
            path
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Path to a YAML file describing the dataset.', type=str, required=True)
    parser.add_argument('--dry', help='If true, no transformation is actually carried out, but only checked whether the data set is already cleaned.', type=bool, default=False)
    parser.add_argument('--clip', help='If true, the data will be clipped to the specified time windows.', type=bool, default=False)
    parser.add_argument('--shift', help='The desired start datetime for this dataset in ISO format.', type=str, default=None)

    args = parser.parse_args()
    cleaner = Cleaner(args)

    cleaner.main()
