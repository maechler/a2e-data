import argparse
import os
from datetime import datetime
from pathlib import Path
import yaml
from argparse import Namespace
import pandas as pd
from pandas import DataFrame
from a2e.data.utility import timestamp_to_date_time, get_recursive_config


class Cleaner:

    def __init__(self, args: Namespace):
        self.args = args
        self.out_folder = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../out',
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

        if self.args.shift is not None:
            self.shift(data_frame, self.args.shift)

        data_frame.set_index(self.get_config('data', 'index_column'), inplace=True)

        if self.args.clip:
            self.clip(data_frame)

        if self.args.dry:
            pass
        else:
            out_path = os.path.join(self.out_folder, Path(self.get_config('data', 'path')).resolve().stem + '.csv')
            print(f'main: writing data to "{str(out_path)}"')
            data_frame.to_csv(out_path, date_format='%s.%f')

        print('main: finished')

    def clip(self, data_frame: DataFrame):
        print('clip: start clipping dataset')
        start_datetime = self.get_config('windows', 'train', 'start')
        end_datetime = self.get_config('windows', 'test_anomalous', 'end')

        data_frame_too_early = data_frame[data_frame.index < start_datetime]
        data_frame_too_late = data_frame[data_frame.index > end_datetime]

        print(f'clip: clipping {len(data_frame_too_early.index)} rows at the beginning of the dataset.')
        data_frame.drop(data_frame_too_early.index, axis=0, inplace=True)

        print(f'clip: clipping {len(data_frame_too_late.index)} rows at the end of the dataset.')
        data_frame.drop(data_frame_too_late.index, axis=0, inplace=True)

        print(f'clip: dataset clipped to {str(data_frame.index[0])} - {str(data_frame.index[-1])}')

    def shift(self, data_frame: DataFrame, target_start):
        print('shift: start dataset shifting')

        data_frame_start_timestamp = data_frame['timestamp'][0]
        target_start_timestamp = datetime.fromisoformat(target_start)
        timestamp_delta = pd.Timedelta(data_frame_start_timestamp - target_start_timestamp)
        index_column = self.get_config('data', 'index_column')

        if timestamp_delta.seconds > 0:
            data_frame[index_column] = data_frame[index_column].apply(lambda x: x - timestamp_delta)
            print(f'shift: dataset start shifted to {str(data_frame["timestamp"][0])}')
        else:
            print('shift: no time shift necessary')

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)

    def to_absolute_path(self, path):
        return os.path.abspath(os.path.join(
            Path(__file__).resolve().parent,
            '../..',
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
