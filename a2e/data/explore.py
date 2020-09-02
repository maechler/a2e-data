import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
import matplotlib.dates as mdates
from argparse import Namespace
from datetime import datetime, timezone
from pandas import DataFrame
import ntpath
from pprint import pprint
from a2e.data.utility import get_recursive_config
from tabulate import tabulate


class Explorer:
    colors = {
        'red': '#D4373E',
        'orange': '#FFA039',
        'green': '#3BCB69',
        'blue': '#3B90C3',
        'purple': '#7D11CD',
        'pink': '#DC72FF'
    }

    labels = {
        'screw_tightened': 'screw tightened'
    }

    def __init__(self, args: Namespace):
        self.args = args
        self.data_frame = None
        self.data_frame_healthy = None
        self.data_frame_anomalous = None
        self.config = None
        self.run_id = ntpath.basename(args.config)
        self.out_folder = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../out',
            self.__class__.__name__.lower(),
            Path(args.config).resolve().stem,
        ))
        self.stats = {}
        self.rolling_window_size = 600
        self.fft_alpha_number_of_samples = 100
        self.stats_data_frame = DataFrame({'overall': [], 'healthy': [], 'anomalous': []})

        with open(self.args.config) as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def run(self):
        self.prepare_data_frames()

        style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretty.mplstyle')
        plt.style.use(style_path)

        self.run_overview()
        self.run_fft_alpha()
        self.run_fft_median()
        self.run_std()
        self.run_mean()
        self.print_stats()

        if args.show_plots:
            plt.show()

    def plot(self, x, y, ylabel, xlabel='time [h]', time_format=True, title=None, ylim=None, color=None, show_screw_tightened=False):
        fig, ax = plt.subplots()
        color = self.colors['blue'] if color is None else color

        if time_format:
            ax.xaxis.set_major_locator(mdates.HourLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(30))

        ax.plot(x, y, color=color)

        if show_screw_tightened:
            anomalous_start = self.get_config('windows', 'test_anomalous', 'start')
            ax.axvline(x=anomalous_start, color=self.colors['red'], linestyle='solid', label=self.labels['screw_tightened'])

        ax.set_xlabel(xlabel, labelpad=15)
        ax.set_ylabel(ylabel, labelpad=15)

        #if title is not None:
        #    ax.set_title(self.get_plot_title(title))

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend()

        self.save_figure(fig, title)

    def run_overview(self):
        self.plot(self.data_frame.index, self.data_frame['rms'], title='RMS', color=self.colors['blue'], ylim=[0, 2], ylabel='RMS', show_screw_tightened=True)
        self.plot(self.data_frame.index, self.data_frame['crest'], title='CREST', color=self.colors['green'], ylim=None, ylabel='CREST', show_screw_tightened=True)
        self.plot(self.data_frame.index, self.data_frame['temperature'], title='Temperature', color=self.colors['orange'], ylim=None, ylabel='Temperature [°C]', show_screw_tightened=True)
        self.plot(self.data_frame.index, self.data_frame['rpm'], title='RPM', color=self.colors['purple'], ylim=None, ylabel='RPM', show_screw_tightened=True)

    def get_plot_title(self, plot_title):
        title = self.get_config('plot', 'title')

        return f'{title} - {plot_title}'

    def prepare_data_frames(self):
        parse_date_time = lambda d: datetime.fromtimestamp(float(d), tz=timezone.utc)
        csv_file_path = self.get_config('data', 'path')
        index_column = self.get_config('data', 'index_column')

        data_frame = pd.read_csv(csv_file_path, parse_dates=[index_column], date_parser=parse_date_time, quotechar='"', delimiter=',', sep=',')
        data_frame.set_index(index_column, inplace=True)

        healthy_start = self.get_config('windows', 'train', 'start')
        healthy_end = self.get_config('windows', 'train', 'end')
        healthy_mask = (data_frame.index > healthy_start) & (data_frame.index <= healthy_end)

        anomalous_start = self.get_config('windows', 'test_anomalous', 'start')
        anomalous_end = self.get_config('windows', 'test_anomalous', 'end')
        anomalous_mask = (data_frame.index > anomalous_start) & (data_frame.index <= anomalous_end)

        self.data_frame = data_frame
        self.data_frame_healthy = data_frame.loc[healthy_mask]
        self.data_frame_anomalous = data_frame.loc[anomalous_mask]

    def run_fft_alpha(self):
        plot_config = [
            {
                'data_frame': self.data_frame_healthy,
                'title': 'healthy'
            },
            {
                'data_frame': self.data_frame_anomalous,
                'title': 'anomalous'
            },
        ]

        for config in plot_config:
            data_frame = config['data_frame']
            fig, ax = plt.subplots()

            number_of_samples = self.fft_alpha_number_of_samples
            data_frame_size = len(data_frame.index)
            number_of_samples = data_frame_size if number_of_samples == -1 else number_of_samples
            alpha = (1 / number_of_samples)
            ffts = []

            for start in range(0, number_of_samples):
                df_subset = data_frame.iloc[start]
                fft = list(map(float, (df_subset['fft_magnitude'].split(','))))
                x = list(range(len(fft)))
                ffts.append(fft)

                ax.plot(x, fft, alpha=alpha, color=self.colors['green'])
                #ax.set_xlim([xmin, xmax])
                #ax.set_ylim([0, 1.2])

            if self.args.save_plots:
                self.save_figure(fig, 'fft_alpha_' + config['title'])

    def run_fft_median(self):
        plot_config = [
            {
                'data_frame': self.data_frame_healthy,
                'title': 'healthy'
            },
            {
                'data_frame': self.data_frame_anomalous,
                'title': 'anomalous'
            },
        ]

        for config in plot_config:
            data_frame = config['data_frame']
            title = config['title']
            fft_list = []

            for index, row in data_frame.iterrows():
                fft = list(map(float, (row['fft_magnitude'].split(','))))
                fft_list.append(fft)

            fft_data_frame = DataFrame(fft_list)

            plot_data_frame = DataFrame()
            plot_data_frame['fft_median'] = fft_data_frame.median(axis=0)

            self.plot(plot_data_frame.index, plot_data_frame['fft_median'], xlabel='Frequency [Hz]', ylabel='Amplitude', title=f'FFT median ({title})', time_format=False)

    def run_std(self):
        columns = ['rms', 'crest']

        for column in columns:
            self.stats_data_frame.loc[f'std_{column}', 'overall'] = self.data_frame[column].std()
            self.stats_data_frame.loc[f'std_{column}', 'healthy'] = self.data_frame_healthy[column].std()
            self.stats_data_frame.loc[f'std_{column}', 'anomalous'] = self.data_frame_anomalous[column].std()

            self.data_frame[f'rolling_std_{column}'] = self.data_frame[column].rolling(window=self.rolling_window_size).std()

            self.plot(self.data_frame.index, self.data_frame[f'rolling_std_{column}'], ylabel=column, title=f'Standard Deviation {column}', show_screw_tightened=True)

    def run_mean(self):
        columns = ['rms', 'crest']

        for column in columns:
            self.stats_data_frame.loc[f'mean_{column}', 'overall'] = self.data_frame[column].mean()
            self.stats_data_frame.loc[f'mean_{column}', 'healthy'] = self.data_frame_healthy[column].mean()
            self.stats_data_frame.loc[f'mean_{column}', 'anomalous'] = self.data_frame_anomalous[column].mean()

            self.data_frame[f'rolling_mean_{column}'] = self.data_frame[column].rolling(window=self.rolling_window_size).mean()

            self.plot(self.data_frame.index, self.data_frame[f'rolling_mean_{column}'], ylabel=column, title=f'Mean {column}', show_screw_tightened=True)

    def save_figure(self, figure, filename):
        sanitized_filename = filename.lower().replace(' ', '_')

        figure.savefig(os.path.join(self.out_folder, sanitized_filename + '.pdf'), format='pdf')
        figure.savefig(os.path.join(self.out_folder, sanitized_filename + '.png'), format='png')

    def print_stats(self):
        print(tabulate(self.stats_data_frame, headers='keys', tablefmt='psql'))

        self.stats_data_frame.to_csv(os.path.join(self.out_folder, 'stats.csv'))

    def save_stats(self):
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Path to a YAML file describing the dataset.', type=str, required=True)
    parser.add_argument('--show_plots', help='Whether to show plots or not.', type=bool, default=True)
    parser.add_argument('--save_plots', help='Whether to save plots to file system or not.', type=bool, default=True)

    args = parser.parse_args()
    explorer = Explorer(args)

    explorer.run()