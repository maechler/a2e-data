import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
import matplotlib.dates as mdates
from argparse import Namespace
from datetime import datetime, timezone, timedelta
from pandas import DataFrame
import ntpath
from a2e_data.utility import get_recursive_config, str2bool
from tabulate import tabulate


mpl.rcParams['agg.path.chunksize'] = 10000


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
        self.windows = {}
        self.config = None
        self.run_id = ntpath.basename(args.config)
        self.out_folder = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../out',
            self.__class__.__name__.lower(),
            Path(args.config).resolve().stem,
        ))
        self.stats = {}
        self.rolling_window_size = 600
        self.fft_alpha_number_of_samples = 100
        self.stats_data_frame = DataFrame()
        self.show_plots = self.args.show_plots

        with open(self.args.config) as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def run(self):
        self.prepare_data_frames()

        style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretty.mplstyle')
        plt.style.use(style_path)

        self.run_overview()
        self.run_closeup()
        self.run_fft_median()
        self.run_std()
        self.run_mean()
        self.run_data_collection_stats()
        self.print_stats()

        if self.show_plots:
            plt.show()

    def plot(self, x, y, ylabel, xlabel='time [h]', time_format=True, title=None, xlim=None, ylim=None, color=None, show_screw_tightened=False, filename=None):
        fig, ax = plt.subplots()
        color = self.colors['blue'] if color is None else color

        if time_format:
            ax.xaxis.set_major_locator(mdates.HourLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(30))

        ax.plot(x, y, color=color)

        if show_screw_tightened:
            anomalous_start = self.get_config('windows', 'test_anomalous', 'start')

            if x[0] < anomalous_start < x[-1]:
                ax.axvline(x=anomalous_start, color=self.colors['red'], linestyle='solid', label=self.labels['screw_tightened'])

        ax.set_xlabel(xlabel, labelpad=15)
        ax.set_ylabel(ylabel, labelpad=15)

        if title is not None:
            ax.set_title(self.get_plot_title(title))

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend()

        self.save_figure(fig, title if filename is None else filename)

        if not self.show_plots:
            plt.close()

    def run_overview(self):
        self.plot(
            self.data_frame.index,
            self.data_frame['rms'],
            title='RMS',
            color=self.colors['blue'],
            ylim=self.get_config('plot', 'rms', 'ylim', default=None),
            ylabel='RMS',
            show_screw_tightened=True
        )

        self.plot(
            self.data_frame.index,
            self.data_frame['crest'],
            title='CREST',
            color=self.colors['green'],
            ylim=self.get_config('plot', 'crest', 'ylim', default=None),
            ylabel='CREST',
            show_screw_tightened=True
        )

        self.plot(
            self.data_frame.index,
            self.data_frame['temperature'],
            title='Temperature',
            color=self.colors['orange'],
            ylim=self.get_config('plot', 'temperature', 'ylim', default=None),
            ylabel='Temperature [°C]',
            show_screw_tightened=True
        )

        self.plot(
            self.data_frame.index,
            self.data_frame['rpm'],
            title='RPM',
            color=self.colors['purple'],
            ylim=self.get_config('plot', 'rpm', 'ylim', default=None),
            ylabel='RPM',
            show_screw_tightened=True
        )

    def run_closeup(self):
        anomalous_start = self.get_config('windows', 'test_anomalous', 'start')
        xlim_start = anomalous_start - timedelta(minutes=1)
        xlim_end = anomalous_start + timedelta(minutes=1)

        self.plot(
            self.data_frame.index,
            self.data_frame['rms'],
            title='RMS (close)',
            color=self.colors['blue'],
            xlim=[xlim_start, xlim_end],
            ylim=self.get_config('plot', 'rms', 'ylim', default=None),
            ylabel='RMS',
            show_screw_tightened=True,
        )

        self.plot(
            self.data_frame.index,
            self.data_frame['crest'],
            title='CREST (close)',
            color=self.colors['green'],
            xlim=[xlim_start, xlim_end],
            ylabel='CREST',
            show_screw_tightened=True
        )

    def get_plot_title(self, plot_title):
        title = self.get_config('plot', 'title')

        return f'{title} - {plot_title}'

    def prepare_data_frames(self):
        parse_date_time = lambda d: datetime.fromtimestamp(float(d), tz=timezone.utc)
        csv_file_path = self.get_config('data', 'path')
        index_column = self.get_config('data', 'index_column')

        self.data_frame = pd.read_csv(csv_file_path, parse_dates=[index_column], date_parser=parse_date_time, quotechar='"', delimiter=',', sep=',')
        self.data_frame.set_index(index_column, inplace=True)

        for window_key, window_config in self.get_config('windows', default={}).items():
            window_start = window_config['start']
            window_end = window_config['end']
            window_mask = (self.data_frame.index > window_start) & (self.data_frame.index <= window_end)

            self.windows[window_key] = self.data_frame.loc[window_mask]

    def run_fft_median(self):
        for window_key, window_data_frame in self.windows.items():
            data_frame_start = window_data_frame.index[0]
            data_frame_end = window_data_frame.index[-1]
            data_frame_splits = 4
            data_frame_delta = (data_frame_end - data_frame_start) / data_frame_splits
            plot_config = [
                {
                    'data_frame': window_data_frame,
                    'title': f'{window_key}-fft-all'
                },
            ]

            for i in range(0, data_frame_splits):
                plot_config.append({
                    'data_frame': window_data_frame.loc[data_frame_start + i*data_frame_delta:data_frame_start + (i+1)*data_frame_delta],
                    'title': f'{window_key}-fft-{i}',
                })

            for rpm in self.get_config('plot', 'fft', 'rpm_plots', default=[]):
                plot_config.append({
                    'data_frame': window_data_frame[window_data_frame.rpm == rpm],
                    'title': f'{window_key}-fft-{rpm}-rpm',
                })
                plot_config.append({
                    'data_frame': window_data_frame[window_data_frame.rpm == rpm],
                    'title': f'{window_key}-fft-{rpm}-rpm',
                })

            for config in plot_config:
                data_frame = config['data_frame']
                title = config['title']
                fft_list = []

                for index, row in data_frame.iloc[:, 4:].iterrows():
                    fft = list(row)
                    fft_list.append(fft)

                fft_data_frame = DataFrame(fft_list)

                plot_data_frame = DataFrame()
                plot_data_frame['fft_median'] = fft_data_frame.median(axis=0)

                self.plot(
                    plot_data_frame.index,
                    plot_data_frame['fft_median'],
                    xlabel='Frequency [Hz]',
                    ylabel='Amplitude',
                    title=f'FFT median ({title})',
                    ylim=self.get_config('plot', 'fft', 'ylim', default=None),
                    time_format=False,
                    filename=f'{window_key}/{title}',
                )

    def run_std(self):
        columns = ['rms', 'crest']

        for window_key, window_config in self.windows.items():
            for column in columns:
                self.stats_data_frame.loc[f'std_{column}', window_key] = self.windows[window_key][column].std()

                self.windows[window_key][f'rolling_std_{column}'] = self.windows[window_key][column].rolling(window=self.rolling_window_size).std()

                self.plot(self.windows[window_key].index, self.windows[window_key][f'rolling_std_{column}'], ylabel=column, title=f'Standard Deviation {window_key}:{column}', show_screw_tightened=True)

    def run_mean(self):
        columns = ['rms', 'crest']

        for window_key, window_config in self.windows.items():
            for column in columns:
                self.stats_data_frame.loc[f'mean_{column}', window_key] = self.windows[window_key][column].mean()

                self.windows[window_key][f'rolling_mean_{column}'] = self.windows[window_key][column].rolling(window=self.rolling_window_size).mean()

                self.plot(self.windows[window_key].index, self.windows[window_key][f'rolling_mean_{column}'], ylabel=column, title=f'Mean {window_key}:{column}', show_screw_tightened=True)

    def run_data_collection_stats(self):
        total_values = len(self.data_frame.index)
        start_date = self.data_frame.index[0]
        end_date = self.data_frame.index[-1]
        total_seconds = (end_date - start_date).total_seconds()
        computed_frequency = "{:.2f}Hz".format(total_values / total_seconds)

        grouped_rms_values = len(self.data_frame.groupby(['rms']))
        duplicated_rms_rows = total_values - grouped_rms_values
        duplicated_rms_percentage = "{:.2f}%".format((duplicated_rms_rows / total_values) * 100)

        grouped_crest_values = len(self.data_frame.groupby(['crest']))
        duplicated_crest_rows = total_values - grouped_crest_values
        duplicated_crest_percentage = "{:.2f}%".format((duplicated_crest_rows / total_values) * 100)

        grouped_fft_values = len(self.data_frame.groupby(['fft_1', 'fft_2', 'fft_3']))
        duplicated_fft_rows = total_values - grouped_fft_values
        duplicated_fft_percentage = "{:.2f}%".format((duplicated_fft_rows / total_values) * 100)

        with open(os.path.join(self.out_folder, 'data_collection_stats.txt'), 'w') as file:
            file.write(f'Total seconds: {total_seconds} \n')
            file.write(f'Total rows: {total_values} \n')
            file.write(f'Computed ferquency: {computed_frequency} \n')
            file.write(f'Duplicated RMS rows: {duplicated_rms_rows} \n')
            file.write(f'Duplicated RMS percentage: {duplicated_rms_percentage} \n')
            file.write(f'Duplicated CREST rows: {duplicated_crest_rows} \n')
            file.write(f'Duplicated CREST percentage: {duplicated_crest_percentage} \n')
            file.write(f'Duplicated FFT rows: {duplicated_fft_rows} \n')
            file.write(f'Duplicated FFT percentage: {duplicated_fft_percentage} \n')

    def save_figure(self, figure, filename):
        sanitized_filename = filename.lower().replace(' ', '_').replace('(', '').replace(')', '')
        out_path = os.path.join(self.out_folder, sanitized_filename + '.png')
        out_dir = os.path.dirname(out_path)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        figure.savefig(out_path, format='png')

    def print_stats(self):
        print(tabulate(self.stats_data_frame, headers='keys', tablefmt='psql'))

        self.stats_data_frame.to_csv(os.path.join(self.out_folder, 'computed_stats.csv'))

    def save_stats(self):
        pass

    def get_config(self, *args, **kwargs):
        return get_recursive_config(self.config, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Path to a YAML file describing the dataset.', type=str, required=True)
    parser.add_argument('--show_plots', help='Whether to show plots or not.', type=str2bool, default=True)
    parser.add_argument('--save_plots', help='Whether to save plots to file system or not.', type=str2bool, default=True)

    args = parser.parse_args()
    explorer = Explorer(args)

    explorer.run()
