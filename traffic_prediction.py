import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

class TrafficPrediction:
    
    def read_from_csv(self, filename, sample_period):
        self.df = pd.read_csv(filename)
        self.df['ds'] = pd.to_datetime(self.df['ds'], unit='s')
        self.df.set_index('ds', inplace=True)
        self.df = self.df.resample(sample_period).sum(numeric_only=True).fillna(0)
        self.df['y'] /= pd.Timedelta(sample_period).total_seconds()
        self.df['y'] *= 8
        self.df['y'] /= 2**20  # Convert to Megabytes
    
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset.iloc[i] - dataset.iloc[i - interval]
            diff.append(value)
        return pd.Series(diff, index=dataset.index[interval:])
    
    def run_arima(self, training_split=.8):
        split_index = int(training_split * len(self.df))
        self.training_data = self.df.iloc[:split_index].copy()
        self.testing_data = self.df.iloc[split_index:].copy()

        # Apply differencing to make the data stationary
        self.training_data['y_diff'] = self.difference(self.training_data['y'])
        self.testing_data['y_diff'] = self.difference(self.testing_data['y'])

        # Drop NaN values created by differencing
        self.training_data.dropna(inplace=True)
        self.testing_data.dropna(inplace=True)

        if len(self.training_data) < 1 or len(self.testing_data) < 1:
            raise ValueError("Not enough data points after differencing")

        # Automatically determine the best ARIMA parameters
        model = auto_arima(self.training_data['y_diff'], seasonal=False, trace=True)
        fitted_model = model.fit(self.training_data['y_diff'])
        
        self.prediction_diff = fitted_model.predict(n_periods=len(self.testing_data))

        # Invert the differencing to get the actual predicted values
        self.prediction = pd.Series(self.prediction_diff, index=self.testing_data.index).cumsum() + self.training_data['y'].iloc[-1]

        # Ensure all predictions are non-negative
        self.prediction = self.prediction.clip(lower=0)

    def plot(self, ax):
        ax.plot(self.training_data.index, self.training_data['y'], label="Training Data", linestyle='dotted', linewidth=2)
        ax.plot(self.testing_data.index, self.testing_data['y'], label="Actual Data", color='orange', linewidth=2)
        ax.plot(self.prediction.index, self.prediction, label="Prediction", color='green', linewidth=2)
        ax.axvline(x=self.training_data.index.max(), color='red', linestyle='--', label='Training Split', linewidth=2)
        ax.legend(fontsize=12)
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Traffic', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)

        # Add y-axis tick labels
        y_max = max(self.training_data['y'].max(), self.testing_data['y'].max(), self.prediction.max())
        ax.set_yticks([i for i in range(0, math.ceil(y_max) + 1)])  # Ensure the y-axis ticks are integers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network traffic prediction script")
    parser.add_argument('--csv', type=str, default="captures", help="Folder containing the csv files to use as input")
    parser.add_argument('--store-plot', type=str, default="plots", help="Folder where to store the plots")
    parser.add_argument('--training-split', type=float, default=.8, help="Percentage of data used for training")
    parser.add_argument('--sample-period', type=str, default="500L", help="Period over which to combine network data")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.store_plot):
        os.mkdir(args.store_plot)
    
    path = args.csv
    for switch in os.listdir(path):
        intf_csv = [file for file in os.listdir(os.path.join(path, switch)) if file.endswith('.csv')]
        
        plot_count = len(intf_csv)
        
        if plot_count == 0:
            print(f"No .csv file found in {switch} folder")
            continue

        num_cols = math.ceil(math.sqrt(plot_count))
        num_rows = math.ceil(plot_count / num_cols)
        
        fig, axs = plt.subplots(num_rows, num_cols, sharey=True, sharex=True, figsize=(20, 15))  # Increased figure size
        axs = axs.flatten()
        
        for ax, interface in zip(axs, intf_csv):
            prediction = TrafficPrediction()
            full_path = os.path.join(path, switch, interface)
            
            print(f"Reading file {full_path}")
            prediction.read_from_csv(full_path, args.sample_period)
            
            print("Running ARIMA prediction...")
            try:
                prediction.run_arima(training_split=args.training_split)
                prediction.plot(ax)
                ax.set_title(interface[:-4], fontsize=16, fontweight='bold')  # Increased title font size and bold
            except ValueError as e:
                print(f"Skipping {interface} due to insufficient data: {e}")

        for ax in axs[plot_count:]:
            ax.axis('off')
        
        fig.suptitle(switch, fontsize=22, fontweight='bold')  # Increased figure title font size and bold
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)  # Adjust layout to fit the title
        plt.savefig(os.path.join(args.store_plot, switch) + '.png', dpi=300)  # Increased dpi for better quality
        plt.show()

