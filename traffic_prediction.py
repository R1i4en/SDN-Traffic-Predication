import os
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

class TrafficPrediction:
    
    def read_from_csv(self, filename, sample_period):
        self.df = pd.read_csv(filename)
        self.df['ds'] = pd.to_datetime(self.df['ds'], unit='s')
        self.df.set_index('ds', inplace=True)
        sample_period = '200L'  # Use 200 milliseconds as the sample period
        self.df = self.df.resample(sample_period).sum(numeric_only=True).fillna(0)
        self.df['y'] /= pd.Timedelta(sample_period).total_seconds()
        self.df['y'] *= 8
        self.df['y'] /= 2**20  # Convert to Megabytes
        self.df['y'] = self.df['y'].clip(lower=0)  # Ensure no negative values
    
    def run_auto_arima(self, training_split=.8, seasonal=True):
        split_index = int(training_split * len(self.df))
        self.training_data = self.df.iloc[:split_index].copy()
        self.testing_data = self.df.iloc[split_index:].copy()

        print("Training data head:\n", self.training_data.head())
        print("Testing data head:\n", self.testing_data.head())

        if len(self.training_data) < 1 or len(self.testing_data) < 1:
            raise ValueError("Not enough data points for training/testing split")

        # Automatically determine the best ARIMA parameters
        model = auto_arima(self.training_data['y'], seasonal=seasonal, trace=True, error_action='ignore')
        print("Best ARIMA parameters:", model.order, "Seasonal order:", model.seasonal_order)
        fitted_model = model.fit(self.training_data['y'])
        
        self.prediction = fitted_model.predict(n_periods=len(self.testing_data))
        self.prediction = pd.Series(self.prediction, index=self.testing_data.index)
        self.prediction = self.prediction.clip(lower=0)  # Ensure no negative values
        print("Predicted values:\n", self.prediction)

    def run_sarima(self, order, seasonal_order, training_split=.8):
        split_index = int(training_split * len(self.df))
        self.training_data = self.df.iloc[:split_index].copy()
        self.testing_data = self.df.iloc[split_index:].copy()

        print("Training data head:\n", self.training_data.head())
        print("Testing data head:\n", self.testing_data.head())

        if len(self.training_data) < 1 or len(self.testing_data) < 1:
            raise ValueError("Not enough data points for training/testing split")

        # Fit SARIMA model
        model = SARIMAX(self.training_data['y'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        
        self.prediction = fitted_model.predict(start=len(self.training_data), end=len(self.training_data) + len(self.testing_data) - 1, dynamic=False)
        self.prediction = pd.Series(self.prediction, index=self.testing_data.index)
        self.prediction = self.prediction.clip(lower=0)  # Ensure no negative values
        print("Predicted values:\n", self.prediction)

    def plot(self, ax):
        ax.plot(self.training_data['y'], label="Training Data", linestyle='dotted', linewidth=2)
        ax.plot(self.testing_data['y'], label="Actual Data", linewidth=2)
        ax.plot(self.prediction, label="Prediction", linewidth=2)
        ax.axvline(x=self.training_data.index.max(), color='red', linestyle='--', label='Training Split', linewidth=2)
        ax.legend(fontsize=12)
        ax.set_xlabel('Time', fontsize=14, fontweight='bold')
        ax.set_ylabel('Traffic (Megabytes)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)

        y_max = max(self.training_data['y'].max(), self.testing_data['y'].max(), self.prediction.max())
        ax.set_yticks([i for i in range(0, int(y_max) + 2)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network traffic prediction script")
    parser.add_argument('--csv', type=str, default="captures", help="Folder containing the csv files to use as input")
    parser.add_argument('--store-plot', type=str, default="plots", help="Folder where to store the plots")
    parser.add_argument('--training-split', type=float, default=.8, help="Percentage of data used for training")
    parser.add_argument('--sample-period', type=str, default="200L", help="Period over which to combine network data")
    parser.add_argument('--use-auto-arima', action='store_true', help="Use auto ARIMA for parameter selection")
    parser.add_argument('--seasonal', action='store_true', help="Consider seasonal components in auto ARIMA")
    parser.add_argument('--order', type=str, default="1,1,1", help="ARIMA order parameters p,d,q")
    parser.add_argument('--seasonal-order', type=str, default="1,1,1,12", help="Seasonal ARIMA order parameters P,D,Q,s")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.store_plot):
        os.mkdir(args.store_plot)
    
    # Parse order and seasonal_order arguments
    order = tuple(map(int, args.order.split(',')))
    seasonal_order = tuple(map(int, args.seasonal_order.split(',')))
    
    path = args.csv
    for switch in os.listdir(path):
        intf_csv = [file for file in os.listdir(os.path.join(path, switch)) if file.endswith('.csv')]
        
        plot_count = len(intf_csv)
        
        if plot_count == 0:
            print(f"No .csv file found in {switch} folder")
            continue

        num_cols = math.ceil(math.sqrt(plot_count))
        num_rows = math.ceil(plot_count / num_cols)
        
        fig, axs = plt.subplots(num_rows, num_cols, sharey=True, sharex=True, figsize=(20, 15))
        axs = axs.flatten()
        
        for ax, interface in zip(axs, intf_csv):
            prediction = TrafficPrediction()
            full_path = os.path.join(path, switch, interface)
            
            print(f"Reading file {full_path}")
            prediction.read_from_csv(full_path, args.sample_period)
            
            if args.use_auto_arima:
                print("Running auto ARIMA prediction...")
                try:
                    prediction.run_auto_arima(training_split=args.training_split, seasonal=args.seasonal)
                except ValueError as e:
                    print(f"Auto ARIMA failed for {interface} with error: {e}. Falling back to manual SARIMA.")
                    print("Running manual SARIMA prediction...")
                    prediction.run_sarima(order=order, seasonal_order=seasonal_order, training_split=args.training_split)
            else:
                print("Running manual SARIMA prediction...")
                prediction.run_sarima(order=order, seasonal_order=seasonal_order, training_split=args.training_split)
                
            prediction.plot(ax)
            ax.set_title(interface[:-4], fontsize=16, fontweight='bold')

        for ax in axs[plot_count:]:
            ax.axis('off')
        
        fig.suptitle(switch, fontsize=22, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)
        plt.savefig(os.path.join(args.store_plot, switch) + '.png', dpi=300)
        plt.show()
