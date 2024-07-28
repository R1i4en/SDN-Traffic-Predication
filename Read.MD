SDN Traffic Prediction Project

This project focuses on predicting network traffic using ARIMA models within a Software-Defined Networking (SDN) environment. The goal is to capture network traffic data, preprocess it, and use ARIMA to make accurate predictions.

Prerequisites:

Before you begin, ensure you have met the following requirements:

- Python 3.6 or later installed.
- Necessary Python libraries installed: `pandas`, `matplotlib`, `pmdarima`.
- Access to a network environment where you can capture traffic data.

Installation:

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/R1i4en/SDN-Traffic-Predication.git
    cd SDN-Traffic-Predication
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

Your project should have the following structure:

your-project/
├── main.py
├── analyzer.py
├── README.md
├── requirements.txt
├── presentation.pptx
└── captures/
└── (CSV files)
└── plots/
└── (PNG files)

bash


Usage:

Capturing Network Traffic

The `main.py` script simulates network traffic and captures the data in CSV files. To capture network traffic data, use the following command:

```bash
sudo -E env "PATH=$PATH" python3 main.py --switches 7 --hosts 2 --cross-connection 0.3 --time 120 --base-flows 3 --flows 2

Parameters for main.py

    --switches: Number of switches in the network.
    --hosts: Number of hosts in the network.
    --cross-connection: Probability of cross-connections between switches.
    --time: Duration of the data capture in seconds.
    --base-flows: Base number of flows.
    --flows: Additional flows.

Example Usage:

You can customize the parameters to fit your specific needs. For example, to capture data for a longer duration or to change the network configuration:

bash

sudo -E env "PATH=$PATH" python3 main.py --switches 10 --hosts 5 --cross-connection 0.5 --time 300 --base-flows 5 --flows 3

Running Traffic Prediction

The analyzer.py script reads the captured data from the CSV files, resamples it to a specified sample period, and preprocesses it for ARIMA modeling. To run the traffic prediction script, use the following command:

bash

python3 analyzer.py --csv captures --store-plot plots --training-split 0.8 --sample-period 100L

Parameters for analyzer.py

    --csv: Folder containing the CSV files to use as input.
    --store-plot: Folder where to store the plots.
    --training-split: Percentage of data used for training.
    --sample-period: Period over which to combine network data (e.g., 100L for 100 milliseconds).

Detailed Explanation of Project Files:
main.py

The main.py script is responsible for capturing network traffic data. It simulates network traffic based on the specified parameters and stores the captured data in CSV files. The captured data includes timestamps and traffic volumes, which are essential for further analysis and prediction.
analyzer.py

The analyzer.py script reads the captured data, resamples it to a specified sample period, and preprocesses it for ARIMA modeling. The preprocessing steps include:

    Converting timestamps to a datetime format.
    Resampling the data to ensure uniform intervals.
    Differencing the data to make it stationary.

After preprocessing, the ARIMA model is trained on the training portion of the data, and predictions are made on the testing portion. The script then visualizes the training data, actual data, and predicted data for comparison.
