# Credit Scoring

> Dataset : https://www.kaggle.com/competitions/home-credit-default-risk/data

## Installation

## Download the data

In order to download the dataset, you can either download directly the data from the Kaggle website or use the Kaggle API.

> The downloaded data must be placed in side the `data/home-credit-defualt-risk` folder at the project root.

```bash
kaggle competitions download -c home-credit-default-risk
```

## Python Setup

1. Create the virtual environment
```bash 
python3 -m venv .venv
```

2. Use the virtual environment
```bash
source .venv/bin/activate
```

3. Install the dependencies
```bash
pip install -r requirements.txt
```


## Executing the project

1. Prepare the data: run the `notebooks/01_data_preparation.ipynb` notebook to clean and preprocess the data. This step should take around a few minutes depending on your computer and should generate two files : `data/processed/{test|train}_final.csv`.
2. Train the mode: run the `notebooks/02_model_training.ipynb` notebook to train and evaluate the model.

> When that's done you should have a `model/model.pkl` file containing the trained model.

Now to run the project with the dashboard, you just have the type the following command (docker must be installed) :
```bash
docker compose up
```

> You can access the dashboard at http://localhost:8501/.