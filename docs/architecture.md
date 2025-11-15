ltsf-linear/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── __init__.py
│   ├── model_config.py      # Cấu hình mô hình
│   └── train_config.py      # Cấu hình training
│
├── data/
│   ├── raw/                 # Dữ liệu gốc (VIC.csv)
│   ├── processed/           # Dữ liệu đã xử lý
│   └── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # UnivariateTimeSeriesDataset, NormalizedDataset
│   │   ├── dataloader.py    # DataLoader utilities
│   │   └── preprocessor.py  # Data preprocessing, feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py          # Base model class
│   │   ├── linear.py        # Linear model
│   │   ├── nlinear.py       # NLinear model
│   │   └── dlinear.py       # DLinear model
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training logic
│   │   ├── evaluator.py     # Evaluation metrics
│   │   └── callbacks.py     # Training callbacks
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py       # MSE, MAE, RMSE, R2
│   │   ├── decomposition.py # Moving average, trend/seasonal decomposition
│   │   └── visualization.py # Plotting functions
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── forecast_pipeline.py  # End-to-end pipeline
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_from_scratch_models.ipynb
│   ├── 03_full_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── predict.py           # Prediction script
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
│
├── experiments/
│   ├── logs/                # Training logs
│   ├── checkpoints/         # Model checkpoints
│   └── results/             # Results, plots
│
└── docs/
    ├── architecture.md
    └── api_reference.md