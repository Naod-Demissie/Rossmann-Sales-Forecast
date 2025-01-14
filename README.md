# Rossmann Sales Forecast

This is an end-to-end machine learning solution for forecasting Rossmann Pharmaceuticals' store sales six weeks ahead, leveraging features like promotions, competition, and seasonality. This project includes data exploration, model development, and deployment using MLOps tools such as DVC, MLFlow, and CI/CD pipelines."

## Project Structure

```
├── checkpoints  
│   ├── best_model
|
├── configs                     
├── data
│   ├── processed
│   │   ├── processed_data.csv    
│   │   ├── X_train.pkl 
│   │   ├── X_val.pkl    
│   │   ├── y_train.pkl    
│   │   ├── y_val.pkl    
│   ├── raw
│   │   ├── sample_submission.csv 
│   │   ├── store.csv             
│   │   ├── test.csv              
│   │   ├── train.csv             
│   ├── raw.dvc                   
│
├── dvc.lock                    
├── dvc.yaml                    
├── logs 
│   ├── train
│   ├── validation
|                
├── notebooks
│   ├── 1.0-data-exploration.ipynb 
│   ├── 2.0-modeling.ipynb 
│   ├── README.md                 
│   ├── __init__.py               
│
├── README.md                   
├── requirements.txt            
├── scripts
│   ├── README.md                 
│   ├── __init__.py               
│
├── src
│   ├── eda
│   │   ├── correlation_analysis.py
│   │   ├── customer_analysis.py   
│   │   ├── promotional_analysis.py
│   │   ├── sales_analysis.py      
│   │   ├── store_analysis.py      
│   │   ├── trend_analysis.py      
│   │   ├── __init__.py            
│   ├── preprocess.py             
│   ├── train.py             
│   ├── __init__.py               
│
├── tests
│   ├── __init__.py               


```

## Installation

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <project_directory>
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv/Scripts/activate`
   pip install -r requirements.txt
   ```

## Contribution

Feel free to fork the repository, make improvements, and submit pull requests.
