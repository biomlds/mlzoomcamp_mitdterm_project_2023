# Smoker Status Prediction using Bio-Signals
## ML-zoomcamp 2023, midterm project

As of 2018, smoking has been considered the leading cause of preventable morbidity and mortality in the world, continuing to plague the world’s overall health. The goal of the project is to predict a smoking status of a person based on the following biosignals:
- age : 5-years gap
- height(cm)
- weight(kg)
- waist(cm) : Waist circumference length
- eyesight(left)
- eyesight(right)
- hearing(left)
- hearing(right)
- systolic : Blood pressure
- relaxation : Blood pressure
- fasting blood sugar
- Cholesterol : total
- triglyceride
- HDL : cholesterol type
- LDL : cholesterol type
- hemoglobin
- Urine protein
- serum creatinine
- AST : glutamic oxaloacetic transaminase type
- ALT : glutamic oxaloacetic transaminase type
- Gtp : γ-GTP
- dental caries

Data source: Walter Reade, Ashley Chow. (2023). Binary Prediction of Smoker Status using Bio-Signals. Kaggle. https://kaggle.com/competitions/playground-series-s3e24

Technology stack:
- [PyCaret 3.0](https://pycaret.org/) is an open-source, low-code machine learning library in Python that automates machine learning workflows.
- docker-compose
- (cloud)


### Model training 
_choose one of the following trining options_

1. Jupyter notebook was used for the model development. The notebook can be reviwed in the [GitHub repo](src/pycaret_train.ipynb) or on a local machine at http://localhost:8888/lab/tree/pycaret_train.ipynb .  
   To reproduce the model setup and trainig spin up the development environment:
```sh
docker-compose up --build jupyter-lab
```


2. A better option is to run a trainig script which converts provided notebook to python script, runs the model training, make an API and pack it as a Docker container
```sh
docker-compose up --build train
```

- [CatBoost](https://catboost.ai/) classifier achived the best AUC score of `0.8651`

### Model deployment
- To deploy the model locally as an api execute the following:
```sh
mv src/Dockerfile services/deploy
mv src/requirements.txt services/deploy
mv src/smoking_clf_api.pkl services/deploy
mv src/smoking_clf_api.py services/deploy

docker-compose up --build deployment
```
The API is available at http://localhost:7777/docs

- Alternatively, the dockerized model can be deployed in the cloud. 
[This link](https://pycaret.gitbook.io/docs/learn-pycaret/official-blog/deploy-machine-learning-pipeline-on-aws-fargate) provides detailed instructions on how to deploy ML pipeline on AWS Fargate.
