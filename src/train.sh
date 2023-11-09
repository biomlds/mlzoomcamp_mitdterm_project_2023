jupyter nbconvert --to script pycaret_train.ipynb
python pycaret_train.py

#adjust deployment container
echo pycaret==3.1.0 > requirements.txt
echo fastapi==0.104.1 >> requirements.txt
echo uvicorn==0.24.0.post1 >> requirements.txt
echo catboost==1.2.2 >> requirements.txt
echo pydantic==1.10.13 >> requirements.txt
sed "s/127\.0\.0\.1/0\.0\.0\.0/" smoking_clf_api.py