# pk-card-match-FastAPI
A Fast API application to embed and match images of pktcg cards to diital reference images via CLIP image embeddings.


Commands used to zip
-uvicorn main:app --reload
-pip install -t dependencies -r requirements.txt
-(cd dependencies; zip ..aws_lambda_artifact.zip -r .)
-zip aws_lambda_artifact.zip -u main.py
