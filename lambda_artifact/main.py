from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from mangum import Mangum
from PIL import Image
import io

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def root():
	return {"Root Route"} 

#List of all the card details
#Likely will be loaded from another file or DB
reference = {}


#Incoming vector as json handling
#Input should be a json list of vector features
class toMatch(BaseModel):
	embedding: List[float]

#Outgoing match results 
class Match(BaseModel):
	id: str
	score: float


#Returns vector embedding of an image from CLIP fine-tuned model
#Takes an image as payload
@app.post("/embed")
def embed_image(file: UploadFile = File(...)) -> List[float]:
	
	content = file.read()
	img = Image.open(io.BytesIO(content))
	
	
	# - Read the image file from the UploadFile object.
	content = file.read()

	# - Convert it to a PIL Image (e.g., using Image.open and BytesIO).
	img = Image.open(io.BytesIO(content))

	# TODO:
	# - Preprocess the image as required by the CLIP model.
	# - Pass the image through the fine-tuned CLIP model.
	# - Extract the embedding (usually a 512D or 768D vector).

	# - Return the embedding as a list of floats (JSON serializable).
	return [0.0] * 768



#Returns the top match for the image from the reference list
#Ex request using curl: 
# curl -X POST -H "Content-Type:application/json" -d '{"embedding":list[768]}' "API_URL/match1"
@app.post("/match1")
def match_t1(req: toMatch) -> Match:
	
	# TODO:
	# - Compare it to all embeddings in the `reference` dictionary.
	# - Use cosine similarity to compute similarity scores.
	# - Identify the reference entry with the highest score.
	# - Return that entry's ID and the score as a Match object.

	return Match(id="dummy_id", score=0.0)


#Returns the top match for the image from the reference list
#Ex request using curl: 
# curl -X POST -H "Content-Type:application/json" -d '{"embedding":list[768]}' "API_URL/match1"
@app.post("/match5")
def match_t5(req: toMatch) -> List[Match]:
	
	# TODO:
	# - Take the input embedding from the request.
	# - Compare it to all embeddings in the `reference` dictionary.
	# - Compute similarity scores for each reference entry.
	# - Sort the entries by similarity score in descending order.
	# - Select the top 5 entries.

	# - Return a list of Match objects (ID and score for each).
	return [Match(id=f"dummy_id_{i}", score=0.0) for i in range(5)]