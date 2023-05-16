import pandas as pd
from spafe.features.rplp import plp
from fastapi import FastAPI, UploadFile
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences
import scipy
import numpy as np
import io
from tensorflow import keras
model = keras.models.load_model('models/PLP-CNN.h5')

data_train = pd.read_csv('data_train.csv')

scaler = MinMaxScaler()
scaler.fit_transform(data_train[['Age', 'Weight', 'Height']])

app = FastAPI()


# Set up allowed origins, methods, and headers
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run(path):
  contents = await path.read()
  audio = io.BytesIO(contents)
  fs, sig = scipy.io.wavfile.read(audio)
  sig = sig.astype(np.float32)
  sig += np.random.normal(0, 0.001, sig.shape)
  plp_features = plp(sig, fs=fs, order=20)
  plp_features = np.mean(plp_features.T, axis=0)
  plp_processed = pad_sequences([plp_features], maxlen=1711, padding="post", truncating="post")
  plp_processed = np.reshape(plp_processed, (1, plp_processed.shape[1], 1))

  print(plp_processed.shape)
  # Predict the target values
  preds = model.predict(plp_processed)
  print("NOT FORMATTED PRED:", preds)

  # Reverse normalization for Age, Weight, and Height
  preds[0, 1:] = scaler.inverse_transform([preds[0, 1:]])

  # # Round gender to the nearest integer
  preds[0, 0] = np.round(preds[0, 0])

  # print(f"Predicted Gender (0 for Male, 1 for Female): {preds[0, 0]}")
  # print(f"Predicted Age: {preds[0, 1]}")
  # print(f"Predicted Weight: {preds[0, 2]}")
  # print(f"Predicted Height: {preds[0, 3]}")
  return {"age": str(preds[0,1]), "gender": str(preds[0,0]), "weight": str(preds[0,2]), "height": str(preds[0,3])}

@app.post("/plp")
async def run_plp(audio_file: UploadFile):
  preds = await run(audio_file)
  return preds
  # return {"filename": audio_file.filename}