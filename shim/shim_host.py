from typing import Any
from uuid import uuid4
import os

import orjson
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse, ORJSONResponse
from schemas import (
  MessageResponse,
  ModelType,
  TextResponse,
)
from starlette.formparsers import MultiPartParser

MultiPartParser.max_file_size = 2**24  # spools to disk if payload is 16 MiB or larger
app = FastAPI()
temp_readthrough: dict[str, UploadFile] = {}

@app.get("/", response_model=MessageResponse)
async def root() -> dict[str, str]:
  return {"message": "Immich ML"}

@app.get("/ping", response_model=TextResponse)
def ping() -> str:
  return "pong"

@app.get("/tmp_access/{tmp_id}")
async def fetch_tmp_file(tmp_id: str):
  if not tmp_id:
    raise HTTPException(400, "tmp file access requires id")
  if tmp_id not in temp_readthrough:
    raise HTTPException(404, "tmp file does not exist")
  ret_file = temp_readthrough[tmp_id]
  if ret_file._in_memory:
    ret_file_bytes = await ret_file.read()
    return Response(content=ret_file_bytes, media_type=ret_file.content_type)
  else:
    return StreamingResponse(ret_file.file, media_type=ret_file.content_type)

@app.post("/predict")
async def predict(
  model_name: str = Form(alias="modelName"),
  model_type: ModelType = Form(alias="modelType"),
  options: str = Form(default="{}"),
  text: str | None = Form(default=None),
  image: UploadFile | None = None,
) -> Any:
  if image is not None:
    uuid = str(uuid4())
    temp_readthrough[uuid] = image
    inputs: str = f"img|{uuid}"
  elif text is not None:
    inputs = text
  else:
    raise HTTPException(400, "Either image or text must be provided")
  try:
    kwargs = orjson.loads(options)
  except orjson.JSONDecodeError:
    raise HTTPException(400, f"Invalid options JSON: {options}")
  print("recieved request:")
  print(f"model name: {model_name} | model type: {model_type}")
  print(options)
  return ORJSONResponse(
    faas_handoff(orjson.dumps({
      "model_name": model_name,
      "model_type": model_type,
      "kwargs": kwargs,
      "inputs": inputs,
      "tmp_host": os.getenv("TMP_CALLBACK")
    }))
  )


import boto3
lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_LAMBDA_REGION"))
def faas_handoff(event_obj: str):
  response = lambda_client.invoke(
    FunctionName=os.getenv("AWS_LAMBDA_ARN"),
    Payload=event_obj,
  )
  return response.Payload.read()