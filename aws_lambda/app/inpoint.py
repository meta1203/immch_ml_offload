import sys
import asyncio
import urllib3

from .models.base import InferenceModel
from .models.cache import ModelCache
from .config import log, settings

from zipfile import BadZipFile
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile  # type: ignore

model_cache = ModelCache(ttl=settings.model_ttl, revalidate=settings.model_ttl > 0)
http = urllib3.PoolManager(1)

async def load(model: InferenceModel) -> InferenceModel:
    if model.loaded:
        return model
    try:
        model.load()
        return model
    except (OSError, InvalidProtobuf, BadZipFile, NoSuchFile):
        log.warn(
            (
                f"Failed to load {model.model_type.replace('_', ' ')} model '{model.model_name}'."
                "Clearing cache and retrying."
            )
        )
        model.clear_cache()
        model.load()
        return model

def get_temp(url: str) -> bytes:
    return http.request("GET", url).data

def handler(event, context):
    print(event)
    inputs: str | bytes = event['inputs']
    if (inputs.startswith("url|")):
        inputs = get_temp(event['tmp_host'] + inputs[4:])
    model = asyncio.run(load(model_cache.get(event['model_name'], event['model_type'])))
    model.configure(**event['kwargs'])
    return model.predict(inputs)