import ray

from madewithml import predict


def get_label(text, predictor):
    sample_ds = ray.data.from_items([{"title": text, "description": "", "tag": "other"}])
    results = predict.predict_proba(ds=sample_ds, predictor=predictor)
    return results[0]["prediction"]
