import numpy as np
import pandas as pd

from madewithml import predict


def get_label(text, predictor):
    df = pd.DataFrame({"title": [text], "description": "", "tag": "other"})
    z = predictor.predict(data=df)["predictions"]
    preprocessor = predictor.get_preprocessor()
    label = predict.decode(np.stack(z).argmax(1), preprocessor.index_to_class)[0]
    return label
