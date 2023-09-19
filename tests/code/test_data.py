import pandas as pd
import pytest
import ray

from madewithml import data


@pytest.fixture(scope="module")
def df():
    data = [{"title": "a0", "description": "b0", "tag": "c0"}]
    df = pd.DataFrame(data)
    return df


@pytest.fixture(scope="module")
def class_to_index():
    class_to_index = {"c0": 0, "c1": 1}
    return class_to_index


def test_load_data(dataset_loc):
    num_samples = 10
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=num_samples)
    assert ds.count() == num_samples


def test_stratify_split():
    n_per_class = 10
    targets = n_per_class * ["c1"] + n_per_class * ["c2"]
    ds = ray.data.from_items([dict(target=t) for t in targets])
    train_ds, test_ds = data.stratify_split(ds, stratify="target", test_size=0.5)
    train_target_counts = train_ds.to_pandas().target.value_counts().to_dict()
    test_target_counts = test_ds.to_pandas().target.value_counts().to_dict()
    assert train_target_counts == test_target_counts


@pytest.mark.parametrize(
    "text, sw, clean_text",
    [
        ("hi", [], "hi"),
        ("hi you", ["you"], "hi"),
        ("hi yous", ["you"], "hi yous"),
    ],
)
def test_clean_text(text, sw, clean_text):
    assert data.clean_text(text=text, stopwords=sw) == clean_text


def test_preprocess(df, class_to_index):
    assert "text" not in df.columns
    outputs = data.preprocess(df, class_to_index=class_to_index)
    assert set(outputs) == {"ids", "masks", "targets"}


def test_fit_transform(dataset_loc, preprocessor):
    ds = data.load_data(dataset_loc=dataset_loc)
    preprocessor = preprocessor.fit(ds)
    preprocessed_ds = preprocessor.transform(ds)
    assert len(preprocessor.class_to_index) == 4
    assert ds.count() == preprocessed_ds.count()
