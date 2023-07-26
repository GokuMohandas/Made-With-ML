import pytest
import utils


@pytest.mark.parametrize(
    "input_a, input_b, label",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_invariance(input_a, input_b, label, predictor):
    """INVariance via verb injection (changes should not affect outputs)."""
    label_a = utils.get_label(text=input_a, predictor=predictor)
    label_b = utils.get_label(text=input_b, predictor=predictor)
    assert label_a == label_b == label


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        ),
    ],
)
def test_directional(input, label, predictor):
    """DIRectional expectations (changes with known outputs)."""
    prediction = utils.get_label(text=input, predictor=predictor)
    assert label == prediction


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This is about graph neural networks.",
            "other",
        ),
    ],
)
def test_mft(input, label, predictor):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = utils.get_label(text=input, predictor=predictor)
    assert label == prediction
