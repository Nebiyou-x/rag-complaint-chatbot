from sentence_transformers import SentenceTransformer


def test_embedding_dimension():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    text = ["This is a test complaint"]
    embedding = model.encode(text)

    assert embedding.shape[1] == 384


def test_embedding_is_not_empty():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = model.encode(["Test"])
    assert embedding is not None
