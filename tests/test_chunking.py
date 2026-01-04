from langchain.text_splitter import RecursiveCharacterTextSplitter


def test_chunking_respects_max_length():
    text = "word " * 1000

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_overlap_exists():
    text = "word " * 200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_text(text)

    # Overlap means chunks share some content
    assert chunks[0][-20:] in chunks[1]
