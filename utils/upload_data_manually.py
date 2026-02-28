import os
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
from utils.global_embedding import GLOBAL_EMBEDDINGS

CONFIG = LoadConfig()


def upload_data_manually() -> None:
    """
    Upload data to vector database using Docling for PDF processing.
    """
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=CONFIG.data_directory,
        persist_directory=CONFIG.persist_directory,
        # embedding_model_engine=GLOBAL_EMBEDDINGS,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        # Docling-specific options
        artifacts_path=getattr(CONFIG, 'docling_artifacts_path', None),
        embed_model_path=getattr(CONFIG, 'embed_model_path', None),
        do_ocr=getattr(CONFIG, 'do_ocr', False),
        do_table_structure=getattr(CONFIG, 'do_table_structure', False),
        ocr_languages=getattr(CONFIG, 'ocr_languages', ["ar", "en"]),
        num_threads=getattr(CONFIG, 'num_threads', 4),
        device=getattr(CONFIG, 'device', "CPU"),
        skip_pages=2,
        csv_output_path="./test_chunks.csv"
    )

    if not os.listdir(CONFIG.persist_directory):
        prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
        print('VectorDB already exists')


# if __name__ == '__main__':
#     upload_data_manually()
