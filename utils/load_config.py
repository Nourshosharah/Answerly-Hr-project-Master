import os
import yaml
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from .global_embedding import GLOBAL_EMBEDDINGS

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
load_dotenv()

class LoadConfig:
    """
    Loads all configuration settings from 'app_config.yml' and manages directories.
    """
    def __init__(self, cfg_path=r"/home/rangpt/Documents/hr_system_eval_LDAP_copy/utils/app_config.yml"):
        with open(cfg_path) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        # Retrieval configs
        self.data_directory = app_config["directories"]["data_directory"]
        self.enable_trulens_evaluation = app_config.get('enable_trulens_evaluation', False)
        self.media_url = app_config["directories"]["media_url"]
        self.persist_directory = app_config["directories"]["persist_directory"]
        self.custom_persist_directory = app_config["directories"]["custom_persist_directory"]
        self.k = app_config["retrieval_config"]["k"]
        self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
        self.emb_model = os.path.basename(self.embedding_model_engine)
        self.embedding_model = GLOBAL_EMBEDDINGS

        # Chunking configs (for retrieval/splitting)
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        # NEW: Docling-specific ingestion configs
        docling_cfg = app_config.get("docling_config", {})
        self.docling_chunk_size = docling_cfg.get("chunk_size", 512)
        self.docling_chunk_overlap = docling_cfg.get("chunk_overlap", 50)
        self.docling_artifacts_path = docling_cfg.get("artifacts_path", None)
        self.embed_model_path = docling_cfg.get("embed_model_path", self.embedding_model_engine)
        self.do_ocr = docling_cfg.get("do_ocr", False)
        self.do_table_structure = docling_cfg.get("do_table_structure", False)
        self.ocr_languages = docling_cfg.get("ocr_languages", ["ar", "en"])
        self.num_threads = docling_cfg.get("num_threads", 4)
        self.device = docling_cfg.get("device", "CPU")

        # Summarizer configs
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.final_summarizer_llm_system_role = app_config["summarizer_config"]["final_summarizer_llm_system_role"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]

        # Cost configs
        self.input_per_1m = app_config["cost"]["input_per_1m"]
        self.output_per_1m = app_config["cost"]["output_per_1m"]

        # Memory configs
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        # Ensure directories exist / clean custom directory
        self.create_directory(self.persist_directory)
        self.remove_directory(self.custom_persist_directory)

    def create_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def remove_directory(self, directory_path: str):
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(f"Removed directory: {directory_path}")
            except OSError as e:
                print(f"Error removing directory {directory_path}: {e}")