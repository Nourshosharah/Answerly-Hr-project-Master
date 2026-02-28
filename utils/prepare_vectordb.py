# ================================================================
# 0.  ENVIRONMENT & IMPORTS
# ================================================================
import os
import re
import uuid
import time
import logging
import csv
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict

from tqdm import tqdm
from langchain_core.documents import Document
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"
except AttributeError:
    pass

from pdf2image import convert_from_path
import base64
import io
import pandas as pd
from PIL import Image
import numpy as np
import pytesseract

# Docling imports
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import TableItem, DocItemLabel
from transformers import AutoTokenizer

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger('data_upload')

# ---------  local config -----------------------------------------
from utils.load_config import LoadConfig
APPCFG = LoadConfig()

# ================================================================
# 1.  HF-OFFLINE HOOK
# ================================================================
os.environ["HF_HUB_OFFLINE"] = "1"

def local_only_download(repo_id, filename=None, *args, **kwargs):
    if repo_id == "unstructuredio/yolo_x_layout":
        return os.path.expanduser(
            "~/.cache/huggingface/hub/models--unstructuredio--yolo_x_layout/"
            "snapshots/7680d6f857780bcf8d49916aa2e8881bd49dee3e/yolox_l0.05.onnx"
        )
    raise RuntimeError(f"Offline mode: no local file for {repo_id}")

import huggingface_hub.file_download as fd
fd.hf_hub_download = local_only_download

forbidden_phrases = [
    ".إن المعلومات المذكورة هنا هي ملكية خاصة لشركة سيريتل، ويجب ألاّ يتم استخدامها أو نسخها أو إظهارها إلا بتصريح مكتوب من المالك. يتعهد مستلم هذه المعلومات بالاحتفاظ بها واستخدامها، كما يوافق على حمايتها من الضياع، السرقة أو الاستخدام غير المسموح به",
    "THE INFORMATION CONTAINED HEREIN IS PROPRIETARY TO SYRIATEL, AND IT SHALL NOT BE USED, REPRODUCED\nOR DISCLOSED TO OTHERS EXCEPT AS SPECIFICALLY PERMITTED IN WRITING BY THE PROPRIETOR. THE RECIPIENT\nOF THIS INFORMATION, BY ITS RETENTION AND USE, AGREES TO PROTECT THE SAME FROM LOSS, THEFT OR\nUNAUTHORIZED USE.",
    "the information contained herein is proprietary to syriatel, and it shall not be used, reproduced\nor disclosed to others except as specifically permitted in writing by the proprietor. the recipient\nof this information, by its retention and use, agrees to protect the same from loss, theft or\nunauthorized use.",
    ".إن المعلومات المذكورة هنا هي ملكية خاصة لشركة سيريتل، ويجب ألاّ يتم استخدامها أو نسخها أو إظهارها إلا بتصريح مكتوب من المالك\. يتعهد مستلم هذه المعلومات بالاحتفاظ بها واستخدامها، كما يوافق على حمايتها من الضياع، السرقة أو الاستخدام غير المسموح به",
    "إن المعلومات المذكورة هنا هي ملكية خاصة لشركة سيريتل، ويجب ألاّ يتم استخدامها أو نسخها أو إظهارها إلا بتصريح مكتوب من المالك\. يتعهد مستلم هذه المعلومات بالاحتفاظ بها واستخدامها، كما يوافق على حمايتها من الضياع، السرقة أو الاستخدام غير المسموح به\.",
    "THE INFORMATION CONTAINED HEREIN IS PROPRIETARY TO SYRIATEL, AND IT SHALL NOT BE USED, REPRODUCED OR DISCLOSED TO OTHERS EXCEPT AS SPECIFICALLY PERMITTED IN WRITING BY THE PROPRIETOR\. THE RECIPIENT OF THIS INFORMATION, BY ITS RETENTION AND USE, AGREES TO PROTECT THE SAME FROM LOSS, THEFT OR UNAUTHORIZED USE\."
]

def build_flexible_pattern(text: str) -> re.Pattern:
    words = re.split(r'\s+', text.strip())
    escaped_words = [re.escape(word) for word in words if word]
    pattern = r'\s+'.join(escaped_words)
    return re.compile(pattern, re.IGNORECASE)

forbidden_patterns = [build_flexible_pattern(phrase) for phrase in forbidden_phrases]


# ================================================================
# 2.  PREPARE VECTOR-DB CLASS (DOCLING VERSION)
# ================================================================
class PrepareVectorDB:
    """
    Build a Chroma vector DB from PDF(s) using Docling.
    - Extracts header pattern from first page before skipping.
    - Drops first N pages (configurable, default 2).
    - Cuts from the FIRST occurrence of trigger phrase (deletes rest of file).
    - Removes forbidden phrases from content.
    - Removes headers using pattern extracted from first page.
    - Processes each PDF independently.
    - Saves chunks to CSV file with chunk_id and chunk_txt columns.
    """

    def __init__(
        self,
        data_directory: Union[str, List[Union[str, object]]],
        persist_directory: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        artifacts_path: Optional[str] = None,
        embed_model_path: Optional[str] = None,
        do_ocr: bool = False,
        do_table_structure: bool = False,
        ocr_languages: Optional[List[str]] = None,
        num_threads: int = 4,
        device: str = "CPU",
        skip_pages: int = 1,
        csv_output_path: Optional[str] = None,
    ) -> None:
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.skip_pages = skip_pages
        self.csv_output_path = csv_output_path

        self.artifacts_path = Path(artifacts_path) if artifacts_path else None
        self.embed_model_path = embed_model_path
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.ocr_languages = ocr_languages or ["ar", "en"]
        self.num_threads = num_threads

        device_map = {
            "CPU": AcceleratorDevice.CPU,
            "CUDA": AcceleratorDevice.CUDA,
            "MPS": AcceleratorDevice.MPS,
            "AUTO": AcceleratorDevice.AUTO,
        }
        self.device = device_map.get(device.upper(), AcceleratorDevice.CPU)

        self._init_docling_converter()
        self._init_chunker()

        # Store header patterns extracted from first pages
        self.header_patterns: Dict[str, re.Pattern] = {}

        print("? All models and settings loaded successfully (Docling version).")
        logger.info("? All models and settings loaded successfully (Docling version).")

    def _init_docling_converter(self) -> None:
        """Initialize Docling pipeline with configured options."""
        pipeline_options = PdfPipelineOptions(
            do_table_structure=self.do_table_structure,
            do_ocr=self.do_ocr,
        )

        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=self.num_threads,
            device=self.device
        )

        if self.do_ocr and self.ocr_languages:
            pipeline_options.ocr_options.lang = self.ocr_languages

        if self.do_table_structure:
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True
            )

        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info("?? Docling DocumentConverter initialized.")

    def _init_chunker(self) -> None:
        """Initialize HybridChunker with HuggingFace tokenizer."""
        if self.embed_model_path:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_path)
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=self.chunk_size,
            )
            self.chunker = HybridChunker(
                tokenizer=self.tokenizer,
                merge_peers=True,
            )
            logger.info(f"?? HybridChunker initialized with max_tokens={self.chunk_size}")
        else:
            self.tokenizer = None
            self.chunker = HybridChunker(merge_peers=True)
            logger.info("?? HybridChunker initialized with default settings.")

    # --------------------------------------------------------
    # 2.1  EXTRACT HEADER PATTERN FROM FIRST PAGE
    # --------------------------------------------------------
    def _extract_header_pattern(self, text: str, pdf_name: str) -> Optional[re.Pattern]:
        """
        Extract header pattern from first page text.
        Looks for Control Number, Revision Number, Date of Issue pattern.
        """
        # Pattern to match the header block
        # Matches:
        # Control Number\t<value>\n\n\nRevision Number\t<value>\n\n\nDate of Issue\t<value>
        header_regex = r'Control Number\s+(\S+)\s+Revision Number\s+(\S+)\s+Date of Issue\s+(\S+\s+\S+\s+\d{4})'
        
        match = re.search(header_regex, text, re.IGNORECASE | re.DOTALL)
        if match:
            control_num = re.escape(match.group(1))
            revision_num = re.escape(match.group(2))
            date_issue = re.escape(match.group(3))
            
            # Build flexible pattern that will match this header even with varying whitespace
            pattern_str = rf'Control Number\s+{control_num}\s+Revision Number\s+{revision_num}\s+Date of Issue\s+{date_issue}'
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            
            logger.info(f"?? Extracted header pattern from {pdf_name}: Control={match.group(1)}, Revision={match.group(2)}, Date={match.group(3)}")
            return compiled_pattern
        
        # Try alternative pattern with tabs and newlines
        alt_pattern = r'Control Number[:\s]+([^\n]+)[\s\n]+Revision Number[:\s]+([^\n]+)[\s\n]+Date of Issue[:\s]+([^\n]+)'
        match = re.search(alt_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            control_num = re.escape(match.group(1).strip())
            revision_num = re.escape(match.group(2).strip())
            date_issue = re.escape(match.group(3).strip())
            
            pattern_str = rf'Control Number[:\s]+{control_num}[\s\n]+Revision Number[:\s]+{revision_num}[\s\n]+Date of Issue[:\s]+{date_issue}'
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
            
            logger.info(f"?? Extracted header pattern (alt) from {pdf_name}")
            return compiled_pattern
        
        logger.warning(f"?? Could not extract header pattern from {pdf_name}")
        return None

    # --------------------------------------------------------
    # 2.2  REMOVE HEADER USING EXTRACTED PATTERN
    # --------------------------------------------------------
    def _remove_header_by_pattern(self, text: str, pdf_name: str) -> str:
        """Remove header using pattern extracted from first page."""
        if pdf_name not in self.header_patterns or self.header_patterns[pdf_name] is None:
            return text
        
        pattern = self.header_patterns[pdf_name]
        cleaned_text = pattern.sub('', text)
        
        # Clean up extra whitespace left behind
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    # --------------------------------------------------------
    # 2.3  DOCUMENT LOADER
    # --------------------------------------------------------
    def __load_all_documents(self) -> List[Tuple[str, "ConversionResult"]]:
        """Load all PDFs using Docling DocumentConverter."""
        pdf_results = []

        if isinstance(self.data_directory, list):
            pdf_files = [f for f in self.data_directory
                        if isinstance(f, str) and f.lower().endswith(".pdf")]
        elif os.path.isfile(self.data_directory):
            pdf_files = [self.data_directory]
        elif os.path.isdir(self.data_directory):
            pdf_files = [
                os.path.join(self.data_directory, f)
                for f in os.listdir(self.data_directory)
                if f.lower().endswith(".pdf")
            ]
        else:
            raise ValueError("Invalid data_directory argument!")

        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = os.path.basename(pdf_path)
            logger.info(f"[{i}/{len(pdf_files)}] Converting PDF with Docling: {pdf_name}")

            start_time = time.time()
            try:
                conv_result = self.doc_converter.convert(pdf_path)
                elapsed = time.time() - start_time
                logger.info(f"?? {pdf_name} converted in {elapsed:.2f} seconds.")
                pdf_results.append((pdf_name, conv_result))
            except Exception as e:
                logger.error(f"? Failed to convert {pdf_name}: {e}")
                continue

        print(f'?? Total PDFs loaded: {len(pdf_results)}')
        logger.info(f"?? Total PDFs loaded: {len(pdf_results)}")
        return pdf_results

    # --------------------------------------------------------
    # 2.4  CLEAN TEXT CONTENT
    # --------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """Remove forbidden phrases from text content."""
        new_content = text
        removed_any = False

        for pattern in forbidden_patterns:
            while pattern.search(new_content):
                removed_any = True
                new_content = pattern.sub('', new_content)
                logger.debug(f'?? Removed a forbidden phrase')

        if removed_any:
            new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)
            new_content = re.sub(r' +', ' ', new_content)
            new_content = new_content.strip()

        return new_content

    # --------------------------------------------------------
    # 2.5  PROCESS DOCLING CHUNKS
    # --------------------------------------------------------
    def _process_docling_chunks(
        self,
        chunks: List,
        pdf_name: str
    ) -> List[Document]:
        """
        Process Docling chunks:
        - Extract header pattern from first page (before skipping)
        - Skip first N pages
        - Cut from FIRST occurrence of trigger phrase (delete rest of file)
        - Clean forbidden phrases
        - Remove headers using extracted pattern
        """
        print(f'?? Processing chunks from {pdf_name}...')

        trigger_phrases = ["related documents", "الوثائق المتعلقة"]
        cut_triggered = False
        processed_docs = []
        first_page_text = None

        for i, chunk in enumerate(chunks):
            # Extract metadata from chunk
            pages = set()
            for item in chunk.meta.doc_items:
                for prov in item.prov:
                    pages.add(prov.page_no)

            min_page = min(pages) if pages else 1
            max_page = max(pages) if pages else 1
            # Extract header pattern from first page before skipping
            if min_page == 1 and first_page_text is None:
                first_page_text = chunk.text
                header_pattern = self._extract_header_pattern(chunk.text, pdf_name)
                self.header_patterns[pdf_name] = header_pattern
                logger.info(f"?? Extracted header pattern from page 1 of {pdf_name}")
                continue  # Skip this chunk (first page)

            # Skip first N pages
            if min_page <= self.skip_pages:
                logger.debug(f'?? Skipping chunk {i} (page {min_page} <= {self.skip_pages})')
                continue

            # Check for trigger phrase - FIRST occurrence cuts the rest
            chunk_text_lower = chunk.text.lower()
            if any(trigger in chunk_text_lower for trigger in trigger_phrases):
                logger.info(f'?? Trigger phrase found in chunk {i} of {pdf_name} - cutting rest of document')
                cut_triggered = True
                break  # Stop processing this PDF entirely

            # Clean text content
            cleaned_text = self._clean_text(chunk.text)
            
            # Remove header using extracted pattern
            cleaned_text = self._remove_header_by_pattern(cleaned_text, pdf_name)

            if not cleaned_text.strip():
                continue

            # Get filename from chunk metadata
            filename = chunk.meta.origin.filename if chunk.meta.origin else pdf_name

            # Create LangChain Document with metadata
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "page": sorted(pages)[0] if pages else 0,
                    "pages": sorted(pages),
                    "chunk_index": i,
                    "cut_by_trigger": False,
                }
            )
            processed_docs.append(doc)

        if cut_triggered:
            # Mark the last chunk as cut point (if any chunks exist)
            if processed_docs:
                processed_docs[-1].metadata["cut_by_trigger"] = True

        logger.info(f'? Final chunks after processing for {pdf_name}: {len(processed_docs)}')

        return processed_docs

    # --------------------------------------------------------
    # 2.6  CHUNK DOCUMENT
    # --------------------------------------------------------
    def __chunk_document(self, conv_result, pdf_name: str) -> List[Document]:
        """Chunk a converted document using Docling's HybridChunker."""
        try:
            doc = conv_result.document
            chunk_iter = self.chunker.chunk(dl_doc=doc)
            chunks = list(chunk_iter)

            logger.info(f"?? Generated {len(chunks)} raw chunks from {pdf_name}")

            # Initialize header pattern storage for this PDF
            self.header_patterns[pdf_name] = None
            # Process chunks
            processed_docs = self._process_docling_chunks(chunks, pdf_name)

            return processed_docs

        except Exception as e:
            logger.error(f"? Error chunking {pdf_name}: {e}")
            return []

    # --------------------------------------------------------
    # 2.7  SAVE CHUNKS TO CSV
    # --------------------------------------------------------
    def _save_chunks_to_csv(self, chunks: List[Document]) -> None:
        """Save chunks to CSV file with chunk_id and chunk_txt columns."""
        if not self.csv_output_path:
            logger.info("No CSV output path specified, skipping CSV export.")
            return
        
        try:
            with open(self.csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['chunk_id', 'chunk_txt', 'source', 'page', 'cut_by_trigger'])
                
                for doc in chunks:
                    chunk_id = str(uuid.uuid4())
                    chunk_txt = doc.page_content
                    source = doc.metadata.get('source', '')
                    page = doc.metadata.get('page', 0)
                    cut = doc.metadata.get('cut_by_trigger', False)
                    writer.writerow([chunk_id, chunk_txt, source, page, cut])
            
            # print(f'?? Saved {len(chunks)} chunks to CSV: {self.csv_output_path}')
            logger.info(f'?? Saved {len(chunks)} chunks to CSV: {self.csv_output_path}')
        except Exception as e:
            logger.error(f"? Failed to save CSV: {e}")

    # --------------------------------------------------------
    # 2.8  BUILD & SAVE VECTOR DB
    # --------------------------------------------------------
    def prepare_and_save_vectordb(self) -> int:
        print('?? Loading all documents with Docling...')
        logger.info('?? Loading all documents with Docling...')
        pdf_results = self.__load_all_documents()

        all_chunks = []

        print(f'?? Processing each PDF individually (skipping first {self.skip_pages} pages, extracting headers, cutting on trigger)...')
        logger.info(f'?? Processing each PDF individually (skipping first {self.skip_pages} pages, extracting headers, cutting on trigger)...')

        for pdf_name, conv_result in pdf_results:
            chunks = self.__chunk_document(conv_result, pdf_name)
            all_chunks.extend(chunks)

        print(f'?? Total chunks across all PDFs: {len(all_chunks)}')
        logger.info(f'?? Total chunks across all PDFs: {len(all_chunks)}')

        # Save chunks to CSV before embedding
        self._save_chunks_to_csv(all_chunks)

        print("??? Initializing Chroma vector store...")
        logger.info("??? Initializing Chroma vector store...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model_path, 
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.embed_model_path}': {e}")
            raise
    
        vectordb = Chroma(
            embedding_function=APPCFG.embedding_model,
            persist_directory=self.persist_directory,
        )

        buffer, buffer_size, total_added = [], 4, 0
        print(f"?? Embedding and storing (buffer={buffer_size})...")
        logger.info(f"?? Embedding and storing (buffer={buffer_size})...")

        for doc in all_chunks:
            txt = doc.page_content.strip()
           
            if not txt:
                continue

            tbl_count = 0
            if doc.metadata.get("text_as_html"):
                try:
                    tbl_count = len(pd.read_html(doc.metadata["text_as_html"]))
                except Exception:
                    pass

            meta = {k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool, type(None), list))}
            meta.update({"tables_in_chunk": tbl_count})

            if "pages" in meta and isinstance(meta["pages"], list):
                meta["pages"] = ",".join(map(str, meta["pages"]))

            buffer.append(Document(page_content=txt, metadata=meta))

            if len(buffer) >= buffer_size:
                ids = [str(uuid.uuid4()) for _ in buffer]
                vectordb.add_documents(documents=buffer, ids=ids)
                total_added += len(buffer)
                print(f'? Added batch of {len(buffer)} docs ? total: {total_added}')
                logger.info(f'? Added batch of {len(buffer)} docs ? total: {total_added}')
                buffer = []

        if buffer:
            ids = [str(uuid.uuid4()) for _ in buffer]
            vectordb.add_documents(documents=buffer, ids=ids)
            total_added += len(buffer)

        vectordb.persist()
        print(f'\n? Vector store saved successfully with {total_added} documents.')
        logger.info(f'\n? Vector store saved successfully with {total_added} documents.')
        return total_added