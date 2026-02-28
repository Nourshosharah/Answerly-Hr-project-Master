import os
from langchain_community.vectorstores import Chroma
from utils.load_config import LoadConfig
from utils.global_embedding import GLOBAL_EMBEDDINGS

def export_all_chunks_to_file(persist_dir: str, output_file: str = "all_chunks_export.txt", batch_size: int = 100):
    if not os.path.exists(persist_dir):
        print(f"? DB not found: {persist_dir}")
        return

    # Use FakeEmbeddings to avoid GPU
    embedding = GLOBAL_EMBEDDINGS
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    coll = db._collection
    total = coll.count()

    if total == 0:
        print("?? No chunks found.")
        return

    all_ids = coll.get(include=[])["ids"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total chunks: {total}\n")
        f.write("=" * 60 + "\n\n")

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i:i + batch_size]
            res = coll.get(ids=batch_ids, include=["metadatas", "documents"])

            for doc, meta in zip(res["documents"], res["metadatas"]):
                page = meta.get("page_number", "N/A")
                tables = meta.get("tables_in_chunk", 0)
                ocr = meta.get("image_ocr", "None")

                f.write(f"[Chunk Index: {i + 1}]\n")
                f.write(f"Page: {page} | Tables: {tables}\n")
                if ocr != "None":
                    f.write(f"OCR: {ocr[:200]}{'...' if len(ocr) > 200 else ''}\n")
                f.write("Text:\n")
                f.write(doc.strip() + "\n")
                f.write("\n" + "=" * 80 + "\n\n")

    print(f"? Export completed: {output_file}")
    for x in coll:
        print(x)
        # print({vars(x.metadata).keys()})

# === CONFIG ===
APPCFG = LoadConfig()
persist_dir = APPCFG.persist_directory  # or APPCFG.custom_persist_directory

export_all_chunks_to_file(persist_dir, "all_chunks_export.txt", batch_size=50)