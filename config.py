import os

class Config:
    """
    Configuration class to manage file paths and other settings for the project.
    """

    # Directories where source PDF documents are located.
    PDF_SOURCE_DIRECTORY_1: str = "pdfs_1"
    PDF_SOURCE_DIRECTORY_2: str = "pdfs_2"

    # Directory where ChromaDB embeddings will be persisted.
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"


    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        # To ensure the PDF source directories exists upon initialization.
        os.makedirs(self.PDF_SOURCE_DIRECTORY_1, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY_1}'.")
        os.makedirs(self.PDF_SOURCE_DIRECTORY_2, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY_2}'.")

# Creates a global instance of the Config class for easy access in the other files.
config = Config()