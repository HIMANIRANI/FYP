import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load PDFs
pdf_paths = [
    'SecuritiesandCommoditiesExchangeMarketRelatedLaws.pdf',
    'SpecializedInvestmentFundRules.pdf',
    'StrategicPlan.pdf',
    'booklet.pdf'
]

all_text = ""

for path in pdf_paths:
    doc = fitz.open(path)
    for page in doc:
        all_text += page.get_text()
    doc.close()

print(f"✅ Loaded and combined {len(pdf_paths)} PDFs.")

# 2. Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(all_text)

print(f"✅ Split into {len(chunks)} text chunks.")

# 3. Create Documents
documents = [Document(page_content=chunk) for chunk in chunks]

# 4. Load Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device':'cuda'}
)

# 5. Create FAISS Vector Store
pdf_vectorstore = FAISS.from_documents(documents, embedding_model)

# 6. Save FAISS Index
save_path = 'pdf_vec'
os.makedirs(save_path, exist_ok=True)
pdf_vectorstore.save_local(save_path)

print(f"✅ PDF vectorstore saved at {save_path}")
