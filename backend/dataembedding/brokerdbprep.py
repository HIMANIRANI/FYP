from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os

# 1. Load broker data
brokers_df = pd.read_csv('brokers_list.csv')  # Adjust if needed

# 2. Prepare documents
documents = []

for idx, row in brokers_df.iterrows():
    broker_no = row['Broker No']
    broker_name = row['Broker Name']
    address = row['Address']
    website = row['Website'] if pd.notna(row['Website']) else "N/A"
    tms = row['TMS'] if pd.notna(row['TMS']) else "N/A"
    
    page_content = f"""
    Broker No: {broker_no}
    Name: {broker_name}
    Address: {address}
    Website: {website}
    TMS URL: {tms}
    """.strip()
    
    documents.append(Document(page_content=page_content))

print(f"✅ Prepared {len(documents)} broker documents!")

# 3. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device':'cuda'}
)

# 4. Create FAISS index
broker_vectorstore = FAISS.from_documents(documents, embedding_model)

# 5. Save FAISS vector store
save_path = 'broker_vec'

# Safely handle directory creation
save_dir = os.path.dirname(save_path)
if save_dir != '':
    os.makedirs(save_dir, exist_ok=True)

broker_vectorstore.save_local(save_path)
print(f"✅ Broker vector store saved to {save_path}!")
