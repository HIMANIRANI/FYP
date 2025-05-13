from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os

# 1. Load fundamental data
fundamental_df = pd.read_csv('fundamental_data.csv')

# 2. Prepare documents
documents = []

for idx, row in fundamental_df.iterrows():
    company_symbol = row['Company Symbol']
    company_name = row['Company Name']
    sector = row['Sector']
    shares_outstanding = row['Shares Outstanding'] if pd.notna(row['Shares Outstanding']) else "N/A"
    eps = row['EPS'] if pd.notna(row['EPS']) else "N/A"
    pe_ratio = row['P/E Ratio'] if pd.notna(row['P/E Ratio']) else "N/A"
    book_value = row['Book Value'] if pd.notna(row['Book Value']) else "N/A"
    pbv = row['PBV'] if pd.notna(row['PBV']) else "N/A"
    market_cap = row['Market Capitalization'] if pd.notna(row['Market Capitalization']) else "N/A"
    dividend = row['% Dividend'] if pd.notna(row['% Dividend']) else "N/A"
    bonus = row['% Bonus'] if pd.notna(row['% Bonus']) else "N/A"
    right_share = row['Right Share'] if pd.notna(row['Right Share']) else "N/A"
    
    page_content = f"""
    Company: {company_name} ({company_symbol})
    Sector: {sector}
    Shares Outstanding: {shares_outstanding}
    EPS: {eps}
    P/E Ratio: {pe_ratio}
    Book Value: {book_value}
    PBV: {pbv}
    Market Capitalization: {market_cap}
    Dividend: {dividend}
    Bonus Share: {bonus}
    Right Share: {right_share}
    """.strip()
    
    documents.append(Document(page_content=page_content))

print(f"✅ Prepared {len(documents)} fundamental documents!")

# 3. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device':'cuda'}
)

# 4. Create FAISS index
fundamental_vectorstore = FAISS.from_documents(documents, embedding_model)

# 5. Save FAISS vector store
save_path = 'fundamental_vec'

# Safely handle directory creation
save_dir = os.path.dirname(save_path)
if save_dir != '':
    os.makedirs(save_dir, exist_ok=True)

fundamental_vectorstore.save_local(save_path)
print(f"✅ Broker vector store saved to {save_path}!")