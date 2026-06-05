
Multi-modal RAG System 
===
  1. Ingestion
  -------------
    1.1. Read the pdf 
    1.2 Find the layout 
      1.2.1 Find the element type 
      1.2.2 if text, extract text 
      1.2.3 if table, convert to dataframe 
      1.2.4 if images, convert the image to base64 or get captions from VLM 
      1.2.5 if scanned content, use OCR or VLM
      1.2.6 if header / footer, avoid duplicate chunks 

    1.3. Chunking (size, overlap)
    1.4. Metadata extraction
    1.5. Embedding Model (dimensionality)
    1.6. Saved in DB 



  2. Retrieval 
