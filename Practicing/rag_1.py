import os

# from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

current_path = os.path.dirname(os.path.abspath(__file__))

file_path = r'c:\Users\HP\OneDrive\Documents\Books\Master Your Emotions.pdf'

db_path = os.path.join(current_path , 'db' , 'chroma_db')
# db_path = r'c:\Users\HP\OneDrive\Documents\Books\Master Your Emotions.pdf'


if os.path.exists(db_path):
    print('vector database already created')
else:
    if not file_path:
        print('txt file does not exists')
    else:
        # load the file
        document_loader = PyMuPDFLoader(file_path)
        document = document_loader.load()

        # split the documents into chunks

        text_splitter = CharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
        docs = text_splitter.split_documents(document)

        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        print(f"Sample chunk:\n{docs[0].page_content}\n")

        # now for initalizing the embedding models

        model_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004" , google_api_key='AIzaSyA77gUQw_Fzk2L4hJx_6fzQOSZipJn_ZTg'
        )

        print('----------------start creating vector db----------------')

        db = Chroma.from_documents(
            docs , model_embeddings , persist_directory=db_path
        )

        # db = Chroma(persist_directory=db_path,
        #     embedding_function=model_embeddings)

        print('----------------Finished creating vector db----------------')

        query = 'what is happiness'

        retriever = db.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k': 3 , 'score_threshold' :0.3}
        )   

        docs = retriever.invoke(query)

        for i , doc in enumerate(docs, 1):
            print(f'document no {i}. \n\n {doc}\n')
            if doc.metadata:
                print(f"source {doc.metadata.get('source' , 'Unknown')}\n")