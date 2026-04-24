import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Chargement de la configuration .env
load_dotenv()

class RAGSystem:
    """Système de Retrieval-Augmented Generation (RAG)."""
    
    def __init__(self, data_dir="data", index_path="faiss_index"):
        self.data_dir = data_dir
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Initialiser l'index au démarrage
        self.initialize_index()

    def initialize_index(self):
        """Charge les documents, les indexe et sauvegarde l'index FAISS."""
        if os.path.exists(self.index_path):
            print(f"Chargement de l'index existant depuis {self.index_path}")
            self.vector_store = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("Création d'un nouvel index...")
            self.refresh_index()

    def refresh_index(self):
        """Re-scanne le dossier data et reconstruit l'index pour les fichiers .txt et .pdf."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        documents = []
        
        # Chargement des fichiers .txt
        txt_loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
        
        # Chargement des fichiers .pdf
        pdf_loader = DirectoryLoader(self.data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        
        if not documents:
            print("Aucun document trouvé pour l'indexation.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.index_path)
        print(f"Index mis à jour avec {len(chunks)} fragments provenant de {len(documents)} documents.")

    def search(self, query: str, k: int = 3):
        """Effectue une recherche de similarité dans l'index."""
        if not self.vector_store:
            return "Aucun index disponible."
        
        results = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])

# Instance globale pour être utilisée par l'agent
rag_system = RAGSystem()
