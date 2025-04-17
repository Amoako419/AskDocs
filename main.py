import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from PIL import Image
import io

# Vector store and embeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Document loaders and processors
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LLM and chains
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles the processing of different document types"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor with configurable chunking parameters
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def process_file(self, file_path: str, file_type: str) -> List[Document]:
        """Process a file based on its type
        
        Args:
            file_path: Path to the file
            file_type: Type of file (PDF, TXT, etc.)
            
        Returns:
            List of Document objects after processing and splitting
        """
        logger.info(f"Processing {file_type} file: {file_path}")
        
        try:
            if file_type.lower() == "pdf":
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif file_type.lower() in ["txt", "text"]:
                loader = TextLoader(file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Split into chunks
            split_documents = self.text_splitter.split_documents(documents)
            logger.info(f"Document split into {len(split_documents)} chunks")
            
            return split_documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

class VectorStoreManager:
    """Manages vector embeddings and retrieval"""
    
    def __init__(self, api_key: str):
        """Initialize vector store with embeddings
        
        Args:
            api_key: Google API key for embeddings
        """
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.vector_store = None
        logger.info("Initialized VectorStoreManager with Google embeddings")
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create a FAISS vector store from documents
        
        Args:
            documents: List of processed document chunks
            
        Returns:
            FAISS vector store with embeddings
        """
        if not documents:
            logger.warning("No documents provided to create vector store")
            return None
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, path: str) -> None:
        """Save the vector store to disk
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store:
            logger.info(f"Saving vector store to {path}")
            self.vector_store.save_local(path)
    
    def load_vector_store(self, path: str) -> Optional[FAISS]:
        """Load a vector store from disk
        
        Args:
            path: Directory path to load the vector store from
            
        Returns:
            Loaded FAISS vector store
        """
        try:
            logger.info(f"Loading vector store from {path}")
            self.vector_store = FAISS.load_local(path, self.embeddings)
            return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store from {path}: {str(e)}")
            return None

class RAGSystem:
    """Main RAG system that coordinates document processing, vector stores, and LLM interactions"""
    
    def __init__(self, google_api_key: str):
        """Initialize the RAG system
        
        Args:
            google_api_key: Google API key for Gemini and embeddings
        """
        self.api_key = google_api_key
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager(api_key=google_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        self.chain = None
        logger.info("RAG system initialized")
    
    def process_document(self, file_path: str, file_type: str) -> List[Document]:
        """Process a document and return the chunked documents
        
        Args:
            file_path: Path to the document
            file_type: Type of document
            
        Returns:
            List of processed Document chunks
        """
        return self.document_processor.process_file(file_path, file_type)
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create vector store from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            FAISS vector store
        """
        return self.vector_store_manager.create_vector_store(documents)
    
    def setup_chain(self) -> None:
        """Set up the retrieval chain with the LLM and vector store"""
        if not self.vector_store_manager.vector_store:
            logger.error("Vector store not initialized")
            raise ValueError("Vector store must be initialized before setting up the chain")
        
        logger.info("Setting up retrieval chain")
        try:
            retriever = self.vector_store_manager.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            logger.info("Retrieval chain setup complete")
        except Exception as e:
            logger.error(f"Error setting up chain: {str(e)}")
            raise
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question
        
        Args:
            question: User's question
            
        Returns:
            Dict containing answer and source documents
        """
        if not self.chain:
            logger.error("Chain not initialized")
            raise ValueError("Chain must be initialized before querying")
        
        logger.info(f"Processing query: {question}")
        try:
            response = self.chain({"question": question})
            return response
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise
    
    def process_image(self, image_data: bytes) -> str:
        """Process an image and extract text using the Gemini model
        
        Args:
            image_data: Binary image data
            
        Returns:
            Text extracted from the image
        """
        logger.info("Processing image with Gemini")
        try:
            # Use Gemini directly for images
            image_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key,
                temperature=0.2
            )
            
            # Convert to PIL for processing
            image = Image.open(io.BytesIO(image_data))
            
            response = image_llm.invoke(
                [
                    {"type": "text", "text": "Extract and describe all text content from this image:"},
                    {"type": "image_url", "image_url": {"url": image}}
                ]
            )
            logger.info("Image processing complete")
            return response.content
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

class StreamlitApp:
    """Streamlit UI for the RAG system"""
    
    def __init__(self):
        """Initialize the Streamlit app"""
        # Set page config
        st.set_page_config(
            page_title="Document RAG System",
            page_icon="🧠",
            layout="wide"
        )
        
        # Initialize session state
        if "rag_system" not in st.session_state:
            st.session_state.rag_system = None
        if "vector_store_path" not in st.session_state:
            st.session_state.vector_store_path = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
        
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def run(self):
        """Run the Streamlit app"""
        st.title("📚 Document RAG System")
        
        # Sidebar for configuration
        self._build_sidebar()
        
        # Main app content
        self._build_main_content()
    
    def _build_sidebar(self):
        """Build the sidebar with configuration options"""
        with st.sidebar:
            st.header("Configuration")
            
            # API Key input
            api_key = st.text_input("Google API Key", type="password")
            
            # Document upload
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, TXT, Image)",
                type=["pdf", "txt", "png", "jpg", "jpeg"],
                accept_multiple_files=True
            )
            
            # Process documents button
            if st.button("Process Documents"):
                if not api_key:
                    st.error("Please enter your Google API key")
                elif not uploaded_files:
                    st.error("Please upload at least one document")
                else:
                    with st.spinner("Processing documents..."):
                        self._process_documents(api_key, uploaded_files)
    
    def _process_documents(self, api_key: str, uploaded_files: List) -> None:
        """Process uploaded documents
        
        Args:
            api_key: Google API key
            uploaded_files: List of uploaded files
        """
        try:
            # Initialize RAG system
            rag_system = RAGSystem(google_api_key=api_key)
            st.session_state.rag_system = rag_system
            
            all_docs = []
            
            # Process each file
            for file in uploaded_files:
                # Save the file to a temporary location
                file_path = os.path.join(self.temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                # Determine file type and process accordingly
                file_ext = os.path.splitext(file.name)[1].lower()
                
                if file_ext in [".png", ".jpg", ".jpeg"]:
                    # For images, we'll just keep the path to process directly with the LLM
                    st.sidebar.info(f"Image {file.name} will be processed directly when queried")
                else:
                    # For text and PDF files, process and add to vector store
                    if file_ext == ".pdf":
                        documents = rag_system.process_document(file_path, "pdf")
                    elif file_ext == ".txt":
                        documents = rag_system.process_document(file_path, "txt")
                    else:
                        st.sidebar.warning(f"Unsupported file type: {file_ext}")
                        continue
                    
                    all_docs.extend(documents)
                    st.sidebar.success(f"Processed {file.name} ({len(documents)} chunks)")
            
            # Create vector store if we have documents
            if all_docs:
                vector_store = rag_system.create_vector_store(all_docs)
                
                # Save vector store
                vector_store_path = os.path.join(self.temp_dir, "vector_store")
                rag_system.vector_store_manager.save_vector_store(vector_store_path)
                st.session_state.vector_store_path = vector_store_path
                
                # Setup the chain
                rag_system.setup_chain()
                
                st.session_state.documents_processed = True
                st.sidebar.success(f"Successfully processed {len(all_docs)} document chunks")
            
        except Exception as e:
            st.sidebar.error(f"Error processing documents: {str(e)}")
            logger.error(f"Error processing documents: {str(e)}")
    
    def _build_main_content(self):
        """Build the main content area"""
        # Display chat interface if documents are processed
        if st.session_state.documents_processed:
            st.subheader("Ask questions about your documents")
            
            # Query input with callback to clear after submission
            if "query_submitted" not in st.session_state:
                st.session_state.query_submitted = False
            
            # Create a callback to handle form submission and clear input
            def submit_query():
                if st.session_state.query_input.strip():
                    st.session_state.query_submitted = True
            
            # Query input
            query = st.text_input("Your question:", key="query_input", on_change=submit_query)
            
            if st.session_state.query_submitted:
                self._process_query(query)
            
            # Display chat history
            self._display_chat_history()
        else:
            st.info("Please upload and process documents in the sidebar to get started.")
            
            # Demo instructions
            st.markdown("""
            ### How to use this RAG system:
            
            1. Enter your Google API key in the sidebar
            2. Upload one or more documents (PDF, TXT) or images
            3. Click "Process Documents" to analyze and index the content
            4. Ask questions about your documents in the chat interface
            
            The system will retrieve relevant information from your documents to answer your questions.
            """)
    
    def _process_query(self, query: str) -> None:
        """Process a user query
        
        Args:
            query: User's question
        """
        try:
            with st.spinner("Thinking..."):
                rag_system = st.session_state.rag_system
                response = rag_system.query(query)
                
                # Add to chat history
                st.session_state.chat_history.append({"question": query, "answer": response})
                
                # Instead of clearing through session_state, we'll handle this differently
                
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Error processing query: {str(e)}")
    
    def _display_chat_history(self):
        """Display the chat history"""
        if not st.session_state.chat_history:
            return
        
        st.subheader("Conversation History")
        
        for idx, exchange in enumerate(st.session_state.chat_history):
            question = exchange["question"]
            response = exchange["answer"]
            
            # Display question
            with st.chat_message("user"):
                st.write(question)
            
            # Display answer
            with st.chat_message("assistant"):
                st.write(response["answer"])
                
                # Display sources if available
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"```\n{doc.page_content[:300]}...\n```")
            
            # Add separator
            if idx < len(st.session_state.chat_history) - 1:
                st.divider()

def main():
    """Main entry point for the application"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()