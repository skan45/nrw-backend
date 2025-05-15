import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined PDF path
PDF_PATH = "NRW_data .pdf"

# Website URL (replace with actual NRW Congress website URL)
WEBSITE_URL = "https://nrw.ieee.tn/Technical-challenges"

def get_pdf_text():
    text = ""
    try:
        pdf_reader = PdfReader(PDF_PATH)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
    return text

def get_web_text():
    text = ""
    try:
        response = requests.get(WEBSITE_URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from relevant HTML elements (e.g., <p>, <h1>, <h2>, etc.)
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            text += element.get_text(strip=True) + " "
        
        # Save the extracted text to a file
        with open("web_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Web content saved to web_content.txt")
            
    except requests.RequestException as e:
        print(f"Error fetching website content: {str(e)}")
    return text

def get_combined_text():
    pdf_text = get_pdf_text()
    web_text = get_web_text()
    combined_text = pdf_text + " " + web_text
    return combined_text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert assistant for the National Robotics Weekend (NRW) Congress, organized by the IEEE Robotics and Automation Society (RAS) Chapter at INSAT. Your role is to provide accurate, concise, and professional responses based on data from the NRW event PDF and the official NRW Congress website. Answer questions about the IEEE RAS, NRW Congress, its events (e.g., Makeathon, technical challenges, speakers), and related details. Use Markdown formatting to emphasize key terms and headings (e.g., *Entrepreneurship Challenge*, *Problem Identification*) in bold, while keeping descriptive text in normal text for clarity. If a question is unclear or outside the provided context, politely clarify or suggest relevant topics.

    Guidelines:
    - Use a professional and engaging tone, suitable for students, researchers, and event participants.
    - For questions about the *Entrepreneurship Challenge*, format the response to include:
      - The challenge title in bold: *Entrepreneurship Challenge*
      - Key skills in bold (e.g., *Problem Identification*, *Ideation*, *Business Planning*, *Pitching*, *Teamwork and Communication*, *Project Management*)
      - Descriptive text in normal text, explaining each skill or the challenge details.
    - Provide clear, factual answers, prioritizing key details from the PDF and website (e.g., dates, speaker names, challenge rules).
    - If information differs between the PDF and website, prioritize the website as the most current source and note any discrepancies politely.
    - Avoid speculative answers; if information is missing, state so politely and offer related insights.
    - Emphasize innovation, collaboration, and robotics advancement when relevant.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def initialize_vector_store():
    if not os.path.exists("faiss_index"):
        print("Initializing vector store...")
        combined_text = get_combined_text()
        if not combined_text.strip():
            print("No content retrieved from PDF or website. Please check the sources.")
            return
        text_chunks = get_text_chunks(combined_text)
        get_vector_store(text_chunks)
        print("Vector store initialized successfully!")

def main():
    print("Welcome to the NRW Chatbot!")
    print("Initializing the system...")
    
    # Initialize vector store
    initialize_vector_store()
    
    print("\nYou can now start asking questions. Type 'quit' to exit.")
    
    while True:
        user_question = input("\nYour question: ")
        
        if user_question.lower() == 'quit':
            print("Goodbye!")
            break
            
        try:
            response = get_response(user_question)
            print("\nBot:", response)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if _name_ == "_main_":
    main()