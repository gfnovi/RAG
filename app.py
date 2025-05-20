# import main Flask class and request object
import base64
import io
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from flask import Flask, json, jsonify, request
import pymupdf
import numpy as np
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
np.float_ = np.float64
# create the Flask app
app = Flask(__name__)

folder_path = "db"
image_folder = "pdf_images"
cached_llm = OllamaLLM(model="llama3:8b")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)
# and finally give a confidence_score from 0-100 percent for your answer. Use white space between words.
raw_prompt = PromptTemplate.from_template(
    """
    Use the provided context to answer the user's question.
    If you don't know the exact answer from the provided context. 
    Do not use your prior knowledge.
    The answer must be relevant to the query.


    Context: {context}
    Question: {input}
    Answer:
    """
)

def extract_images_from_pdf(pdf_path, pdf_filename):
    """Extract images + OCR text from PDF"""
    doc = pymupdf.open(pdf_path)
    image_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Generate unique filename
            image_filename = f"{pdf_filename}_page{page_num+1}_img{img_index+1}.{base_image['ext']}"
            image_path = os.path.join("pdf_images", image_filename)

            # Save image to file
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Extract text from image using OCR
            extracted_text = extract_text_from_image(image_bytes)

            image_data.append({
                "page": page_num + 1,
                "index": img_index + 1,
                "filename": image_filename,
                "path": image_path,
                "format": base_image['ext'],
                "text": extracted_text,  # Extracted OCR text
                "base64": base64.b64encode(image_bytes).decode('utf-8')
            })

    return image_data


def extract_text_from_image(image_bytes):
    """Extract text from an image using OCR (Tesseract)"""
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        # Apply OCR
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""


@app.route("/askPorter", methods=["POST"])
def ask_porter():
    print("Post /askPdf called")
    json_content = request.json
    companyName = json_content.get("companyName")
    query = """
        Generate a Porter's Five Forces analysis for %s group in JSON format with the following structure. 
        Include an score (-2,-1,0,1,2) -2 for the very bad, 0 for neutral, 2 very good. Becareful when scoring, because the 2 means that this company is perfect and -2 means that this company is very bad can't be saved.
        and a detailed explanation (2-5 sentences) can give numeric support data for each sub-key . 
        On the output don't give any brief, just go straight to the output format. 
        Use this template:

        Main Keys & Sub-Keys:
        1. Threat of New Entrants
            -Industry Growth Rate
            -Access to Distribution Channel
            -Capital Requirement
            -How well incumbents protect their market shares
            -Economies of Scale
            -Regulations / Government Policies

        2. Competitive Rivalry
            -Number of Competitors (Portfolio)
            -Number of Competitors (Fund)
            -Relative Sizes of Competitors
            -Strategies to Mitigate Competition
            -Brand Loyalty (Product)
            - Product Differentiation

        3. Supplier Power
            -Number of Suppliers
            -Switching Cost (FL)
            -Threat of Forward Integration
            -Access to Potential Company

        4.Buyer Power
            -Number of Buyers
            -Availability of Substitutes
            -Buyer Switching Cost (fund)
            -Buyers' Perception of the Brand

        5.Threat of Substitutes
            -Number of Substitutes
            -Relative Quality of Substitutes

        Output format :
        {
            "threat of newentrants": {
                "industry growth rate": {"score": "-2/-1/0/1/2", "explanation": "..."},
                "access to distribution channel": {"score": "...", "explanation": "..."},
                ...
                },
            "competitive rivalry": { ... },
            "supplier power": { ... },
            "buyer power": { ... },
            "threat of substitutes": { ... }
        }

        Fill in each field with a realistic score based on general industry knowledge or specify an industry if needed. And makesure the json format is closed with }
    """ % (companyName)

    print(f"query: {query}")
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path,
                          embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    print('chain created 3')

    result = chain.invoke({"input": query})
    print('chain invoke')

    print(result)
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"],
                "page_content": doc.page_content},
        )
    # response_answer = {"answer": result["answer"], "sources": sources}
    return result["answer"]


@app.route("/askPdf", methods=["POST"])
def ask_pdf():
    print("Post /askPdf called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path,
                          embedding_function=embedding)
    print("Creating chain")
    retriever = vector_store.as_retriever(
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    print('chain created 3')

    result = chain.invoke({"input": query})
    print('chain invoke')

    print(result)
    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"],
                "page_content": doc.page_content},
        )
    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdfImage", methods=["POST"])
def pdfImagePost():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_name = file.filename
    save_file = "pdf/" + file_name
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    file.save(save_file)
    print(f"filename: {file_name}")

    # Extract images from PDF +OCR
    images = extract_images_from_pdf(save_file, file_name)
    print(f"Extracted {len(images)} images from PDF")

    # Process text content
    loader = PyPDFLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")
    for img in images:
        if img["text"]:  # If OCR extracted text
            docs.append(Document(
                page_content=img["text"],
                metadata={
                    "source": f"Image: {img['filename']}",
                    "page": img["page"]
                }
            ))
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
        "images_extracted": len(images),
        "images": images[:5],  # Return first 5 images metadata (avoid huge response)
        "message": "Full image data available in response if needed"
    }
    return jsonify(response)


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PyPDFLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


@app.route("/askAi", methods=["POST"])
def ask_ai():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")
    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return response_answer


if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
