from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import pdfplumber
import tiktoken
import re
import json

import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API"


from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document


import tiktoken
import re
import json
import nltk
#nltk.download()

import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API"



def extract_text_from_directory(directory):
    texts = ""
    if not os.path.isdir(directory):
        return "Errore: La directory non esiste."

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Controlla se è un file
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    texts = texts + "\n" + file.read()
            except Exception as e:
                texts = f"Errore: {e}"

    return texts  # Restituisce un dizionario con nome file e contenuto
    
# 3️⃣ Divisione in chunk semantici
def split_into_semantic_chunks(text, max_tokens=500, overlap=50):
    """Divide il testo in chunk che rispettano i confini delle frasi e non dividono troppo il contenuto."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sentences = nltk.sent_tokenize(text)  # Suddivide il testo in frasi

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)

        # Verifica se l'aggiunta della frase supererebbe la lunghezza massima del chunk
        if current_length + len(sentence_tokens) > max_tokens:
            if current_chunk:  # Se ci sono frasi nel chunk corrente, aggiungile al risultato
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]  # Inizia un nuovo chunk con la frase attuale
            current_length = len(sentence_tokens)  # Imposta la lunghezza del nuovo chunk
        else:
            current_chunk.append(sentence)  # Aggiungi la frase al chunk corrente
            current_length += len(sentence_tokens)  # Aggiungi la lunghezza della frase al chunk

    # Aggiungi l'ultimo chunk, se non è vuoto
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"🔹 Divisi in {len(chunks)} chunk.")  # Debug sul numero di chunk creati
    return chunks

# 4️⃣ Processamento AI in batch
def process_chunks_with_ai(chunks, batch_size=5):
    """Invia i chunk all'AI in batch per ridurre il numero di chiamate"""
    processed_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        prompt = f"Rielabora i seguenti chunk di testo per migliorarne la chiarezza e la coerenza:\n\n"
        for chunk in batch:
            prompt += f"{chunk}\n"
        
        print(f"🧠 Inviando batch di {len(batch)} chunk per rielaborazione...")  # Debug per la dimensione del batch
        # Simulazione chiamata all'AI (da sostituire con API reale)
        improved_batch = ai_call(prompt)  
        
        print(f"Batch rielaborato contiene {len(improved_batch)} chunk.")  # Debug per il numero di chunk ricevuti
        processed_chunks.extend(improved_batch)
    
    print(f"✅ Rielaborati {len(processed_chunks)} chunk con AI.")  # Debug finale
    return processed_chunks


def ai_call(prompt):
    """Chiamata all'AI per migliorare i chunk"""
    # Invia la query al modello
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    messages = [
        "system",
        f"""
            Rework these chunks of text to improve clarity and coherence. Every chunk.
            Every chunk will then be used for a RAG system so optimize chunks for retrieval.
            ***RULES***
            split each chunk by going to a new line do not go to a new line if it is the same chunk if for example you return 3 chunks you must do it in this format:
            "text of the first chunk".
            "text of the second chunk".
            "text of the third chunk".
            avoid telling me anything else just give me the reworked chunks.
            always write in English.

            here are the chunks to rework: {prompt}
        """,
    ]
    response = llm.invoke(messages)
    
    # Ottieni il contenuto della risposta (che è una lunga stringa)
    improved_text = response.content  # Questa è la risposta di Gemini come stringa
    
    # Separiamo il testo in chunk utilizzando un delimitatore (ad esempio \n per separare i chunk)
    improved_chunks = improved_text.split("\n")  # Cambia il delimitatore se necessario
    
    # Debug per il numero di chunk ricevuti dalla risposta AI
    print(f"AI ha restituito {len(improved_chunks)} chunk rielaborati.")

    return improved_chunks


# 5️⃣ Creazione di metadati e salvataggio in JSON
def process_document(directory_path, output_json="chunks.json"):
    """Esegue l'intero preprocessing del documento PDF"""
    #raw_text = extract_text_from_pdf(pdf_path)
    #cleaned_text = clean_text(raw_text)
    cleaned_text = extract_text_from_directory(directory_path)
    
    # Modifica qui per ottenere chunk di dimensioni adeguate
    chunks = split_into_semantic_chunks(cleaned_text, max_tokens=500, overlap=50)

    # Processa i chunk con l'AI in batch
    improved_chunks = process_chunks_with_ai(chunks)

    # Aggiungi i chunk migliorati a FAISS
    structured_chunks = []
    for i, chunk in enumerate(improved_chunks):
        if chunk:  # Evita chunk vuoti
           structured_chunks.append({
            "chunk_id": i,
            "text": chunk,  # Usa solo il singolo chunk qui
            #"source": "manuale utente",
            #"argomento": "dichiarazioni annuali"
        })


    # Salva i chunk in JSON per debugging
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_chunks, f, indent=4, ensure_ascii=False)

    print(f"✅ Preprocessing completato! {len(improved_chunks)} chunk salvati in {output_json}")
    return improved_chunks

# 6️⃣ Creazione o caricamento dell’indice FAISS
def load_or_create_faiss_index(pdf_directory, index_name="faiss_index", chunks=None):
    """
    Carica un indice FAISS esistente o ne crea uno nuovo se non esiste.
    Se i chunk rielaborati sono forniti, li aggiunge all'indice.
    """
    embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={'normalize_embeddings': True})

    if os.path.exists(index_name):
        print(f"📥 Caricamento dell'indice FAISS esistente da '{index_name}'...")
        vectorstore = FAISS.load_local(index_name, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("📌 Creazione di un nuovo indice FAISS...")

        if chunks is None:
            loader = PyPDFDirectoryLoader(pdf_directory)
            docs = loader.load()
            print(f"📄 Numero di pagine caricate: {len(docs)}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            print(f"🔹 Numero di chunk creati: {len(chunks)}")

        # Converte i chunk in oggetti Document
        documents = [Document(page_content=chunk, metadata={"id": str(i)}) for i, chunk in enumerate(chunks)]
        
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(index_name)
        print(f"✅ Indice FAISS salvato in '{index_name}'.")

    if chunks:
        print("🔄 Aggiungendo i nuovi chunk rielaborati all'indice FAISS...")

        # Converte i chunk migliorati in oggetti Document
        document_chunks = [Document(page_content=chunk, metadata={"id": str(i)}) for i, chunk in enumerate(chunks)]

        vectorstore.add_documents(document_chunks)
        vectorstore.save_local(index_name)
        print(f"✅ Nuovi chunk rielaborati aggiunti all'indice FAISS.")

    return vectorstore

# 7️⃣ Creazione/Aggiornamento dell'indice FAISS con i chunk rielaborati
def update_faiss_index(directory,  index_name="faiss_index2"):
    """
    Aggiorna l'indice FAISS con nuovi PDF, includendo i chunk rielaborati.
    """
    # Processamento del documento e miglioramento dei chunk
    improved_chunks = process_document(directory)

    # Carica o crea l'indice FAISS e aggiungi i chunk rielaborati
    vectorstore = load_or_create_faiss_index(directory, index_name, chunks=improved_chunks)

    print(f"✅ Indice FAISS aggiornato con {len(improved_chunks)} chunk rielaborati.")
    return vectorstore


index_name = "index_for_all2"
pdf_directory= "risultati/"

# 1️⃣ Preprocessing del PDF singolo con AI e aggiornamento dell'indice FAISS
vectorstore = update_faiss_index(pdf_directory, index_name)

