from langchain_community.document_loaders import PyPDFLoader
import os
import google.generativeai as genai

def read_single_pdf(pdf_path):
    """Legge il contenuto di un singolo file PDF e restituisce il testo."""
    print(f"Caricando il file: {os.path.basename(pdf_path)}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Estrai il contenuto testuale del PDF
    full_text = "\n".join([doc.page_content for doc in docs])
    
    return full_text

def process_text_with_model(model, text):
    """Passa il testo estratto al modello AI per l'elaborazione."""
    try:
        prompt = f"Rewrite this text more clearly, do not add information but enrich the text and rewrite it in detail without summarizing or skipping anything, In english:\n{text}"
        response = model.generate_content(prompt)
        return response.text  # Restituisce una stringa
    except google.api_core.exceptions.ResourceExhausted:
        print("Quota API esaurita, prova a ridurre il numero di richieste.")
        return None

def save_response_to_file(response, output_filename):
    """Salva il testo elaborato in un file .txt."""
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(response)
    print(f"Risultato salvato in '{output_filename}'.")

if __name__ == "__main__":
    genai.configure(api_key="AIzaSyCD_erV8-h5Uoa-A-rKz3_KJk8xrfO_n0Y")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    pdf_directory = "LLM_doc/"

    # Scansiona tutti i file nella cartella
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print("Nessun file PDF trovato nella directory.")
    else:
        print(f"Trovati {len(pdf_files)} file PDF. Inizio elaborazione...\n")
        
        for pdf_filename in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_filename)
            extracted_text = read_single_pdf(pdf_path)
            
            if extracted_text:
                response = process_text_with_model(model, extracted_text)
                if response:
                    output_filename = f"risultato_{os.path.splitext(pdf_filename)[0]}.txt"
                    save_response_to_file(response, output_filename)
