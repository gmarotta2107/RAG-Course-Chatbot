import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
import os
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import io
from google.api_core import exceptions
import time 

genai.configure(api_key="YOUR_API")


############################################TESTO###########################################
import os

def extract_text_from_directory(input_directory, output_directory="output_texts", result_filename="result_pre_pre.txt"):
    texts = ""
    
    # Controlla se la directory di input esiste
    if not os.path.isdir(input_directory):
        return "Errore: La directory di input non esiste."
    
    # Crea la directory di output se non esiste
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        if os.path.isfile(file_path):  # Controlla se è un file
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    texts += "\n" + content
                    
                    # Salva il contenuto in un nuovo file nella cartella di output
                    output_file_path = os.path.join(output_directory, filename)
                    with open(output_file_path, "w", encoding="utf-8") as output_file:
                        output_file.write(content)
            except Exception as e:
                texts = f"Errore: {e}"
    
    # Salva il risultato aggregato in result_pre_pre.txt
    with open(result_filename, "w", encoding="utf-8") as result_file:
        result_file.write(texts)

    return texts  # Restituisce il testo aggregato




##########################################################################################







def process_text_and_images_with_model(model,  images_by_page):
    """Passa il testo e le immagini estratte al modello AI per l'elaborazione."""
    image_descriptions_by_page = []
    
    try:
        # Elabora le immagini con ritardo
        for page_images in images_by_page:
            page_descriptions = []
            for image_bytes in page_images:
                description = describe_image_with_model(model, image_bytes)
                if description:
                    page_descriptions.append(description)
            image_descriptions_by_page.append(page_descriptions)
    
    except exceptions.ResourceExhausted:
        print("Quota API completamente esaurita. Interrompo l'elaborazione.")
    
    return  image_descriptions_by_page

def describe_image_with_model(model, image_bytes):
    """Passa l'immagine al modello AI per la descrizione con un prompt specifico."""
    try:
        # Converti i byte dell'immagine in un oggetto PIL.Image
        image = Image.open(io.BytesIO(image_bytes))
        prompt = (
            "Provide a title for the image that indicates what it represents(e.g., if the architecture of a transformer is depicted, the title could be 'Transformer Architecture Description')"
            "Describe this image in a detailed and technical way. "
            " -If the image represents an architecture (such as a transformer, neural network, or complex system),describe the main components, data flow, and the role of each part. "
            " -If the image represents a process or flowchart, explain each step in the process and how the steps are related to each other. "
            " -If the image represents an abstract concept (such as an embedding or vector representation), describe the meaning of the concept and how it is applied in context. "
            " -If the image contains mathematical formulas or equations, describe their meaning and their role in the overall context. "
            " -If the image is a graph (such as a bar, line, or scatter plot), explain what the axes, main data, and observable trends represent. "
            " -If the image is a snippet of code, explain the code and transcribe it whithout losing information. "
            "- If the image is an irrelevant elements such as logos, decorative images or photos without technical context completely IGNORE IT (you don't have write anything). "
            "Be precise, technical, and do not add information not in the image."
            "Don't start the description with phrases like 'The image describe' , 'The image is', you have just to describe it as if it were normal text."
            "separate the description of each image with a special character like ============================ and go to the next line"
        )
        # Passa sia il prompt che l'immagine al modello
        response = model.generate_content([prompt, image])
        return response.text  # Restituisce una stringa
    except exceptions.ResourceExhausted:  # Eccezione correttamente importata
        print("Quota API esaurita, prova a ridurre il numero di richieste.")
        return None
    except Exception as e:
        print(f"Errore durante l'elaborazione dell'immagine: {e}")
        return None
    

def read_single_pdf(pdf_path):
    """Legge il contenuto di un singolo file PDF e restituisce il testo e le immagini con la loro posizione."""
    print(f"Caricando il file: {os.path.basename(pdf_path)}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Estrai il contenuto testuale del PDF
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Estrai le immagini dal PDF e associale alle pagine
    doc = fitz.open(pdf_path)
    images_by_page = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        images = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
        
        images_by_page.append(images)
    
    return  images_by_page


def save_response_to_file(image_descriptions_by_page, output_filename):
    """Salva le descrizioni delle immagini in un file .txt."""
    with open(output_filename, "w", encoding="utf-8") as file:
        for page_num, page_descriptions in enumerate(image_descriptions_by_page):
            if page_descriptions:
                file.write(f"\n--- Descrizioni delle immagini (Pagina {page_num + 1}) ---\n")
                for desc in page_descriptions:
                    file.write(desc + "\n")
    
    print(f"Risultato salvato in '{output_filename}'.")






#salvo i testi presi dai pdf 
extract_text_from_directory("src/","output_txt/")


#salvo le descrizioni delle immagini come testo 
pdf_directory = "src/"
pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
model = genai.GenerativeModel('gemini-2.0-flash')

for pdf_filename in pdf_files:
            try:  # Aggiunto blocco try per gestire errori per file
                pdf_path = os.path.join(pdf_directory, pdf_filename)
                extracted_images = read_single_pdf(pdf_path)
                
                if extracted_images:
                    image_descriptions = process_text_and_images_with_model(model, extracted_images)
    
                    output_filename = f"images_{os.path.splitext(pdf_filename)[0]}.txt"
                    save_response_to_file(image_descriptions, output_filename)
                else:
                    print(f"Nessuna immagine trovata in {pdf_filename}, salto il file.")


        
                time.sleep(3)  # Ritardo aggiuntivo tra i file
                
            except Exception as e:
                print(f"Errore durante l'elaborazione di {pdf_filename}: {e}")

                continue
