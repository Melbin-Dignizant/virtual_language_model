import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

# Optional for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf(pdf_path, ocr_fallback=True):
    """
    Extract text from a PDF. Uses direct extraction if possible,
    otherwise applies OCR on images.
    
    Args:
        pdf_path (str): Path to the PDF file.
        ocr_fallback (bool): Whether to apply OCR on image-based PDFs.
        
    Returns:
        str: Extracted text content.
    """
    all_text = ""

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                all_text += f"\n--- Page {page_num} (Text Layer) ---\n{text}"
            elif ocr_fallback:
                print(f"[INFO] No text layer found on page {page_num}. Applying OCR.")
                # Convert single page to image
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    all_text += f"\n--- Page {page_num} (OCR) ---\n{ocr_text}"
    except Exception as e:
        all_text += f"\n[ERROR] Failed to process PDF: {e}"

    return all_text.strip()

# Example usage
if __name__ == "__main__":
    pdf_file = r"D:\Melbin\VLM-Examples\static\pdf\02_page_2.pdf"  # Replace with your file
    extracted_text = extract_text_from_pdf(pdf_file)
    print("Final Extracted Text:\n", extracted_text)
