import requests
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Get the OCR API key
ocr_api_key = os.getenv("OCR_API_KEY")


def ocr_space_file(filename, overlay=False, api_key=None, language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    If None, will use the global ocr_api_key variable.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'eng'.
    :return: Result in JSON format.
    """
    
    # Use the global API key if none provided
    if api_key is None:
        api_key = ocr_api_key
    
    if not api_key:
        raise ValueError("API key is required. Please set OCR_API_KEY in your .env file.")
    
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
    }
    
    try:
        with open(filename, 'rb') as f:
            # Use 'file' as the key name for the file upload
            r = requests.post('https://api.ocr.space/parse/image',
                              files={'file': f},
                              data=payload,
                              )
        
        # Check if request was successful
        r.raise_for_status()
        
        # Parse JSON response
        response_data = r.json()
        return response_data
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {r.text}")
        return None


def ocr_space_url(url, overlay=False, api_key=None, language='eng'):
    """ OCR.space API request with remote file.
        Python3.5 - not tested on 2.7
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    If None, will use the global ocr_api_key variable.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'eng'.
    :return: Result in JSON format.
    """
    
    # Use the global API key if none provided
    if api_key is None:
        api_key = ocr_api_key
    
    if not api_key:
        raise ValueError("API key is required. Please set OCR_API_KEY in your .env file.")
    
    payload = {
        'url': url,
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
    }
    
    try:
        r = requests.post('https://api.ocr.space/parse/image',
                          data=payload,
                          )
        
        # Check if request was successful
        r.raise_for_status()
        
        # Parse JSON response
        response_data = r.json()
        return response_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {r.text}")
        return None


def extract_text_from_response(response):
    """Extract text from OCR.space API response"""
    if not response:
        return None
    
    try:
        if 'ParsedResults' in response and len(response['ParsedResults']) > 0:
            return response['ParsedResults'][0]['ParsedText']
        else:
            print("No text found in response")
            return None
    except (KeyError, IndexError) as e:
        print(f"Error extracting text: {e}")
        return None


# Usage examples:
if __name__ == "__main__":
    # Test with file
    test_file_response = ocr_space_file(
        filename=r'D:\Melbin\VLM-Examples\static\pdf\02_page_2.jpg', 

    )
    
    if test_file_response:
        print("File OCR Response:")
        print(json.dumps(test_file_response, indent=2))
        
        # Extract just the text
        extracted_text = extract_text_from_response(test_file_response)
        if extracted_text:
            print("\nExtracted Text:")
            print(extracted_text)
    
    # Test with URL (uncomment to use)
    # test_url_response = ocr_space_url(url='http://i.imgur.com/31d5L5y.jpg')
    # if test_url_response:
    #     print("\nURL OCR Response:")
    #     print(json.dumps(test_url_response, indent=2))