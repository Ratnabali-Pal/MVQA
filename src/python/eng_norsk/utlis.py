# utlis.py

from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
import config

def get_translate_client():
    """
    Initializes and returns the Google Translate client with credentials.

    Returns:
        google.cloud.translate_v2.client.Client: The Google Translate client.
    """
    credentials = service_account.Credentials.from_service_account_file(config.GOOGLE_APPLICATION_CREDENTIALS)
    translate_client = translate.Client(credentials=credentials)
    return translate_client

def translate_text(client, text, source_lang, target_lang):
    """
    Translates a given text to the target language.

    Args:
        client (google.cloud.translate_v2.client.Client): The Google Translate client.
        text (str): The text to be translated.
        source_lang (str): The source language code.
        target_lang (str): The target language code.

    Returns:
        str: The translated text.
    """
    result = client.translate(text, source_language=source_lang, target_language=target_lang)
    return result['translatedText']

def translate_dataset(data):
    """
    Translates the 'question' and 'answers' fields in the dataset.

    Args:
        data (list): A list of dictionaries representing the dataset.

    Returns:
        list: The translated dataset.
    """
    translate_client = get_translate_client()
    
    for i, item in enumerate(data):
        # Translate the question
        item['question'] = translate_text(translate_client, item['question'], config.SOURCE_LANGUAGE, config.TARGET_LANGUAGE)
        
        # Translate the answers
        for answer in item['answers']:
            answer['answer'] = translate_text(translate_client, answer['answer'], config.SOURCE_LANGUAGE, config.TARGET_LANGUAGE)
        
        print(f"Translated item {i+1}/{len(data)}")
        
    return data