import pytesseract
import pyautogui
import keyboard
import requests
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from email.utils import parseaddr
import re
from urllib.parse import urlparse
import time
from difflib import SequenceMatcher


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)



def capture_and_extract_text():

    email_region = (637, 187, 596, 69)
    screenshot = pyautogui.screenshot(region=email_region)
    screenshot.save("email_region.png")
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(screenshot, config=custom_config)

def capture_and_extract_text_full_screen():

    content = (623, 263, 1883-623, 1016-263)
    screenshot_full_screen = pyautogui.screenshot(region=content)
    screenshot_full_screen.save("email_content.png")
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(screenshot_full_screen, config=custom_config)

def extract_domain(email):
    if '<' in email and '>' in email:
        email = email.split('<')[-1].split('>')[0]
    return email.split('@')[-1].lower().strip()


def is_phishing(email):
    trusted_domains = ["duolingo.com"]
    domain = extract_domain(email)
    return domain not in trusted_domains


def process_email(text,text_full_screen):
    email_match = re.search(r'(?:<)?([\w\.-]+@[\w\.-]+)(?:>)?', text)
    
    if not email_match:
        print("Nenhum e-mail válido encontrado no texto")
        return

    email = email_match.group(1)
    domain = extract_domain(email)
    print(f"Email encontrado: {email}")
    print(f"Domínio extraído: {domain}")
    
    if is_phishing(email):
        print("ALERTA DE PHISHING! Motivos:")
        print(f"- Domínio '{domain}' não está na lista de confiáveis")
    else:
        print(f"✅ Domínio '{domain}' verificado com sucesso")
    
    print("\nDomínios confiáveis: duolingo.com, twitter.com, twitch.tv")


    if len(text.split()) < 5:  
        print("\n⚠ Texto muito curto para análise de conteúdo")
        return
    
    try:
        inputs = tokenizer(text_full_screen, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            phishing_prob = probabilities[1].item()
            
            print(f"\n🔍 Análise de Conteúdo:")
            print(f"- Probabilidade de phishing: {phishing_prob:.2%}")
            
            if phishing_prob > 0.75:  
                print("⚠ ATENÇÃO: Conteúdo classificado como potencial phishing")
            elif phishing_prob > 0.45:
                print("⚠ Suspeita: Conteúdo pode conter elementos suspeitos")
            else:
                print("✅ Conteúdo parece seguro")
    
    except Exception as e:
        print(f"\nErro na análise de conteúdo: {str(e)}")

def monitor_hotkeys():
    print("Aguardando atalho (X) para capturar e analisar e-mail...")
    while True:
        if keyboard.is_pressed('x'):
            print("\nCapturando área do e-mail...")
            text = capture_and_extract_text()
            text_full_screen = capture_and_extract_text_full_screen()
            print("\nTexto extraído:")
            print(text_full_screen[:500] + ("..." if len(text) > 500 else ""))  
            process_email(text,text_full_screen)
            time.sleep(1)

if __name__ == "__main__":
    monitor_hotkeys()