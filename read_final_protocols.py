from pypdf import PdfReader
import sys

pdf_path = r'C:\Users\sidha\.dsh\attachments\v1\files\f2\f2349c69fd51ad8c369d788fa415d370552d0183473a403d7cd23be4812400b5\Columbia_River_ACP_updated_2021_Final.pdf'
try:
    reader = PdfReader(pdf_path)
    text = ''
    # Targeting the actual operational sections: pages 50 to 65.
    for i in range(49, min(66, len(reader.pages))):
        page_text = reader.pages[i].extract_text()
        if page_text:
            text += f'\n--- Page {i+1} ---\n'
            text += page_text
    
    # Use sys.stdout.reconfigure to handle unicode characters on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
except Exception as e:
    print(f'Error: {e}')
