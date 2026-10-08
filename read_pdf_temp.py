from pypdf import PdfReader
import sys

pdf_path = r'C:\Users\sidha\.dsh\attachments\v1\files\f2\f2349c69fd51ad8c369d788fa415d370552d0183473a403d7cd23be4812400b5\Columbia_River_ACP_updated_2021_Final.pdf'
try:
    reader = PdfReader(pdf_path)
    text = ''
    for i in range(min(15, len(reader.pages))):
        text += f'\n--- Page {i+1} ---\n'
        text += reader.pages[i].extract_text()
    print(text)
except Exception as e:
    print(f'Error: {e}')
