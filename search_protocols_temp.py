from pypdf import PdfReader
import sys

pdf_path = r'C:\Users\sidha\.dsh\attachments\v1\files\f2\f2349c69fd51ad8c369d788fa415d370552d0183473a403d7cd23be4812400b5\Columbia_River_ACP_updated_2021_Final.pdf'
try:
    reader = PdfReader(pdf_path)
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        if 'In-Situ Burn' in text or 'Bioremediation' in text or '1670.2' in text or '1670.3' in text:
            print(f'--- Page {i+1} ---')
            print(text)
except Exception as e:
    print(f'Error: {e}')
