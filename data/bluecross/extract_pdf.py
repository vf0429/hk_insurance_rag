from pypdf import PdfReader

reader = PdfReader('/Users/vfzzz/Desktop/Petwell insurance RAG/data/bluecross/LovePet_Insurance_TnC.pdf')
for i, page in enumerate(reader.pages):
    print(f'=== Page {i+1} ===')
    print(page.extract_text())
    print()
