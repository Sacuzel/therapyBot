import pymupdf

"""
This function will convert a pdf book (with selectable text)
to a txt file while preserving the structure of the book.
"""
pdf_file = '/home/sacuzel/Source material/Books/Theory and practice of counseling and psychotherapy (Gerald Corey, California State University etc.).pdf'
txt_file = '/home/sacuzel/telegram_t_bot/sourceMaterial/Theory and practice of counseling and psychotherapy.txt'
def pdf_to_text(pdf_path, txt_path):
    # First open the PDF
    pdf_document = pymupdf.open(pdf_path)
    # Create a text file to store the extracted text
    with open(txt_path, "w", encoding="utf-8") as text_file:

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text = page.get_text()
            text_file.write(text)

    # Close the PDF
    pdf_document.close()

def main():
    pdf_to_text(pdf_file, txt_file)

if __name__=="__main__":
    main()