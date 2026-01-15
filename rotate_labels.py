from PyPDF2 import PdfReader, PdfWriter

input_pdf = r"C:\Users\jerem\Desktop\scraper\label_output\labels.pdf"
output_pdf = r"C:\Users\jerem\Desktop\scraper\label_output\labels_rotated.pdf"

reader = PdfReader(input_pdf)
writer = PdfWriter()

for page in reader.pages:
    page.rotate(90)  # Rechtsom kwartslag (270 voor linksom)
    writer.add_page(page)

with open(output_pdf, "wb") as f:
    writer.write(f)

print("Nieuwe PDF opgeslagen als:", output_pdf)
