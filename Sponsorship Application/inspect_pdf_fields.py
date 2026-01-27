"""
Inspect PDF form fields to see what field names are available
"""

from pypdf import PdfReader
import sys

def inspect_pdf_fields(pdf_path):
    """Show all fillable fields in a PDF"""
    try:
        reader = PdfReader(pdf_path)
        
        print(f"\n{'='*60}")
        print(f"PDF: {pdf_path}")
        print(f"{'='*60}\n")
        
        if "/AcroForm" not in reader.trailer["/Root"]:
            print("❌ This PDF has no fillable form fields")
            return
        
        # Get form fields
        fields = reader.get_fields()
        
        if not fields:
            print("❌ No form fields found")
            return
        
        print(f"✓ Found {len(fields)} form fields:\n")
        
        for field_name, field_info in fields.items():
            field_type = field_info.get('/FT', 'Unknown')
            field_value = field_info.get('/V', '')
            
            print(f"Field Name: {field_name}")
            print(f"  Type: {field_type}")
            if field_value:
                print(f"  Current Value: {field_value}")
            print()
        
    except Exception as e:
        print(f"❌ Error reading PDF: {str(e)}")

if __name__ == "__main__":
    pdfs = [
        "IMM1344_blank.pdf",
        "IMM0008_blank.pdf", 
        "IMM5532_blank.pdf"
    ]
    
    for pdf in pdfs:
        inspect_pdf_fields(pdf)
