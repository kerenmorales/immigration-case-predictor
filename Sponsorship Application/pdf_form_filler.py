"""
PDF Form Filler for IRCC Spousal Sponsorship Forms
Requires: pip install pypdf
"""

try:
    from pypdf import PdfReader, PdfWriter
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pypdf not installed. Install with: pip install pypdf")

import json
from pathlib import Path


class IRCCFormFiller:
    def __init__(self, data_file="sponsorship_data.json"):
        self.data = self.load_data(data_file)
        self.form_mappings = self.get_form_mappings()
    
    def load_data(self, filename):
        """Load collected data from JSON"""
        if not Path(filename).exists():
            raise FileNotFoundError(f"{filename} not found. Run main.py first.")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['responses']
    
    def get_form_mappings(self):
        """Map collected data to PDF form field names"""
        return {
            "IMM1344": {
                # Sponsor section
                "sponsor_name": "sponsor_full_name",
                "sponsor_dob": "sponsor_dob",
                "sponsor_citizenship": "sponsor_citizenship",
                "sponsor_address": "sponsor_address",
                "sponsor_phone": "sponsor_phone",
                "sponsor_email": "sponsor_email",
            },
            "IMM0008": {
                # Principal applicant section
                "applicant_name": "applicant_full_name",
                "applicant_dob": "applicant_dob",
                "applicant_citizenship": "applicant_citizenship",
                "applicant_passport": "applicant_passport",
                "applicant_address": "applicant_address",
                "applicant_phone": "applicant_phone",
                "applicant_email": "applicant_email",
            },
            "IMM5532": {
                # Relationship section
                "marriage_date": "marriage_date",
                "marriage_location": "marriage_location",
                "first_met_date": "first_met_date",
                "first_met_location": "first_met_location",
                "relationship_start": "relationship_start",
                "living_together": "living_together",
            }
        }
    
    def fill_pdf_form(self, input_pdf, output_pdf, form_type):
        """Fill a PDF form with collected data"""
        if not PDF_SUPPORT:
            print("PDF support not available. Install pypdf first.")
            return False
        
        if not Path(input_pdf).exists():
            print(f"Warning: {input_pdf} not found. Skipping.")
            return False
        
        try:
            reader = PdfReader(input_pdf)
            writer = PdfWriter()
            
            # Get form fields
            if "/AcroForm" in reader.trailer["/Root"]:
                writer.append(reader)
                
                # Map data to form fields
                mapping = self.form_mappings.get(form_type, {})
                form_data = {}
                
                for pdf_field, data_key in mapping.items():
                    if data_key in self.data:
                        form_data[pdf_field] = self.data[data_key]
                
                # Fill the form
                writer.update_page_form_field_values(writer.pages[0], form_data)
                
                # Write output
                with open(output_pdf, 'wb') as output_file:
                    writer.write(output_file)
                
                print(f"✓ Filled: {output_pdf}")
                return True
            else:
                print(f"Warning: {input_pdf} has no fillable fields")
                return False
                
        except Exception as e:
            print(f"Error filling {input_pdf}: {str(e)}")
            return False
    
    def fill_all_forms(self):
        """Fill all IRCC forms"""
        forms = [
            ("IMM1344_blank.pdf", "IMM1344_filled.pdf", "IMM1344"),
            ("IMM0008_blank.pdf", "IMM0008_filled.pdf", "IMM0008"),
            ("IMM5532_blank.pdf", "IMM5532_filled.pdf", "IMM5532"),
        ]
        
        print("\n" + "="*60)
        print("FILLING PDF FORMS")
        print("="*60 + "\n")
        
        success_count = 0
        for input_pdf, output_pdf, form_type in forms:
            if self.fill_pdf_form(input_pdf, output_pdf, form_type):
                success_count += 1
        
        print(f"\n✓ Successfully filled {success_count}/{len(forms)} forms")
        print("\nNote: Place blank PDF forms in this directory with names:")
        print("  - IMM1344_blank.pdf")
        print("  - IMM0008_blank.pdf")
        print("  - IMM5532_blank.pdf")


if __name__ == "__main__":
    filler = IRCCFormFiller()
    filler.fill_all_forms()
