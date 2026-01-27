import json
from datetime import datetime
from pathlib import Path

class IRCCSponsorshipChat:
    def __init__(self):
        self.responses = {}
        self.questions = self.load_questions()
    
    def load_questions(self):
        """Define questions for IRCC spousal sponsorship forms"""
        return {
            # Sponsor Information (IMM 1344)
            "sponsor": [
                ("sponsor_full_name", "What is the sponsor's full legal name?"),
                ("sponsor_dob", "What is the sponsor's date of birth? (YYYY-MM-DD)"),
                ("sponsor_citizenship", "What is the sponsor's citizenship status? (Canadian Citizen/Permanent Resident)"),
                ("sponsor_address", "What is the sponsor's current residential address?"),
                ("sponsor_phone", "What is the sponsor's phone number?"),
                ("sponsor_email", "What is the sponsor's email address?"),
            ],
            # Principal Applicant Information (IMM 0008)
            "applicant": [
                ("applicant_full_name", "What is the applicant's (spouse) full legal name?"),
                ("applicant_dob", "What is the applicant's date of birth? (YYYY-MM-DD)"),
                ("applicant_citizenship", "What is the applicant's country of citizenship?"),
                ("applicant_passport", "What is the applicant's passport number?"),
                ("applicant_address", "What is the applicant's current address?"),
                ("applicant_phone", "What is the applicant's phone number?"),
                ("applicant_email", "What is the applicant's email address?"),
            ],
            # Relationship Information (IMM 5532)
            "relationship": [
                ("marriage_date", "When did you get married? (YYYY-MM-DD)"),
                ("marriage_location", "Where did you get married? (City, Country)"),
                ("first_met_date", "When did you first meet? (YYYY-MM-DD)"),
                ("first_met_location", "Where did you first meet?"),
                ("relationship_start", "When did your relationship begin? (YYYY-MM-DD)"),
                ("living_together", "Are you currently living together? (Yes/No)"),
            ]
        }
    
    def chat(self):
        """Main chat interface"""
        print("\n" + "="*60)
        print("IRCC SPOUSAL SPONSORSHIP APPLICATION ASSISTANT")
        print("="*60)
        print("\nI'll help you fill out your spousal sponsorship forms.")
        print("Please answer the following questions accurately.\n")
        
        # Ask questions by section
        for section, questions in self.questions.items():
            print(f"\n--- {section.upper()} INFORMATION ---\n")
            for key, question in questions:
                answer = input(f"{question}\n> ").strip()
                self.responses[key] = answer
        
        print("\n" + "="*60)
        print("Thank you! I've collected all the information.")
        print("="*60)
    
    def save_responses(self, filename="sponsorship_data.json"):
        """Save responses to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "responses": self.responses
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Data saved to {filename}")
    
    def generate_form_summary(self):
        """Generate a summary of collected information"""
        print("\n" + "="*60)
        print("FORM DATA SUMMARY")
        print("="*60)
        
        for key, value in self.responses.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"{formatted_key}: {value}")
        
        print("\n" + "="*60)
    
    def fill_forms(self):
        """Placeholder for PDF form filling functionality"""
        print("\n📄 Form filling functionality:")
        print("   - IMM 1344 (Application to Sponsor)")
        print("   - IMM 0008 (Generic Application Form)")
        print("   - IMM 5532 (Relationship Information)")
        print("\nNote: To fill actual PDF forms, install: pip install PyPDF2 or pdfrw")
        print("Place your blank PDF forms in the same directory.")


def main():
    chat = IRCCSponsorshipChat()
    
    # Run the chat interface
    chat.chat()
    
    # Save responses
    chat.save_responses()
    
    # Show summary
    chat.generate_form_summary()
    
    # Form filling info
    chat.fill_forms()
    
    print("\n✓ Process complete!")


if __name__ == "__main__":
    main()
