from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import json
from pathlib import Path

app = Flask(__name__)

# Store conversation state per session (in production, use proper session management)
conversations = {}

QUESTIONS = {
    "sponsor": [
        {"id": "sponsor_full_name", "text": "What is the sponsor's full legal name?", "type": "text"},
        {"id": "sponsor_dob", "text": "What is the sponsor's date of birth?", "type": "date"},
        {"id": "sponsor_citizenship", "text": "What is the sponsor's citizenship status?", "type": "select", "options": ["Canadian Citizen", "Permanent Resident"]},
        {"id": "sponsor_address", "text": "What is the sponsor's current residential address?", "type": "text"},
        {"id": "sponsor_phone", "text": "What is the sponsor's phone number?", "type": "tel"},
        {"id": "sponsor_email", "text": "What is the sponsor's email address?", "type": "email"},
    ],
    "applicant": [
        {"id": "applicant_full_name", "text": "What is the applicant's (spouse) full legal name?", "type": "text"},
        {"id": "applicant_dob", "text": "What is the applicant's date of birth?", "type": "date"},
        {"id": "applicant_citizenship", "text": "What is the applicant's country of citizenship?", "type": "text"},
        {"id": "applicant_passport", "text": "What is the applicant's passport number?", "type": "text"},
        {"id": "applicant_address", "text": "What is the applicant's current address?", "type": "text"},
        {"id": "applicant_phone", "text": "What is the applicant's phone number?", "type": "tel"},
        {"id": "applicant_email", "text": "What is the applicant's email address?", "type": "email"},
    ],
    "relationship": [
        {"id": "marriage_date", "text": "When did you get married?", "type": "date"},
        {"id": "marriage_location", "text": "Where did you get married? (City, Country)", "type": "text"},
        {"id": "first_met_date", "text": "When did you first meet?", "type": "date"},
        {"id": "first_met_location", "text": "Where did you first meet?", "type": "text"},
        {"id": "relationship_start", "text": "When did your relationship begin?", "type": "date"},
        {"id": "living_together", "text": "Are you currently living together?", "type": "select", "options": ["Yes", "No"]},
    ]
}

def get_all_questions():
    """Flatten all questions into a single list"""
    all_questions = []
    for section, questions in QUESTIONS.items():
        all_questions.extend(questions)
    return all_questions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_conversation():
    """Initialize a new conversation"""
    session_id = request.json.get('session_id', 'default')
    conversations[session_id] = {
        'current_question': 0,
        'responses': {},
        'started_at': datetime.now().isoformat()
    }
    
    questions = get_all_questions()
    first_question = questions[0]
    
    return jsonify({
        'message': "Hi! I'm here to help you with your IRCC spousal sponsorship application. Let's start with the sponsor information.",
        'question': first_question,
        'progress': 0,
        'total': len(questions)
    })

@app.route('/api/answer', methods=['POST'])
def process_answer():
    """Process user's answer and return next question"""
    data = request.json
    session_id = data.get('session_id', 'default')
    answer = data.get('answer')
    question_id = data.get('question_id')
    
    if session_id not in conversations:
        return jsonify({'error': 'Session not found'}), 400
    
    conv = conversations[session_id]
    conv['responses'][question_id] = answer
    conv['current_question'] += 1
    
    questions = get_all_questions()
    
    if conv['current_question'] >= len(questions):
        # All questions answered
        save_responses(session_id, conv['responses'])
        return jsonify({
            'completed': True,
            'message': "Thank you! I've collected all the information. You can now download your data or view the summary.",
            'progress': len(questions),
            'total': len(questions)
        })
    
    next_question = questions[conv['current_question']]
    
    # Add section headers
    section_message = ""
    if conv['current_question'] == 7:
        section_message = "Great! Now let's collect the applicant's information."
    elif conv['current_question'] == 14:
        section_message = "Perfect! Finally, let's talk about your relationship."
    
    return jsonify({
        'question': next_question,
        'section_message': section_message,
        'progress': conv['current_question'],
        'total': len(questions)
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary of collected data"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in conversations:
        return jsonify({'error': 'Session not found'}), 400
    
    conv = conversations[session_id]
    return jsonify({
        'responses': conv['responses'],
        'timestamp': conv['started_at']
    })

@app.route('/api/download', methods=['GET'])
def download_data():
    """Download collected data as JSON"""
    session_id = request.args.get('session_id', 'default')
    
    if session_id not in conversations:
        return jsonify({'error': 'Session not found'}), 400
    
    filename = f'sponsorship_data_{session_id}.json'
    return send_file(
        filename,
        mimetype='application/json',
        as_attachment=True,
        download_name='sponsorship_data.json'
    )

def save_responses(session_id, responses):
    """Save responses to JSON file"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'responses': responses
    }
    filename = f'sponsorship_data_{session_id}.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
