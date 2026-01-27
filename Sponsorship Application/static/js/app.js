let sessionId = 'session_' + Date.now();
let currentQuestion = null;

function startChat() {
    document.querySelector('.welcome-message').style.display = 'none';
    document.getElementById('inputContainer').style.display = 'block';
    
    fetch('/api/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId })
    })
    .then(response => response.json())
    .then(data => {
        addBotMessage(data.message);
        if (data.question) {
            currentQuestion = data.question;
            addBotMessage(data.question.text);
            updateProgress(data.progress, data.total);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addBotMessage('Sorry, something went wrong. Please refresh and try again.');
    });
}

function submitAnswer(event) {
    event.preventDefault();
    
    const input = document.getElementById('answerInput');
    const answer = input.value.trim();
    
    if (!answer) return;
    
    addUserMessage(answer);
    input.value = '';
    
    fetch('/api/answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            answer: answer,
            question_id: currentQuestion.id
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.completed) {
            addBotMessage(data.message);
            updateProgress(data.progress, data.total);
            document.getElementById('inputContainer').style.display = 'none';
            document.getElementById('completionActions').style.display = 'flex';
        } else {
            if (data.section_message) {
                addSectionHeader(data.section_message);
            }
            currentQuestion = data.question;
            addBotMessage(data.question.text);
            updateProgress(data.progress, data.total);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addBotMessage('Sorry, something went wrong. Please try again.');
    });
}

function addBotMessage(text) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="message-content">${text}</div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addUserMessage(text) {
    const chatContainer = document.getElementById('chatContainer');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="message-content">${text}</div>
    `;
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addSectionHeader(text) {
    const chatContainer = document.getElementById('chatContainer');
    const headerDiv = document.createElement('div');
    headerDiv.className = 'section-header';
    headerDiv.textContent = text;
    chatContainer.appendChild(headerDiv);
    scrollToBottom();
}

function updateProgress(current, total) {
    const percentage = (current / total) * 100;
    document.getElementById('progressFill').style.width = percentage + '%';
    document.getElementById('progressText').textContent = `Question ${current} of ${total}`;
}

function scrollToBottom() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function viewSummary() {
    fetch(`/api/summary?session_id=${sessionId}`)
    .then(response => response.json())
    .then(data => {
        displaySummary(data.responses);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error loading summary');
    });
}

function displaySummary(responses) {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';
    
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'summary-container';
    
    const sections = {
        'Sponsor Information': [
            'sponsor_full_name', 'sponsor_dob', 'sponsor_citizenship',
            'sponsor_address', 'sponsor_phone', 'sponsor_email'
        ],
        'Applicant Information': [
            'applicant_full_name', 'applicant_dob', 'applicant_citizenship',
            'applicant_passport', 'applicant_address', 'applicant_phone', 'applicant_email'
        ],
        'Relationship Information': [
            'marriage_date', 'marriage_location', 'first_met_date',
            'first_met_location', 'relationship_start', 'living_together'
        ]
    };
    
    for (const [sectionName, fields] of Object.entries(sections)) {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'summary-section';
        sectionDiv.innerHTML = `<h3>${sectionName}</h3>`;
        
        fields.forEach(field => {
            if (responses[field]) {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'summary-item';
                const label = field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                itemDiv.innerHTML = `
                    <div class="summary-label">${label}:</div>
                    <div class="summary-value">${responses[field]}</div>
                `;
                sectionDiv.appendChild(itemDiv);
            }
        });
        
        summaryDiv.appendChild(sectionDiv);
    }
    
    chatContainer.appendChild(summaryDiv);
    scrollToBottom();
}

function downloadData() {
    window.location.href = `/api/download?session_id=${sessionId}`;
}

function startOver() {
    sessionId = 'session_' + Date.now();
    currentQuestion = null;
    
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <h2>Welcome!</h2>
            <p>I'll guide you through collecting all the information needed for your IRCC spousal sponsorship application.</p>
            <p>This includes:</p>
            <ul>
                <li>Sponsor information (IMM 1344)</li>
                <li>Applicant information (IMM 0008)</li>
                <li>Relationship details (IMM 5532)</li>
            </ul>
            <button class="btn btn-primary" onclick="startChat()">Start Application</button>
        </div>
    `;
    
    document.getElementById('inputContainer').style.display = 'none';
    document.getElementById('completionActions').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = 'Ready to start';
}

// Allow Enter key to submit
document.addEventListener('DOMContentLoaded', function() {
    const input = document.getElementById('answerInput');
    if (input) {
        input.focus();
    }
});
