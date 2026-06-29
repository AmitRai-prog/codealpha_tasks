
from flask import Flask, request, jsonify, render_template_string
import re
import string

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

# FAQ Dataset
faq_data = {
    "What is AI?":
        "Artificial Intelligence is the simulation of human intelligence by machines.",

    "What is Machine Learning?":
        "Machine Learning is a subset of AI that allows systems to learn from data.",

    "What is Deep Learning?":
        "Deep Learning uses neural networks with multiple layers to learn patterns.",

    "What is Python?":
        "Python is a popular programming language used in AI, web development, and automation.",

    "What is NLP?":
        "Natural Language Processing helps computers understand human language.",

    "What is Flask?":
        "Flask is a lightweight Python web framework."
}

questions = list(faq_data.keys())


ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves"
}

def preprocess(text):
    text = text.lower()
    cleaned = "".join(char if char not in string.punctuation else " " for char in text)
    words = re.findall(r"\b[a-z0-9]+\b", cleaned)
    filtered = [word for word in words if word not in ENGLISH_STOPWORDS]
    return " ".join(filtered)


processed_questions = [
    preprocess(question)
    for question in questions
]

if SKLEARN_AVAILABLE:
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(
        processed_questions
    )
else:
    question_keywords = [set(question.split()) for question in processed_questions]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Base</title>
    <style>
        :root {
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #111827;
            background: #f3f4f6;
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
            background: linear-gradient(180deg, #e2e8f0 0%, #f8fafc 100%);
        }
        .chat-container {
            width: min(100%, 840px);
            border-radius: 24px;
            overflow: hidden;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
        }
        .header {
            padding: 32px 36px 24px;
            border-bottom: 1px solid #e5e7eb;
        }
        .header h1 {
            margin: 0;
            font-size: clamp(2rem, 2.3vw, 2.5rem);
            line-height: 1.05;
            letter-spacing: -0.04em;
        }
        .header p {
            margin: 14px 0 0;
            color: #4b5563;
            font-size: 1rem;
            line-height: 1.75;
            max-width: 720px;
        }
        .details {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 22px;
        }
        .detail {
            padding: 12px 16px;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            color: #374151;
            font-size: 0.95rem;
        }
        .chat-box {
            min-height: 460px;
            max-height: 62vh;
            overflow-y: auto;
            padding: 24px 28px;
            background: #ffffff;
        }
        .message-row {
            display: flex;
            margin-bottom: 18px;
            gap: 12px;
        }
        .message-row.user {
            justify-content: flex-end;
        }
        .message-row.bot {
            justify-content: flex-start;
        }
        .message {
            max-width: 78%;
            padding: 18px 20px;
            border-radius: 20px;
            line-height: 1.7;
            letter-spacing: 0.01em;
            word-break: break-word;
            border: 1px solid transparent;
        }
        .message.user {
            background: #1d4ed8;
            color: white;
            border-bottom-right-radius: 8px;
            box-shadow: 0 16px 32px rgba(29, 78, 216, 0.18);
        }
        .message.bot {
            background: #f8fafc;
            color: #111827;
            border-bottom-left-radius: 8px;
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.08);
        }
        .status {
            margin: 0;
            color: #6b7280;
            font-size: 0.95rem;
            text-align: center;
        }
        .input-area {
            display: grid;
            gap: 14px;
            padding: 22px 28px 28px;
            background: #f8fafc;
            border-top: 1px solid #e5e7eb;
        }
        .input-row {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 14px;
        }
        .input-row input {
            width: 100%;
            padding: 16px 18px;
            border-radius: 16px;
            border: 1px solid #d1d5db;
            background: white;
            color: #111827;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .input-row input:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.12);
        }
        .input-row button,
        .secondary-button {
            padding: 16px 22px;
            border-radius: 16px;
            border: none;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.15s ease, box-shadow 0.15s ease;
        }
        .input-row button {
            background: #2563eb;
            color: white;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.18);
        }
        .input-row button:hover,
        .secondary-button:hover {
            transform: translateY(-1px);
            opacity: 0.95;
        }
        .secondary-button {
            align-self: flex-start;
            background: transparent;
            color: #374151;
            border: 1px solid #d1d5db;
            width: fit-content;
            justify-self: start;
        }
        .hint {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
            color: #6b7280;
            font-size: 0.93rem;
        }
        .hint strong {
            color: #111827;
        }
        @media (max-width: 700px) {
            .chat-box {
                padding: 20px;
            }
            .input-row {
                grid-template-columns: 1fr;
            }
            .input-row button,
            .secondary-button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
<div class="chat-container">
    <div class="header">
        <h1>Knowledge Base</h1>
        <p>Search the FAQ library for fast, accurate answers. This interface is designed to look like a polished help center rather than an AI demo.</p>
        <div class="details">
            <div class="detail">FAQ search</div>
            
        </div>
    </div>
    <div class="chat-box" id="chatBox">
        <p class="status">Enter your question to search the FAQ library.</p>
    </div>
    <div class="input-area">
        <div class="input-row">
            <input type="text" id="userInput" placeholder="Type a question or topic..." autocomplete="off" />
            <button type="button" onclick="sendMessage()">Search</button>
        </div>
        <div class="hint">
            <span><strong>Enter</strong> to search • <strong>Clear</strong> resets.</span>
            <button type="button" class="secondary-button" onclick="clearChat()">Clear</button>
        </div>
    </div>
</div>
<script>
    const chatBox = document.getElementById("chatBox");
    const userInput = document.getElementById("userInput");

    userInput.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    function appendMessage(role, text) {
        const row = document.createElement("div");
        row.className = `message-row ${role}`;
        row.innerHTML = `<div class="message ${role}">${text}</div>`;
        if (chatBox.querySelector(".status")) {
            chatBox.innerHTML = "";
        }
        chatBox.appendChild(row);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function sendMessage() {
        const question = userInput.value.trim();
        if (!question) return;

        appendMessage("user", question);
        userInput.value = "";
        userInput.focus();

        appendMessage("bot", "Searching...");
        const lastBot = chatBox.querySelector(".message-row.bot:last-child .message");

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            });
            const data = await response.json();
            if (lastBot) {
                lastBot.textContent = data.answer;
            }
        } catch (error) {
            if (lastBot) {
                lastBot.textContent = "Unable to fetch results. Please try again.";
            }
            console.error(error);
        }
    }

    function clearChat() {
        chatBox.innerHTML = '<p class="status">Enter your question to search the FAQ library.</p>';
        userInput.value = "";
        userInput.focus();
    }
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/chat", methods=["POST"])
def chat():

    user_question = request.json["question"]

    processed_input = preprocess(user_question)

    if SKLEARN_AVAILABLE:
        user_vector = vectorizer.transform(
            [processed_input]
        )

        similarities = cosine_similarity(
            user_vector,
            question_vectors
        )

        best_match_index = similarities.argmax()
        best_score = similarities[0][best_match_index]
    else:
        user_terms = set(processed_input.split())
        best_score = 0.0
        best_match_index = 0

        for idx, question_terms in enumerate(question_keywords):
            if not question_terms:
                continue
            score = len(user_terms & question_terms) / len(question_terms)
            if score > best_score:
                best_score = score
                best_match_index = idx

    if best_score < 0.2:
        answer = (
            "Sorry, I couldn't find a matching FAQ."
        )
    else:
        matched_question = questions[
            best_match_index
        ]

        answer = faq_data[
            matched_question
        ]

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

