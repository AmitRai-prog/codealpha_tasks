from flask import Flask, request, jsonify, render_template_string
from deep_translator import GoogleTranslator

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Language Translator</title>

    <style>
        :root {
            color-scheme: dark;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #eef2ff;
            color: #111827;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            background: radial-gradient(circle at top, rgba(99, 102, 241, 0.18), transparent 35%),
                        linear-gradient(180deg, #eef2ff 0%, #f8fafc 100%);
            padding: 24px;
        }

        .container {
            width: min(100%, 1040px);
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 28px;
            padding: 32px;
            box-shadow: 0 28px 80px rgba(15, 23, 42, 0.12);
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: clamp(1.9rem, 2.2vw, 2.7rem);
            letter-spacing: -0.03em;
            line-height: 1.05;
        }

        .subtitle {
            color: #4b5563;
            font-size: 0.98rem;
            margin-top: 8px;
        }

        .section {
            display: grid;
            gap: 22px;
        }

        .languages {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 18px;
            align-items: end;
        }

        .select-group {
            display: grid;
            gap: 10px;
        }

        label {
            font-size: 0.92rem;
            font-weight: 700;
            color: #1f2937;
        }

        select {
            appearance: none;
            width: 100%;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #d1d5db;
            background: #ffffff;
            font-size: 0.96rem;
            color: #111827;
            transition: border-color 0.2s ease;
        }

        select:focus {
            border-color: #6366f1;
            outline: none;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
        }

        .swap-button {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            cursor: pointer;
            box-shadow: 0 16px 30px rgba(99, 102, 241, 0.18);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .swap-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(99, 102, 241, 0.22);
        }

        .textareas {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        textarea {
            width: 100%;
            min-height: 260px;
            padding: 18px;
            border-radius: 22px;
            border: 1px solid #d1d5db;
            resize: vertical;
            font-size: 1rem;
            line-height: 1.6;
            color: #111827;
            background: #f8fafc;
            transition: border-color 0.2s ease, background 0.2s ease;
        }

        textarea:focus {
            outline: none;
            border-color: #6366f1;
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08);
        }

        textarea[readonly] {
            background: #f3f4f6;
            color: #374151;
            cursor: text;
        }

        .toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
            justify-content: center;
        }

        .toolbar button {
            flex: 1 1 140px;
            min-width: 140px;
            border: none;
            border-radius: 14px;
            padding: 14px 18px;
            font-size: 0.98rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        .toolbar button.primary {
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
        }

        .toolbar button.secondary {
            background: #eef2ff;
            color: #3730a3;
        }

        .toolbar button:disabled {
            opacity: 0.55;
            cursor: not-allowed;
            transform: none;
        }

        .toolbar button:not(:disabled):hover {
            transform: translateY(-1px);
        }

        .status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            min-height: 1.5rem;
            margin-top: 18px;
            color: #475569;
            font-size: 0.95rem;
        }

        .status span {
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .status .loader {
            width: 16px;
            height: 16px;
            border: 3px solid rgba(99, 102, 241, 0.2);
            border-top-color: #6366f1;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        .footer {
            margin-top: 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #6b7280;
            font-size: 0.88rem;
        }

        .footer strong {
            color: #111827;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 860px) {
            .languages {
                grid-template-columns: 1fr 1fr;
            }

            .swap-button {
                width: 100%;
                border-radius: 16px;
            }
        }

        @media (max-width: 720px) {
            .textareas {
                grid-template-columns: 1fr;
            }

            .toolbar {
                justify-content: stretch;
            }
        }
    </style>
</head>

<body>

<div class="container">
    <div class="header">
        <div>
            <h1> Language Translator</h1>
            <p class="subtitle">Fast, clean and accessible translation with copy, speech and quick swap controls.</p>
        </div>
    </div>

    <div class="section">
        <div class="languages">
            <div class="select-group">
                <label for="source">Source Language</label>
                <select id="source">
                    <option value="en">English</option>
                    <option value="hi">Hindi</option>
                    <option value="gu">Gujarati</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="es">Spanish</option>
                </select>
            </div>

            <button class="swap-button" type="button" onclick="swapLanguages()" aria-label="Swap languages">⇆</button>

            <div class="select-group">
                <label for="target">Target Language</label>
                <select id="target">
                    <option value="hi">Hindi</option>
                    <option value="en">English</option>
                    <option value="gu">Gujarati</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="es">Spanish</option>
                </select>
            </div>
        </div>

        <div class="textareas">
            <textarea id="inputText" placeholder="Type or paste text here..." aria-label="Source text"></textarea>
            <textarea id="outputText" placeholder="Translation appears here..." readonly aria-label="Translated text"></textarea>
        </div>

        <div class="toolbar">
            <button class="primary" id="translateButton" type="button" onclick="translateText()">Translate</button>
            <button class="secondary" type="button" onclick="clearText()">Clear</button>
            <button class="secondary" type="button" onclick="copyText()">Copy</button>
            <button class="secondary" type="button" onclick="speakText()">🔊 Speak</button>
        </div>

        <div id="status" class="status"></div>
    </div>

    
</div>

<script>
    
async function translateText() {

    let text = document.getElementById("inputText").value;
    let source = document.getElementById("source").value;
    let target = document.getElementById("target").value;

    if(text.trim() === ""){
        alert("Enter text first");
        return;
    }

    const response = await fetch("/translate",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            text:text,
            source:source,
            target:target
        })
    });

    const data = await response.json();

    document.getElementById("outputText").value =
        data.translated;
}

function copyText(){

    let text =
        document.getElementById("outputText").value;

    navigator.clipboard.writeText(text);

    alert("Copied!");
}

function speakText(){

    let text =
        document.getElementById("outputText").value;

    let speech =
        new SpeechSynthesisUtterance(text);

    speech.lang =
        document.getElementById("target").value;

    speechSynthesis.speak(speech);
}

function clearText(){

    document.getElementById("inputText").value = "";
    document.getElementById("outputText").value = "";
}

function swapLanguages(){

    let source =
        document.getElementById("source");

    let target =
        document.getElementById("target");

    let temp = source.value;

    source.value = target.value;
    target.value = temp;
}


</script>

</body>
</html>
"""


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()

    translated = GoogleTranslator(
        source=data["source"],
        target=data["target"]
    ).translate(data["text"])

    return jsonify({"translated": translated})


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(debug=True)

