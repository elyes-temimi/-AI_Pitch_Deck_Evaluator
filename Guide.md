# AI Pitch Deck Evaluator

An AI-powered pitch deck evaluation system developed as part of the V3 Factory technical assignment.

The project analyzes startup pitch decks using a multi-agent architecture and Retrieval-Augmented Generation (RAG) to simulate investor-style evaluation.

## Features

* Supports PDF, PPTX, TXT, and JSON pitch decks
* Automatic extraction and cleaning of slide content
* Multi-agent evaluation architecture:

  * Clarity Agent
  * Narrative Agent
  * Problem-Solution Fit Agent
  * Investor Calibration Agent (RAG)
  * Investor Committee Agent
* Benchmark-based comparison using retrieval
* Streamlit web interface
* Local LLM execution using Ollama
* No external APIs required

---

## Architecture

Input Deck

↓

Extractor

↓

Cleaning Layer

↓

Section Detection

↓

RAG Retrieval

↓

Clarity Agent

Narrative Agent

Problem-Solution Agent

Calibration Agent

↓

Investor Committee

↓

Final Report

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI-Pitch-Deck-Evaluator
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

This project uses a local LLM through Ollama.

Download and install Ollama:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

---

## Download the Model

The project was developed and tested using:

```bash
ollama pull llama3.1
```

You may also use another compatible model if desired.

---

## Run Ollama

Start Ollama:

```bash
ollama serve
```

Default endpoint:

```text
http://localhost:11434
```

---

## Run the Application

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## Supported Input Formats

### JSON

```json
[
  {
    "slide": 1,
    "title": "Problem",
    "content": "Customer acquisition is expensive."
  }
]
```

### PowerPoint

```text
.pptx
```

### PDF

```text
.pdf
```

### Text

```text
.txt
```

Slides separated by:

```text
---
```

---

## Design Decisions

Several iterations were performed during development:

### Version 1

Single LLM evaluation.

### Version 2

Multi-agent architecture for clearer reasoning.

### Version 3

RAG-based benchmark calibration.

### Version 4

Improved extraction and cleaning pipeline:

* Footer removal
* Template detection
* Promotional slide filtering
* Placeholder text detection

These improvements significantly increased evaluation consistency on real-world pitch decks.

---

## Notes

* The system runs entirely locally.
* No OpenAI API key is required.
* Evaluation quality depends on the quality of extracted slide text.
* OCR for image-only PDFs is not currently implemented.

---

## Tested With

* Python 3.11+
* Ollama
* Llama 3.1
* Windows 11

---

## Author

Elyes Temimi

Computer Science Student — ISI Ariana
