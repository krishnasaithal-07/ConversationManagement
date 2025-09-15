# Groq API Conversation Management & Classification

Implementation of conversation management with summarization and JSON schema-based information extraction using Groq API.

## Features Implemented

### Task 1: Conversation Management
- ✅ Conversation history management
- ✅ Truncation by turns and character length
- ✅ Periodic summarization after K runs
- ✅ Comprehensive demonstration with sample data

### Task 2: Information Extraction  
- ✅ JSON schema for 5 fields (name, email, phone, location, age)
- ✅ OpenAI function calling with Groq API
- ✅ Validation against schema patterns
- ✅ Processing of multiple chat samples

## How to Run
1. Open the notebook in Google Colab
2. Replace API key with your Groq API key
3. Run all cells to see demonstrations
4. Check outputs for both tasks

## Dependencies
- openai (for Groq API compatibility)
- requests
- Standard Python libraries (json, re, datetime, typing)

## API Configuration
The system uses the Groq API with the following configuration:
- Base URL: https://api.groq.com/openai/v1
- Default Model: llama-3.1-8b-instant
- Temperature: 0.1-0.3 for consistent results
- Response Format: JSON object for structured data extraction


## Example Output
- The system provides detailed output including:
- Chat transcripts with speaker identification
- Conversation metrics (message count, characters, words)
- Extraction results with quality assessments
- Performance statistics for information extraction
