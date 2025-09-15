import json
import requests
from openai import OpenAI
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Groq API setup - using OpenAI SDK for compatibility
my_groq_key = "your_groq_api_key"

groq_client = OpenAI(
    api_key=my_groq_key,
    base_url="https://api.groq.com/openai/v1",
)

class ChatHistoryHandler:
    """Handles conversation flows with smart memory management"""
    
    def __init__(self, ai_model="llama-3.1-8b-instant"):
        self.ai_model = ai_model
        self.chat_log = []
        self.message_count = 0
        self.last_summary = ""
        
    def record_message(self, speaker: str, text: str):
        """Records a new message in the chat"""
        new_entry = {
            "role": speaker,
            "content": text,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_log.append(new_entry)
        self.message_count += 1
        
    def keep_recent_turns(self, how_many_turns: int) -> List[Dict]:
        """Keeps only the most recent conversation turns"""
        messages_to_keep = how_many_turns * 2  # user + assistant = 1 turn
        
        if len(self.chat_log) <= messages_to_keep:
            return self.chat_log
            
        return self.chat_log[-messages_to_keep:]
        
    def fit_to_size(self, character_limit: int) -> List[Dict]:
        """Trims conversation to fit within character budget"""
        selected_messages = []
        chars_used = 0
        
        # Work backwards from newest messages
        for msg in reversed(self.chat_log):
            msg_size = len(msg["content"])
            
            if chars_used + msg_size > character_limit:
                break
                
            selected_messages.insert(0, msg)
            chars_used += msg_size
                
        return selected_messages
        
    def create_summary(self, messages_to_summarize: List[Dict]) -> str:
        """Creates a concise summary of the conversation"""
        if not messages_to_summarize:
            return "Nothing to summarize yet."
            
        # Build conversation text
        full_chat = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages_to_summarize
        ])
            
        try:
            ai_response = groq_client.chat.completions.create(
                model=self.ai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize conversations briefly. Focus on key points and context in 2-3 sentences max."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this chat:\n\n{full_chat}"
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            summary_text = ai_response.choices[0].message.content
            return summary_text
            
        except Exception as error:
            return f"Couldn't create summary: {str(error)}"
            
    def check_and_compress(self, after_n_messages: int):
        """Compresses conversation history after reaching threshold"""
        if self.message_count < after_n_messages:
            return
            
        print(f"\n🔄 Compressing conversation (reached {after_n_messages} messages)...")
        
        # Generate summary of everything so far
        self.last_summary = self.create_summary(self.chat_log)
        
        # Start fresh with just the summary
        self.chat_log = [{
            "role": "system",
            "content": f"Context from earlier: {self.last_summary}",
            "timestamp": datetime.now().isoformat()
        }]
        
        self.message_count = 0
        print(f"✅ Compressed! Summary: {self.last_summary}\n")
            
    def get_chat_metrics(self) -> Dict:
        """Calculate conversation statistics"""
        char_count = sum(len(m["content"]) for m in self.chat_log)
        word_count = sum(len(m["content"].split()) for m in self.chat_log)
        
        metrics = {
            "total_messages": len(self.chat_log),
            "total_characters": char_count,
            "total_words": word_count,
            "turns": len(self.chat_log) // 2
        }
        
        return metrics
        
    def show_chat(self, specific_messages: List[Dict] = None):
        """Displays the conversation nicely"""
        messages_to_show = specific_messages or self.chat_log
        
        print("📋 CHAT TRANSCRIPT:")
        print("=" * 50)
        
        for idx, msg in enumerate(messages_to_show, 1):
            icon = "🤖" if msg["role"] == "assistant" else "👤"
            if msg["role"] == "system":
                icon = "⚙️"
                
            speaker = msg['role'].upper()
            print(f"{idx}. {icon} {speaker}: {msg['content']}")
            
        print("=" * 50)

class DataMiner:
    """Extracts structured info from messy chat conversations"""
    
    def __init__(self, ai_model="llama-3.1-8b-instant"):
        self.ai_model = ai_model
        
    def mine_conversation(self, raw_chat: str) -> Dict[str, Any]:
        """Pulls structured data from unstructured chat using JSON response format"""
        try:
            ai_response = groq_client.chat.completions.create(
                model=self.ai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Extract user details from conversations and return ONLY a valid JSON object with the following structure:
                        {
                            "name": "string or empty if not mentioned",
                            "email": "string or empty if not mentioned", 
                            "phone": "string or empty if not mentioned",
                            "location": "string or empty if not mentioned",
                            "age": "integer or null if not mentioned"
                        }
                        Return ONLY the JSON object, no other text."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract user information from this conversation:\n\n{raw_chat}"
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            json_response = ai_response.choices[0].message.content
            found_info = json.loads(json_response)
            
            # Convert empty strings to None for age field
            if "age" in found_info and (found_info["age"] == "" or found_info["age"] is None):
                found_info["age"] = None
            elif "age" in found_info and isinstance(found_info["age"], str):
                # Try to convert string age to integer
                try:
                    found_info["age"] = int(found_info["age"])
                except (ValueError, TypeError):
                    found_info["age"] = None
                    
            return found_info
            
        except Exception as error:
            print(f"❌ Extraction failed: {str(error)}")
            return {}
            
    def check_quality(self, extracted_info: Dict) -> Dict[str, Any]:
        """Checks if extracted data looks reasonable"""
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "extracted_fields": [k for k, v in extracted_info.items() if v not in [None, ""]]
        }
        
        # Email check
        if "email" in extracted_info and extracted_info["email"]:
            if "@" not in extracted_info["email"] or "." not in extracted_info["email"]:
                report["warnings"].append("Email looks suspicious")
                
        # Age sanity check
        if "age" in extracted_info and extracted_info["age"]:
            age_val = extracted_info["age"]
            if not isinstance(age_val, int) or age_val < 0 or age_val > 150:
                report["warnings"].append("Age seems off")
                
        # Phone validation
        if "phone" in extracted_info and extracted_info["phone"]:
            digits_only = re.sub(r'[^\d]', '', extracted_info["phone"])
            if len(digits_only) < 7 or len(digits_only) > 15:
                report["warnings"].append("Phone number length unusual")
                
        return report

def run_conversation_demo():
    """Shows off the conversation management features"""
    print("🚀 TASK 1: SMART CONVERSATION MANAGEMENT")
    print("=" * 60)
    
    chat_handler = ChatHistoryHandler()
    
    # Simulate a customer service chat
    conversation_flow = [
        ("user", "Hi there! I'm looking for information about your product pricing."),
        ("assistant", "Hello! I'd be happy to help you with pricing information. What specific product are you interested in?"),
        ("user", "I'm interested in the premium subscription plan. What does it include?"),
        ("assistant", "The premium plan includes unlimited access, priority support, advanced features, and costs $29/month."),
        ("user", "That sounds good. Do you offer any discounts for annual subscriptions?"),
        ("assistant", "Yes! We offer a 20% discount for annual subscriptions, bringing it down to $278/year instead of $348."),
        ("user", "Perfect! Can you also tell me about the refund policy?"),
        ("assistant", "We offer a 30-day money-back guarantee. If you're not satisfied, you can get a full refund within 30 days."),
        ("user", "Great! I think I'm ready to sign up. What's the next step?"),
        ("assistant", "Wonderful! I can send you a signup link. You'll need to create an account and choose your payment method.")
    ]
    
    # Load up the conversation
    for speaker, message in conversation_flow:
        chat_handler.record_message(speaker, message)
        
    print("\n1️⃣ COMPLETE CONVERSATION:")
    chat_handler.show_chat()
    
    # Show metrics
    metrics = chat_handler.get_chat_metrics()
    print(f"\n📊 CHAT METRICS:")
    print(f"Messages: {metrics['total_messages']}")
    print(f"Turns: {metrics['turns']}")
    print(f"Characters: {metrics['total_characters']}")
    print(f"Words: {metrics['total_words']}")
    
    # Demo: Keep only recent stuff
    print(f"\n2️⃣ LAST 2 TURNS ONLY:")
    recent_only = chat_handler.keep_recent_turns(2)
    chat_handler.show_chat(recent_only)
    
    # Demo: Size limit
    print(f"\n3️⃣ FIT TO 300 CHARACTERS:")
    size_limited = chat_handler.fit_to_size(300)
    chat_handler.show_chat(size_limited)
    
    # Demo: Summarization
    print(f"\n4️⃣ CONVERSATION SUMMARY:")
    summary = chat_handler.create_summary(chat_handler.chat_log)
    print(f"📝 Summary: {summary}")
    
    # Demo: Auto-compression
    print(f"\n5️⃣ AUTO-COMPRESSION DEMO:")
    print("Adding more messages...")
    
    extra_messages = [
        ("user", "I have one more question about technical support."),
        ("assistant", "Of course! What would you like to know about our technical support?"),
        ("user", "What are your support hours?"),
        ("assistant", "Our support team is available 24/7 for premium subscribers.")
    ]
    
    for speaker, message in extra_messages:
        chat_handler.record_message(speaker, message)
        
    chat_handler.check_and_compress(after_n_messages=5)
    
    print("\n6️⃣ AFTER COMPRESSION:")
    chat_handler.show_chat()

def run_extraction_demo():
    """Shows the information extraction capabilities"""
    print("\n\n🚀 TASK 2: INFORMATION EXTRACTION FROM CHATS")
    print("=" * 60)
    
    miner = DataMiner()
    
    # Different chat scenarios
    test_conversations = [
        {
            "id": "chat_1",
            "content": """
            User: Hi, I'd like to create an account
            Assistant: Sure! I can help you with that. Could you provide some basic information?
            User: My name is John Smith, email is john.smith@email.com
            Assistant: Great! Any other contact information?
            User: Yes, my phone is 555-123-4567 and I'm 28 years old
            Assistant: Perfect! What's your location?
            User: I'm in New York, NY
            """
        },
        {
            "id": "chat_2", 
            "content": """
            User: I need help with my profile
            Assistant: I can help you update your profile. What needs to be changed?
            User: I moved recently. My new address is 123 Main St, Los Angeles, CA
            Assistant: I'll update your location. Is your contact info still current?
            User: My email changed to mary.johnson@newcompany.com
            Assistant: Got it. Any other updates needed?
            User: No that's all, thanks!
            """
        },
        {
            "id": "chat_3",
            "content": """
            User: Can you help me with billing?
            Assistant: Of course! What's your question about billing?
            User: I think there's an error on my account. I'm Mike Davis, age 35
            Assistant: Let me look that up. Could you confirm your email?
            User: It's mike.davis@company.org and my phone is (555) 987-6543
            Assistant: Thank you! I found your account. What seems to be the issue?
            User: The charge seems too high this month
            """
        }
    ]
    
    print("🔍 MINING DATA FROM CONVERSATIONS:")
    print("=" * 50)
    
    all_results = []
    
    for num, chat_data in enumerate(test_conversations, 1):
        print(f"\n{num}️⃣ ANALYZING {chat_data['id'].upper()}:")
        print(f"📄 Preview: {chat_data['content'][:100]}...")
        
        # Extract the goods
        found_data = miner.mine_conversation(chat_data['content'])
        
        # Quality check
        quality_report = miner.check_quality(found_data)
        
        # Save for later
        all_results.append({
            "chat_id": chat_data['id'],
            "extracted_info": found_data,
            "validation": quality_report
        })
        
        # Show what we found
        print(f"✅ FOUND:")
        for field, value in found_data.items():
            if value:
                print(f"   {field.capitalize()}: {value}")
        
        print(f"🔍 QUALITY CHECK:")
        print(f"   Valid: {quality_report['is_valid']}")
        print(f"   Fields found: {len(quality_report['extracted_fields'])}")
        
        if quality_report['warnings']:
            print(f"   ⚠️ Issues: {', '.join(quality_report['warnings'])}")
            
    # Overall performance
    print(f"\n📊 EXTRACTION PERFORMANCE:")
    print("=" * 30)
    
    fields_to_track = ['name', 'email', 'phone', 'location', 'age']
    success_rates = {f: 0 for f in fields_to_track}
    
    for result in all_results:
        for field in fields_to_track:
            if field in result['extracted_info'] and result['extracted_info'][field]:
                success_rates[field] += 1
                
    for field, successes in success_rates.items():
        percentage = (successes / len(test_conversations)) * 100
        print(f"{field.capitalize()}: {successes}/{len(test_conversations)} ({percentage:.1f}%)")
        
    return all_results

def main():
    """Runs the complete demonstration"""
    print("🎯 CONVERSATION MANAGEMENT & DATA EXTRACTION SYSTEM")
    print("=" * 70)
    
    if my_groq_key == "your-groq-api-key-here":
        print("⚠️  Please set up your Groq API key first!")
        return
    
    try:
        # Run both demonstrations
        run_conversation_demo()
        extraction_results = run_extraction_demo()
        
        print(f"\n✅ ALL TASKS COMPLETED!")
        print("=" * 50)
        print("📋 What we accomplished:")
        print("✓ Conversation management with smart truncation")
        print("✓ Automatic summarization and compression") 
        print("✓ Information extraction from unstructured chats")
        print("\n🚀 System ready for production use!")
        
    except Exception as problem:
        print(f"❌ Something went wrong: {str(problem)}")
        print("🔧 Check your API key and connection")

# Execute everything
if __name__ == "__main__":
    main()