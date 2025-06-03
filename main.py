from transformers import pipeline

class QuestionAnsweringSystem:
    def __init__(self, model_name="gpt2"):
        self.models = {
            "distilgpt2": "distilgpt2",
            "gpt2": "gpt2",
        }
        
        if model_name not in self.models:
            print(f"⚠️  Model '{model_name}' not found. Using default 'gpt2'")
            model_name = "gpt2"
        
        self.current_model = model_name
        print(f"Loading model: {model_name}")
        
        self.pipeline = pipeline("text-generation", model=self.models[model_name], 
                               device=-1, pad_token_id=50256)
        print("Model loaded successfully")
    
    def answer(self, question):
        try:
            prompt = f"Question: {question}\nAnswer:"
            result = self.pipeline(prompt, max_new_tokens=128, do_sample=True, 
                                 temperature=0.7, pad_token_id=50256)
            
            generated = result[0]['generated_text']
            if "Answer:" in generated:
                return generated.split("Answer:")[-1].strip()
            return generated.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def run(self):
        print("\n" + "="*60)
        print("🤖 AI QUESTION ANSWERING SYSTEM")
        print("="*60)
        print("Available models:")
        model_list = " | ".join(self.models.keys())
        print(f"  {model_list}")
        print(f"Current model: {self.current_model}")
        print("="*60)
        
        while True:
            question = input("\n💬 Your question (or 'quit'): ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if not question:
                continue
                
            print(f"\n🔍 Processing with {self.current_model}...")
            answer = self.answer(question)
            print(f"\n✨ Answer:\n{answer}")
            print("-" * 40)

if __name__ == "__main__":
    qa = QuestionAnsweringSystem("gpt2")
    qa.run()
