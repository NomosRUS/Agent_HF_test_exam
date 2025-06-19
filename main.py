from app import BasicAgent

# Edit this variable with the question you want to ask while debugging
question = "What is the capital of France?"

if __name__ == "__main__":
    agent = BasicAgent()
    answer = agent(question)
    print("Question:", question)
    print("Answer:", answer)
