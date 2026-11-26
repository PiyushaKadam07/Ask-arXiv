"""Check that the Gemini API key works and list the models we can use."""
import os

from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from the environment

print("Flash models this key can use:")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []) and "flash" in model.name:
        print("  ", model.name)

reply = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    contents="In one sentence, what is retrieval-augmented generation?",
)
print("\nTest answer:", reply.text)