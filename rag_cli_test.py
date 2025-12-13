import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from hr_rag import retrieve_hr_policies_context


load_dotenv(override=True)


def main() -> None:
    client = OpenAI()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    logger.info("RAG CLI test. Ask HR policy questions based on HR-Policies-Manuals.pdf.")
    logger.info("Type 'exit' to quit.\n")

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        snippets = retrieve_hr_policies_context(q)
        if not snippets:
            print("RAG> No relevant HR policy found. The document might not cover this.")
            continue

        context_block = "\n\n".join(snippets)
        system_prompt = (
            "You are an HR assistant for SPIL (Sirca Paints India Ltd).\n\n"
            "Use the following HR policy excerpts to answer the user's question. "
            "You may summarize, explain, or describe what is covered in the excerpts. "
            "If the user asks a specific question and the answer is clearly not present in the excerpts, "
            "say 'This is not mentioned in the HR policies document.'\n\n"
            "Keep your answer concise (1-3 sentences).\n\n"
            f"HR policy excerpts:\n{context_block}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ],
            max_tokens=256,
        )
        answer = resp.choices[0].message.content
        print(f"HR-Bot> {answer}\n")


if __name__ == "__main__":
    main()


