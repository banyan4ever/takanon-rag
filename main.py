import config


def main():
    for name in ("GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "DATABASE_URL"):
        print(f"{name}: {'set' if getattr(config, name) else 'MISSING'}")


if __name__ == "__main__":
    main()
