from MLFlow import ModelCard
import sys

if __name__ == "__main__":
    try:
        input = sys.argv[1]
        parts = input.rsplit(' ', 1)
        ModelCard(parts[0], int(parts[1]))
    except Exception as e:
        print(e)