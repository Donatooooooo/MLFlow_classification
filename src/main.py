from MLFlow import ModelCard
import sys

if __name__ == "__main__":
    input = sys.argv[1]
    parts = input.rsplit(' ', 1)
    ModelCard(parts[0], int(parts[1]))