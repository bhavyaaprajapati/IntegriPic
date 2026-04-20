with open("analysis/visualization_service.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("'currentColor'", "'#e2e8f0'").replace('"currentColor"', '"#e2e8f0"')

with open("analysis/visualization_service.py", "w", encoding="utf-8") as f:
    f.write(text)
