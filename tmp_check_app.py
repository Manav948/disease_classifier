from pathlib import Path
path = Path('app.py')
text = path.read_text(encoding='utf-8')
for i, line in enumerate(text.splitlines(), 1):
    if i >= 100 and i <= 120:
        print(f'{i}: {line}')
