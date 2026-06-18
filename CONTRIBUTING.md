# Contributing

Thank you for helping improve Semantic Timeline Generator. Contributions should stay focused on document parsing, event extraction, semantic topic discovery, timeline visualization, tests, or documentation.

## Development Setup

Use Python 3.12 and work in a virtual environment:

```powershell
git clone https://github.com/ashimaryal25-ops/Semantic_Timeline_Generator.git
cd Semantic_Timeline_Generator
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the application:

```powershell
streamlit run semantic_timeline_3.py
```

## Before Opening a Pull Request

1. Create a focused branch from `main`.
2. Keep changes limited to one problem or feature.
3. Verify the application with representative text and a selectable-text PDF.
4. Run the syntax check:

```powershell
python -m py_compile semantic_timeline_3.py
```

5. Update the README when behavior, dependencies, or setup steps change.

## Reporting Bugs

Use the bug report template and include:

- Python version and operating system
- the command used to start the app
- the complete error message
- the input type (`.pdf`, `.txt`, or pasted text)
- a minimal reproducible sample when it can be shared safely

Do not upload documents containing confidential or personal information.

## Pull Request Expectations

- Explain the problem and the chosen solution.
- Describe how the change was tested.
- Avoid unrelated refactoring or generated-file changes.
- Preserve existing behavior unless the pull request explicitly changes it.

