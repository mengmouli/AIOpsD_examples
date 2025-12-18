# Setup Instructions

If you are already familiar with Python virtual environments,
you may create and activate `.venv` in your preferred way.

Otherwise, follow the instructions below to set it up before running the code.

---

## 1. Open a terminal

Open your Python IDE (e.g., VS Code) and open a terminal,
(or open Terminal / PowerShell and navigate to this folder).

**Important:** This project is tested with **Python 3.12**.
Do **not** use very new versions such as Python 3.14, since some scientific packages may fail to install.

---

## 2. Install Python 3.12 (if you do not have it)

## Windows
Install Python 3.12 from the official Python installer, then confirm:
```powershell
py -3.12 --version
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## macOS / Linux
 ```bash
brew install python@3.12
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


