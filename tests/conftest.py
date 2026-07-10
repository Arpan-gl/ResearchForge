from pathlib import Path
import os
import tempfile


_REPO_TEMP = Path('C:/tmp/researchforge-pytest-temp')
_REPO_TEMP.mkdir(parents=True, exist_ok=True)
os.environ['TMP'] = str(_REPO_TEMP)
os.environ['TEMP'] = str(_REPO_TEMP)
tempfile.tempdir = str(_REPO_TEMP)
