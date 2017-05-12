# tdepps

Pure python package to do time dependent point source analysis.

## Installation

Directly via `pip`:
```
pip3 install git+https://github.com/mennthor/tdepps.git
pip3 install git+https://github.com/mennthor/python_modules3.git
```

Or locally to make it editable (`-e` option):

```
git clone https://github.com/mennthor/tdepps.git
cd tdepps
pip3 install -e .
pip3 install git+https://github.com/mennthor/python_modules3.git
```

## Build a html version of the sphinx docs

```
pip3 install sphinx sphinx-rtd-theme
cd /install/path/tdepps/docs
make html
```

Open `_build/html/index.html` in browser.
