rm -rf dist
python -m build 
python -m twine upload --repository pypi dist/* --verbose