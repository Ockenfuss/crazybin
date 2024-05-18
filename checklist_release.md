# Steps before a release:

## Tests
- Run all tests! Use python from the command line to ensure no unnecessary print output remains!
```bash
python3 -m unittest discover -s tests/unit
python3 -m unittest discover -s tests/integration
```

- Check also the creation of the examples in the corresponding folder

## Documentation
- Update whats-new.md
- Bump versions in `pyproject.toml` and `__init__.py`

## Release
```bash
python3 -m build
twine upload dist/crazybin-1.1.0-py3-none-any.whl dist/crazybin-1.1.0.tar.gz
```