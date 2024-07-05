name: Test and publish to PyPI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  release:
    types: [ created ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.7', '3.8', '3.9', '3.10']

    steps:
    - uses: actions/checkout@v2

    - name: 'Set up Python ${{matrix.python-version}}'
      uses: actions/setup-python@v2
      with:
        python-version: ${{matrix.python-version}}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        python setup.py install  # Install dependencies from setup.py

    - name: Install my package
      run: |
        python -m pip install .

    - name: Install test dependencies
      run: |
        pip install flake8 pytest coverage

    # Diagnostic steps to print the working directory and Python path
    - name: Print working directory
      run: pwd

    - name: Print Python path
      run: |
        python -c "import sys; print(sys.path)"

    - name: Check SciPy version
      run: |
        python -c "import scipy; print(scipy.__version__)"

    - name: Test with pytest
      run: |
        coverage run -m pytest

    - name: Clean up coverage data
      run: |
        # This is a workaround for the fact that the `[coverage:paths]` section
        # of `setup.cfg` is not actually applies until we run `combine`; so we
        # rename the report such that we can then "combine" it.
        mv .coverage .coverage.hack
        coverage combine
        coverage report

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v1
      with:
        token: ${{ secrets.CODECOV_TOKEN }}

  deploy:
    needs: test
    runs-on: ubuntu-latest
    name: deploy
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python '3.8'
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: "Install"
      run: |
        python -m pip install --upgrade pip;
        python -m pip install build
        pip install setuptools wheel twine
        python setup.py sdist bdist_wheel

    - uses: actions/upload-artifact@v2
      with:
        name: dist
        path: dist

    # Uncomment and configure if needed for Test PyPI
    # - name: Publish to Test PyPI (always)
    #   uses: pypa/gh-action-pypi-publish@master
    #   with:
    #     user: __token__
    #     password: ${{ secrets.test_pypi_password }}
    #     repository_url: https://test.pypi.org/legacy/

    - name: Publish to PyPI (on tag)
      if: startsWith(github.ref, 'refs/tags/v')
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        user: __token__
        password: ${{ secrets.pypi_token }}
