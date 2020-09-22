#!/bin/bash

echo flake8
flake8 || exit 1
echo black
black --check . || exit 1
echo pydocstyle
pydocstyle mcp || exit 1
echo mypy
mypy --ignore-missing-imports --package mcp --package tests || exit 1
