#!/bin/sh
export FLASK_APP=/app/src/main.py
pipenv run flask --debug run -h 0.0.0.0