#!/bin/sh
export FLASK_APP=./app/predict_next_word_api.py
pip run flask --debug run -h 0.0.0.0