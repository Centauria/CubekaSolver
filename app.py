# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def search_with_marks(request_dict):
    left, top, right, bottom = request_dict['left'], request_dict['top'], request_dict['right'], request_dict['bottom']
    return f'<tr><td>{left}</td><td>{top}</td></tr>'


@app.route('/search', methods=['GET', 'POST'])
def search():
    return render_template('result-sheet.html', data=request.form)
