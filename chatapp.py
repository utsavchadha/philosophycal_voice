from __future__ import unicode_literals
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import spacy

import numpy as np
from numpy import dot
from numpy.linalg import norm

nlp = spacy.load('en_core_web_lg')
doc_nietzsche = nlp(open("data/nietzsche_texts.txt").read())
doc_camus = nlp(open("data/camus_texts.rtf").read())
doc_simone = nlp(open("data/simone_texts.rtf").read())

tokens_nietzsche = list(set([w.text for w in doc_nietzsche if w.is_alpha]))
tokens_camus = list(set([w.text for w in doc_camus if w.is_alpha]))
tokens_simone = list(set([w.text for w in doc_simone if w.is_alpha]))


def vec(s):
    return nlp.vocab[s].vector


def cosine(v1, v2):
    if(norm(v1) > 0 and norm(v2) > 0):
        return dot(v1, v2) / (norm(v1) * norm(v2))
    else:
        return 0.0


def spacy_closest(token_list, vec_to_check, n=10):
    return sorted(token_list, key=lambda x: cosine(vec_to_check, vec(x)), reverse=True)[:n]


#print(spacy_closest(tokens, vec("crime")))


def sentvec(s):
    sent = nlp(s)
    return meanv([w.vector for w in sent])


sentences_nietzsche = list(doc_nietzsche.sents)
sentences_camus = list(doc_camus.sents)
sentences_simone = list(doc_simone.sents)


def meanv(coords):
    # assumes every item in coords has same length as item 0
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean


def spacy_closest_sent(space, input_str, n=5):
    input_vec = sentvec(input_str)
    return sorted(space, key=lambda x: cosine(np.mean([w.vector for w in x], axis=0), input_vec), reverse=True)[:n]


# https://flask-socketio.readthedocs.io/en/latest/
# https://github.com/socketio/socket.io-client

app = Flask(__name__)

app.config['SECRET_KEY'] = 'AJDJSAKABAK'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('./home.html')


@app.route('/talk/<philosopher_name>')
def openchat(philosopher_name=None):
    return render_template('./chatApp.html', philosopher=philosopher_name)


@socketio.on('my event')
def handle_my_custom_event(data):
    print('recived something: ' + str(data))

#    output = "oh lord! why have you forsaken me?"
    output = ''
    question = data['message']
    sentences = ''
    philosopher_name = data['philosopher_name']

    if philosopher_name == "nietzsche":
        sentences = sentences_nietzsche
    elif philosopher_name == "camus":
        sentences = sentences_camus
    elif philosopher_name == "simone":
        sentences = sentences_simone
    # print("\n\nUSER QUERY")
    # print("----------------------------")
    # print(question + '\n\n\n')
    for sent in spacy_closest_sent(sentences, question):
        output += " " + sent.text.strip(' \n\t\r')

    # print(" NIETZSCHE, ON A HARD DRIVE")
    # print("----------------------------")
    output = output.replace('- ZARATHUSTRA', '')
    output = output.replace('Beyond Good and Evil', '')
    print(output)
    # with open('user_test.txt', 'a') as the_file:
    #     the_file.write('\n' + question + '\n\n' + output + '\n')

    response = {}
    response['user_name'] = data['user_name']
    response['responder_name'] = philosopher_name
    response['message'] = output
    print(response)

    response_json = json.dumps(response)
    socketio.emit('my response', response_json)


if __name__ == '__main__':
    socketio.run(app, debug=True)
