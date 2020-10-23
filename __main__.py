######################### Flask Imports ###################################
from flask import Flask
from flask import request
from flask import url_for
from flask import redirect
from flask import render_template
from flask import session
###########################################################################

############################################################################################
import os
import models
############################################################################################

app = Flask(__name__, static_url_path = '/static')

@app.route('/', methods = [ 'POST' , 'GET' ])
def index():
    if(request.method == 'GET'):
        return render_template('index.html', data = '')
    elif(request.method == 'POST'):
        data = [0]*13
        for k, v in request.form.items():
            if('v' in k):
                i = int(k[1:].strip()) - 1
                data[i] = int(v) if v else 0
        data = models.predict(data)[0]
        return render_template('index.html', data = data)
    else:
        return request.url, 404

app.run()
