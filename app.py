from flask import Flask

app = Flask(__name__)

@app.route('/')
def say_hi():
    return 'Hi'

@app.route('/hiu')
def say_hiu():
    return 'hiu'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
