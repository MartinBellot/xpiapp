from flask import Flask, render_template, request
from prediction import predict_sentiment

app = Flask(__name__)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
    return render_template('prediction.html', sentiment=sentiment)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
