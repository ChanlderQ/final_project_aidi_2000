from flask import Flask, render_template, request, jsonify
import model as mymodel

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        # Preprocess the tweet
        processed_tweet = str(tweet)
        # Make prediction
        prediction = mymodel.find_sentiment(processed_tweet)
        # Convert prediction to sentiment label
        return jsonify(sentiment=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
