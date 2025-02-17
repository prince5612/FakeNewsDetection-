
from flask import Flask, render_template, request, redirect, url_for
from fakenews import predict_news, store_feedback ,retrain_model_with_feedback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    result = predict_news(news_text)
    # Pass the news_text and predicted verdict as query parameters for feedback
    return render_template('index.html', result=result, news_text=news_text, predicted_verdict=result["verdict"])

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        news_text = request.form['news_text']
        predicted_verdict = request.form['predicted_verdict']
        correct_label = request.form['correct_label']
        feedback_details = request.form['feedback_details']
        # You can set score to 0 here or derive it from context
        store_feedback(news_text, predicted_verdict, correct_label, feedback_details, score=0)
        return redirect(url_for('index'))
    else:
        news_text = request.args.get('news_text', '')
        predicted_verdict = request.args.get('predicted_verdict', '')
        return render_template('feedback.html', news_text=news_text, predicted_verdict=predicted_verdict)

@app.route('/retrain', methods=['GET'])
def retrain():
    retrain_model_with_feedback()
    return "Model retrained with feedback data.", 200

if __name__ == '__main__':
    app.run(debug=False)
