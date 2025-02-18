from flask import Flask, render_template, request, redirect, url_for
from fakenews import predict_news, store_feedback, retrain_model_with_feedback
from flask_apscheduler import APScheduler
import os
from waitress import serve

# Render-specific configuration
PORT = int(os.environ.get('PORT', 10000))  


app = Flask(__name__)

# ------------------------------
# Scheduler Configuration
# ------------------------------
class Config:
    SCHEDULER_API_ENABLED = True

app.config.from_object(Config())

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

# Define a scheduled job to retrain the model every 2 hours
@scheduler.task('interval', id='retrain_job', hours=2)
def scheduled_retrain():
    print("Scheduled retraining started...")
    retrain_model_with_feedback()
    print("Scheduled retraining complete.")

# ------------------------------
# Flask Routes
# ------------------------------
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
        # Store feedback in the database (score is set to 0 by default here)
        store_feedback(news_text, predicted_verdict, correct_label, feedback_details, score=0)
        return redirect(url_for('index'))
    else:
        news_text = request.args.get('news_text', '')
        predicted_verdict = request.args.get('predicted_verdict', '')
        return render_template('feedback.html', news_text=news_text, predicted_verdict=predicted_verdict)

# You can also create a manual retrain route if desired:
@app.route('/retrain', methods=['POST'])
def retrain():
    retrain_model_with_feedback()
    return "Model retrained with feedback data.", 200

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=PORT)
