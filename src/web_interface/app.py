#import svm_prediction
import os
import csv
import json
import sys
import random
sys.path.append('../source/modeling/')
sys.path.append('../')
from config import Config

from constructiveness_predictor import ConstructivenessPredictor

# Load models
predictor = ConstructivenessPredictor()

from flask import Flask
from flask import render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_prediction", methods=["GET", "POST"])
def get_prediction():
    text = request.args.get("result")
    selected_model = request.args.get("model")
    label = 'Default'
    print("SELECTED MODEL: ", selected_model)
    if selected_model == "svm":
        print("You selected SVM.")
        #prediction = svm_prediction.predict(text, model_path)[0]
        label = predictor.predict_svm(text)
    elif selected_model == "lstm":
        print("You selected bidirectional LSTM.")
        label = predictor.predict_bilstm(text)
        # do whatever needs to be done for lstm
    else:
        print("Did you forget to select the model? Please select the model first.")
        return jsonify(predicted_label="Please select a model first")

    #label = "Constructive"

    print(text)
    return jsonify(predicted_label="According to our " + selected_model.upper() + " model the comment is likely to be " + label.upper() + ".")

@app.route("/select_model", methods=["GET", "POST"])
def select_model():
    model = request.args.get('result')
    return jsonify(resp='You selected: ' + model)

@app.route("/get_feedback", methods=["GET", "POST"])
def get_feedback():
    text = request.args.get('comment_text')
    correct_label = request.args.get('correct_label')
    comments = request.args.get('comments')
    print('comment_text: ', text)
    print('correct_label: ', correct_label)
    print('comments: ', comments)

    file_exists = os.path.isfile(Config.FEEDBACK_CSV_PATH)
    with open(Config.FEEDBACK_CSV_PATH, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Comment_text', 'Label', 'Comments'])
        writer.writerow([text, correct_label, comments])
        return jsonify(feedback='Thank you for your feedback!')

@app.route("/get_sample_comment", methods=["GET", "POST"])
def get_sample_comment():
        sample_comments = (["I have 3 daughters, and I told them that Mrs. Clinton lost because she did not have a platform. If she did, she, and her party, did a poor job explaining it to the masses. The only message that I got from her was that Mr. Trump is not fit to be in office and that she wanted to be the first female President. I honestly believe that she lost because she offered no hope, or direction, to the average American. Mr. Trump, with all his shortcomings, at least offered change and some hope. He now has to make it happen or he will be out of power in 4 years.",
                            "Is this a joke? Marie Henein as feminist crusader, advising us what to tell our daughters?? no thanks",
                            "Why don't the NDP also promise 40 acres and a mule? They will never lead this country. Panderers to socialists and unionists.",
                            "In my opinion, criticizing the new generation is not going to solve any problem. If you want to produce children, you should be prepared to pay for their care.",
                            "Simpson is right: it's a political winner and a policy dud - just political smoke and mirrors. Mulcair is power-hungry. He wants Canada to adopt a national childcare model so he can hang on to seats in Quebec, that's all. Years ago I worked with a political strategist working to get a Liberal candidate elected in Conservative Calgary. He actually told his client to talk about national daycare - this was in the early 90's. The Liberal candidate said, `Canada can't afford that!' to which the strategist responded `Just say the words, you don't have to actually do it. It'll be good for votes.' I could barely believe the cynicism, but over the years I've come to realize that's what it is: vote getting and power politics. Same thing here.",
                            "If it happens once in a while to everyone, it's a crime. If it happens disproportionately to a particular demographic, it's more than just a crime. Harper is way out to lunch on this one. This is not an issue the police can solve. It's going to take a society, a country, to do that, and that takes leadership and information."])
        sample_comment = random.choice(sample_comments)
        return jsonify(sample_comment=sample_comment)



if __name__ == '__main__':
    app.run(host=Config.HOST, port=Config.PORT)

