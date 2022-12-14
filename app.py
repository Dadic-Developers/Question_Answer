# !/usr/bin/python
#_*_ coding=utf-8 _*_

from flask import Flask
from flask import abort
from flask import jsonify
from flask import request
from main import QuestionSimilarity

app = Flask(__name__)
Q_sim = QuestionSimilarity()

@app.route("/questions", methods=["POST"])
def run():
    if not request.json or not "message" in request.json:
        abort(405, description="NOT ALLOW")
    txt = request.json["message"]
    print(request.json["message"])
    if len((txt).split()) >= 5:
        question = Q_sim.SimilarityCalculation(txt)
        if question:
            return jsonify(question), 200
        else:
            return jsonify({'message': 'Result not found'}), 200
    else:
        return jsonify({'message': 'لطفا متنی با طول بیش از پنج کلمه وارد کنید.'}), 400


# @app.route("/answer", methods=["POST"])
# def run():
#     if not request.json or not "question_id" in request.json:
#         abort(405, description="NOT ALLOW")
#     qid = request.json["question_id"]
#     add_ans = request.json["add_answers"]
#     print(qid)
#
#     answer = json.dumps(Q_sim.AnswerOfQuestion(qid, add_ans), ensure_ascii=False)
#     return jsonify({'answer': answer }), 200

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(host="0.0.0.0", debug=True)