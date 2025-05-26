#installing Flask class for creating simple back-end app
from flask import Blueprint, request, render_template
from search_module.parsing_script import get_related_articles

#importing our predictor model here
from models.predictor import predict_verdict

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template("index.html")

@routes.route('/search', methods=["GET"])
def search():
    query = request.args.get('q') 
    
    if not query:
        return render_template("index.html", error = "Please, enter the news headline!")
    
    verdict = predict_verdict(query)
    articles = get_related_articles(query)
    return render_template("results.html", query = query, articles = articles, verdict = verdict)

