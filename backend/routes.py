#installing Flask class for creating simple back-end app
from flask import Blueprint, request, render_template
from search_module.parsing_script import get_related_articles

routes = Blueprint('routes', __name__)

@routes.route('/')
def home():
    return render_template("index.html")

@routes.route('/search', methods=["GET"])
def search():
    query = request.args.get('q') 
    
    if not query:
        return render_template("index.html", error = "Please, enter the news headline!")
    
    articles = get_related_articles(query)
    return render_template("results.html", query = query, articles = articles)

