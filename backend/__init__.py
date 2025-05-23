
#this file is the config + setup layer.
#shows where is template folder
from flask import Flask
from backend.routes import routes  # ensure routes.py exists in same folder

def create_app():
    app = Flask(__name__, template_folder='../frontend/templates')  # tell Flask where your HTML lives
    app.register_blueprint(routes)
    return app
