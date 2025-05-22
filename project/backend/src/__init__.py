from flask import Flask
from config.config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    
    return app


