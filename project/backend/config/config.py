import os
from dotenv import load_dotenv

load_dotenv() 

class Config:

    def __init__(self):
        self.FLASK_ENV = os.getenv("FLASK_ENV", "development")
        self.FLASK_DEBUG = os.getenv("FLASK_DEBUG", True)
        self.FLASK_HOST = os.getenv("FLASK_HOST", "Raul")
        self.FLASK_PORT = os.getenv("FLASK_PORT", 5000)