from app import app
import os

if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8080, debug =True, use_reloader=True)
