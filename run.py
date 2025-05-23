from backend import create_app
#importing from backend create_app module

app = create_app()
#starts the development server only when run directly

if __name__ == "__main__":
    app.run(debug=True)
