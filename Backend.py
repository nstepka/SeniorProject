from Flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('view.html')

if __name__ == '__main__':
    app.run(debug=True)