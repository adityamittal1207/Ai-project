from flask import Flask, request, render_template
from alternate_approach import AILegislationAnalyzer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        urls = request.form['urls'].splitlines()
        api_key = "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"

        analyzer = AILegislationAnalyzer(urls, api_key)

        final_result = analyzer.run_analysis()
        return render_template('results.html', results=final_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
