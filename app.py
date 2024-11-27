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
