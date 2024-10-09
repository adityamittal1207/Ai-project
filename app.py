from flask import Flask, request, render_template
from alternate_approach import AILegislationAnalyzer

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        urls = request.form['urls'].splitlines()
        api_key = "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"

        analyzer = AILegislationAnalyzer(urls, api_key)

        response_template = """
        For all documents listed, identify all relevant pros, cons, and neutral viewpoints. Ensure each document is addressed separately.

        - **Document Identifier:** 
          - Pros:
            - ...
          - Cons:
            - ...
          - Neutral:
            - ...

        List out every distinct point that you can extract from each document.
        """

        prompt_template = f"""
        You are an AI legislation policy-maker analyzing multiple documents. For all documents, extract every pro, con, and neutral viewpoint on generative AI. Strictly use information found in the documents for all points. Follow the response template:

        {response_template}

        <context>
        {{context}}
        </context>

        Question: {{input}}"""

        final_result = analyzer.run_analysis(prompt_template)
        return render_template('results.html', results=final_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
