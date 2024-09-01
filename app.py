from flask import Flask, request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Other routes and logic...

import subprocess


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        role = request.form['role']
        user = request.form['user']
        account = request.form['account']

        # Run the query_data.py script with the parameters
        result = run_query(query, role, user, account)

        return render_template('index.html', result=result, query=query, role=role, user=user, account=account)
    return render_template('index.html', result=None)

def run_query(query, role, user, account):
    try:
        # Run the query_data.py script with the specified parameters
        result = subprocess.check_output(
            ['python', 'query_data.py', query, '--role', role, '--user', user, '--account', account],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        return result
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.output}"

if __name__ == '__main__':
    app.run(debug=True)

