from flask import Flask, render_template
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name_exp", type=str)
parser.add_argument("dir_html", type=str, choices=["pred", "gt"])
args = parser.parse_args()

dir_exp = "../drnn/experiments"
name_exp = args.name_exp
dir_html = os.path.join(dir_exp, name_exp, "test", args.dir_html)
print(dir_html)

app = Flask(__name__, template_folder=dir_html)


@app.route('/<string:file_idx>')
def serve(file_idx):
    #print(path)
    #path = file_idx + ".html"
    path = file_idx
    return render_template(path)


@app.route('/test')
def test():
    return "test"


if __name__ == "__main__":
    # allow access from anyware : 0.0.0.0
    app.run(debug=True, host='0.0.0.0', port=8080)
