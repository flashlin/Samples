# https://github.com/flasgger/flasgger
from flask import Flask, request, jsonify
from flasgger import Swagger
from swagger_utils import create_swagger_blueprint


app = Flask(__name__)
app.config['SWAGGER'] = {
    "title": "My API",
    "description": "My API",
    "version": "1.0.2",
    "termsOfService": "",
    "hide_top_bar": False,
    "specs_route": "/swagger/"
}
Swagger(app)

@app.route('/api/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Get All Node List
    Retrieve node list
    ---
    tags:
      - Node APIs
    produces: application/json,
    parameters:
    - name: name
      in: path
      type: string
      required: true
    - name: node_id
      in: path
      type: string
      required: true
    responses:
      401:
        description: Unauthorized error
      200:
        description: Retrieve node list
        examples:
          node-list: [{"id":26},{"id":44}]
    """
    req = request.json
    messages = req['messages']
    result = {
        'outputs': '123',
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, use_reloader=False)
