from flask import jsonify


def response_success(resp):
    return jsonify({
        'success': resp
    })


def response_failure(err):
    return jsonify({
        'failure': err
    })
