"""Email accounts. Sessions live in an HttpOnly cookie. Passwords are never
returned and failed logins use the same message either way.
"""
from flask import Blueprint, jsonify, request, session

from security import rate_limit, valid_email, valid_password
from services.account_service import authenticate, create_user, get_user

auth_bp = Blueprint('auth', __name__)


def current_user():
    return get_user(session.get('user_id'))


@auth_bp.route('/auth/register', methods=['POST'])
@rate_limit('register', limit=5, window=600)
def register():
    body = request.get_json(silent=True) or {}
    # Bots fill this hidden field. Humans never see it.
    if body.get('company'):
        return jsonify({'error': 'Could not create that account.'}), 400
    email = valid_email(body.get('email'))
    password = valid_password(body.get('password'))
    if not email or not password:
        return jsonify({
            'error': 'Use a valid email and a password of at least 10 characters.',
        }), 400
    user, error = create_user(email, password)
    if error:
        return jsonify({'error': error}), 409
    session['user_id'] = user['id']
    session.permanent = True
    return jsonify({'user': user}), 201


@auth_bp.route('/auth/login', methods=['POST'])
@rate_limit('login', limit=8, window=600)
def login():
    body = request.get_json(silent=True) or {}
    email = valid_email(body.get('email'))
    password = body.get('password') if isinstance(body.get('password'), str) else ''
    user = authenticate(email, password) if email and password else None
    if not user:
        return jsonify({'error': 'Email or password is incorrect.'}), 401
    session['user_id'] = user['id']
    session.permanent = True
    return jsonify({'user': user})


@auth_bp.route('/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True})


@auth_bp.route('/auth/me')
def me():
    user = current_user()
    if not user:
        return jsonify({'user': None})
    return jsonify({'user': user})
