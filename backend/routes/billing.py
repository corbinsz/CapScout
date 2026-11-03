"""Stripe-ready checkout. With no secret key this returns a stub and does not
charge anyone. A live key is read from the environment and is never written
into the repo.
"""
from flask import Blueprint, jsonify, request

import config
from routes.auth import current_user
from security import rate_limit
from services.account_service import set_tier

billing_bp = Blueprint('billing', __name__)

PRO_PRICE_LABEL = '$29/month'


def stripe_configured():
    key = config.STRIPE_SECRET_KEY
    price = config.STRIPE_PRICE_ID
    if not key or not price:
        return False
    if not (key.startswith('sk_test_') or key.startswith('sk_live_')):
        return False
    if not price.startswith('price_'):
        return False
    return True


@billing_bp.route('/billing/status')
def billing_status():
    user = current_user()
    return jsonify({
        'stripe_configured': stripe_configured(),
        'price_label': PRO_PRICE_LABEL,
        'tier': user['tier'] if user else None,
        'signed_in': bool(user),
    })


@billing_bp.route('/billing/checkout', methods=['POST'])
@rate_limit('checkout', limit=10, window=600)
def checkout():
    user = current_user()
    if not user:
        return jsonify({'error': 'Sign in before upgrading.', 'needs_auth': True}), 401
    if user['tier'] == 'pro':
        return jsonify({'mode': 'already_pro', 'configured': stripe_configured()})
    if not stripe_configured():
        return jsonify({
            'mode': 'stub',
            'configured': False,
            'price_label': PRO_PRICE_LABEL,
            'message': (
                'Stripe is not connected on this server. The owner needs to set '
                'STRIPE_SECRET_KEY and STRIPE_PRICE_ID. No card is charged.'
            ),
        })
    try:
        import stripe
    except ImportError:
        return jsonify({
            'mode': 'stub',
            'configured': False,
            'message': 'The stripe package is not installed on the server yet. No card is charged.',
        })

    stripe.api_key = config.STRIPE_SECRET_KEY
    origin = request.headers.get('Origin') or request.host_url.rstrip('/')
    try:
        session = stripe.checkout.Session.create(
            mode='subscription',
            customer_email=user['email'],
            line_items=[{'price': config.STRIPE_PRICE_ID, 'quantity': 1}],
            success_url=f'{origin}/account?checkout=success',
            cancel_url=f'{origin}/?checkout=cancel',
            client_reference_id=str(user['id']),
            metadata={'user_id': str(user['id'])},
        )
    except Exception:
        return jsonify({'error': 'Stripe could not start a checkout. No card was charged.'}), 502
    return jsonify({'mode': 'stripe', 'configured': True, 'url': session.url})


@billing_bp.route('/billing/webhook', methods=['POST'])
def webhook():
    """Mark a user Pro after Stripe signs the event. Unsigned events are refused."""
    if not config.STRIPE_WEBHOOK_SECRET:
        return jsonify({'error': 'Stripe webhooks are not configured.'}), 501
    try:
        import stripe
    except ImportError:
        return jsonify({'error': 'The stripe package is not installed.'}), 501
    payload = request.get_data()
    signature = request.headers.get('Stripe-Signature', '')
    try:
        event = stripe.Webhook.construct_event(payload, signature, config.STRIPE_WEBHOOK_SECRET)
    except Exception:
        return jsonify({'error': 'Invalid Stripe signature.'}), 400
    if event.get('type') == 'checkout.session.completed':
        data = event['data']['object']
        user_id = (data.get('metadata') or {}).get('user_id') or data.get('client_reference_id')
        customer = data.get('customer')
        if user_id:
            set_tier(user_id, 'pro', stripe_customer_id=customer)
    return jsonify({'received': True})


@billing_bp.route('/billing/dev-activate', methods=['POST'])
def dev_activate():
    """Test-only upgrade. Refused unless CAPSCOUT_ALLOW_DEV_BILLING=1."""
    if not config.CAPSCOUT_ALLOW_DEV_BILLING:
        return jsonify({'error': 'Not found.'}), 404
    user = current_user()
    if not user:
        return jsonify({'error': 'Sign in first.'}), 401
    updated = set_tier(user['id'], 'pro')
    return jsonify({'user': updated, 'mode': 'dev'})
