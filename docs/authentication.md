# Authentication

The Nimbus API supports two authentication methods: API keys and OAuth 2.0.

## API keys

API keys are intended for server-to-server integrations. Send the key in the `Authorization` header using the Bearer scheme, for example `Authorization: Bearer nb_live_xxx`. Keys prefixed with `nb_test_` only work against the sandbox environment. Each workspace can create up to 10 active API keys. Keys can be rotated or revoked at any time from the dashboard.

## OAuth 2.0

OAuth 2.0 is recommended for applications that act on behalf of other Nimbus users. The platform supports the Authorization Code flow with PKCE. Access tokens expire after 1 hour and refresh tokens expire after 30 days of inactivity.

## Unsupported methods

Basic authentication (username and password) is not supported. Mutual TLS is not currently available.
