# Webhooks

Webhooks notify your application when events happen in your Nimbus workspace, such as `payment.succeeded` or `customer.deleted`.

## Delivery and retries

Nimbus sends webhook events as HTTPS POST requests with a JSON body. Your endpoint must respond with a 2xx status within 10 seconds. Failed deliveries are retried up to 8 times over 24 hours using exponential backoff.

## Signatures

Every webhook request includes a `Nimbus-Signature` header containing an HMAC-SHA256 signature of the payload. Verify the signature using your endpoint's signing secret before processing the event.

## Delivery logs

Webhook delivery logs, including request payloads and response codes, are retained for 30 days. After 30 days the logs are permanently deleted and cannot be recovered. Logs can be viewed and filtered in the dashboard under Developers > Webhooks.
