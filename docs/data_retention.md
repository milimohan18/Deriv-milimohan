# Data Retention and Deletion

## Retention periods

Customer records are retained for as long as your account is active. API request logs are retained for 90 days. Audit logs are retained for 1 year on Growth plans and 7 years on Enterprise plans.

## Deleting customer data

You can delete an individual customer record at any time by calling `DELETE /v1/customers/{id}`. The record is removed from the API immediately, but it is kept in encrypted backups for up to 35 days before it is permanently purged. Enterprise customers can request expedited purging from backups by contacting support.

## Account closure

When an account is closed, all workspace data is deleted within 30 days. You can export your data before closing your account using the export endpoint.
