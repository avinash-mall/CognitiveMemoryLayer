# Security Policy

## Supported versions

| Version | Supported     |
|---------|----------------|
| 1.x     | Yes            |
| 0.x     | Best effort    |

Supported versions are also documented on the [releases](https://github.com/avinash-mall/CognitiveMemoryLayer/releases) page.

## Reporting a vulnerability

We take security seriously. Please report security vulnerabilities via **GitHub Security Advisories**:

**https://github.com/avinash-mall/CognitiveMemoryLayer/security/advisories/new**

Do **not** open a public GitHub issue for security-sensitive bugs. We will acknowledge receipt and work with you to understand and address the issue. Allow a reasonable time for a fix (e.g. 90 days) before any public disclosure.

## Security practices

- **Secrets:** Do not commit API keys, passwords, or tokens. Use environment variables and `.env` (see `.env.example`); ensure `.env` is in `.gitignore`.
- **API auth:** The API uses config-based API keys (see `AUTH__API_KEY`, `AUTH__ADMIN_API_KEY`). Use strong keys in production and HTTPS.
- **Dependencies:** Keep dependencies up to date and run `pip audit` or similar as part of CI.

Thank you for helping keep Cognitive Memory Layer secure.
