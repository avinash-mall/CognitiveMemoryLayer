# Security Policy

## Supported versions

We release security updates for the current major version. Supported versions are documented in the [releases](https://github.com/your-org/CognitiveMemoryLayer/releases) page.

## Reporting a vulnerability

We take security seriously. If you believe you have found a security vulnerability, please report it responsibly.

**Do not** open a public GitHub issue for security-sensitive bugs.

**Do:**

1. **Email** the maintainers (or use the project’s private security contact if one is listed on GitHub) with:
   - A clear description of the issue and steps to reproduce
   - Impact (e.g. data exposure, privilege escalation)
   - Any suggested fix or mitigation, if you have one

2. Allow a reasonable time for a fix (e.g. 90 days) before any public disclosure.

We will acknowledge receipt and work with you to understand and address the issue. We may ask for more detail and will keep you updated on progress.

## Security practices

- **Secrets:** Do not commit API keys, passwords, or tokens. Use environment variables and `.env` (see `.env.example`); ensure `.env` is in `.gitignore`.
- **API auth:** The API uses config-based API keys (see `AUTH__API_KEY`, `AUTH__ADMIN_API_KEY`). Use strong keys in production and HTTPS.
- **Dependencies:** Keep dependencies up to date and run `pip audit` or similar as part of CI.

Thank you for helping keep Cognitive Memory Layer secure.
