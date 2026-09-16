"""Fake credentials shared by the library's tests."""

# ── Credential-shaped fixtures ──────────────────────────────────────────────
#
# Assembled at run time, never written out as one literal: a complete credential-shaped
# string trips GitHub push protection, while the split value still exercises the detector.
FAKE_DATABRICKS_TOKEN = "dapi" + "0123456789abcdef" * 2
FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdef" * 2 + "1234"
FAKE_GOOGLE_API_KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
# AWS's own documentation example pair — synthetic by publication.
FAKE_AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_AWS_SECRET = "wJalrXUtnFEMI/" + "K7MDENG/bPxRfiCYEXAMPLEKEY"
