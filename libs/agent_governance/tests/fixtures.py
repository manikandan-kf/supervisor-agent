"""Fake credentials shared by the library's tests."""

# ── Credential-shaped fixtures ──────────────────────────────────────────────
#
# Assembled at run time, never written out as one literal: a complete
# credential-shaped string is indistinguishable from a real leaked credential
# to every secret scanner that matters (GitHub push protection, this
# repository's own scan). The values are obviously synthetic counting patterns,
# split at the prefix so a scanner never sees a contiguous run, while the
# assembled value exercises the detector exactly as a real credential would.
FAKE_DATABRICKS_TOKEN = "dapi" + "0123456789abcdef" * 2
FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdef" * 2 + "1234"
FAKE_GOOGLE_API_KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
# AWS's own documentation example pair — synthetic by publication.
FAKE_AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_AWS_SECRET = "wJalrXUtnFEMI/" + "K7MDENG/bPxRfiCYEXAMPLEKEY"
