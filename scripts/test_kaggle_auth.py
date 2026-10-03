from kaggle_app.kaggle_auth import (
    auth_cache_valid,
    credential_fingerprint,
    credentials_ready,
    validate_credential_fields,
)


assert credentials_ready("user", "token", "") is True
assert credentials_ready("user", "", "legacy") is True
assert credentials_ready("", "token", "") is False
assert credentials_ready("user", "", "") is False

fp1 = credential_fingerprint("user", "token")
fp2 = credential_fingerprint("user", "token")
fp3 = credential_fingerprint("user", "rotated")
assert fp1
assert fp1 == fp2
assert fp1 != fp3
assert credential_fingerprint("", "token") == ""
assert credential_fingerprint("user", "") == ""

cache = {"fingerprint": fp1, "ts": 100.0}
assert auth_cache_valid(cache, fp1, now=200.0, max_age_seconds=300) is True
assert auth_cache_valid(cache, fp1, now=500.1, max_age_seconds=300) is False
assert auth_cache_valid(cache, fp3, now=200.0, max_age_seconds=300) is False
assert auth_cache_valid(cache, "", now=200.0, max_age_seconds=300) is False

validate_credential_fields("valid-user_1", "token", "")
validate_credential_fields("valid.user", "", "legacy-key")

invalid_cases = [
    ("", "token", ""),
    ("user", "", ""),
    ("bad user", "token", ""),
    ("user\nINJECTED=1", "token", ""),
    ("user", "token\nINJECTED=1", ""),
    ("user", "", "legacy\rBAD=1"),
]
for args in invalid_cases:
    try:
        validate_credential_fields(*args)
        raise AssertionError(f"invalid credentials accepted: {args!r}")
    except ValueError:
        pass

try:
    validate_credential_fields("user", "x" * 4097, "")
    raise AssertionError("oversized token accepted")
except ValueError:
    pass

print("Kaggle auth policy tests passed.")
