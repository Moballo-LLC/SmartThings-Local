from __future__ import annotations

import subprocess

from tools import check_share_safety


def test_documentation_addresses_and_synthetic_uuid_are_safe():
    text = (
        "192.0.2.10 198.51.100.20 203.0.113.30 "
        "2001:db8::10 11111111-2222-3333-4444-555555555555"
    )
    assert check_share_safety.scan_text("fixture.txt", text) == []


def test_dotted_object_identifiers_are_not_ipv4_addresses():
    text = "extendedKeyUsage = 1.3.6.1.4.1.51414.0.1.2"

    assert check_share_safety.scan_text("fixture.txt", text) == []


def test_public_github_attachment_uuid_is_safe_but_bare_uuid_is_not():
    value = "cc1dca15-f272-4625-" + "a13c-2dc82283ff95"
    public_url = f"https://github.com/user-attachments/assets/{value}"

    assert check_share_safety.scan_text("README.md", public_url) == []
    assert check_share_safety.scan_text("fixture.txt", value) == [
        check_share_safety.Finding("fixture.txt", 1, "UUID")
    ]


def test_findings_never_echo_matched_content():
    cases = {
        "PEM_PRIVATE_KEY": "-----BEGIN " + "PRIVATE KEY-----",
        "EMAIL_ADDRESS": "person" + "@example.net",
        "MAC_ADDRESS": "aa:bb:cc:" + "dd:ee:ff",
        "NON_DOCUMENTATION_IPV4": "10." + "24.8.9",
        "NON_DOCUMENTATION_IPV6": "fd00" + 2 * chr(58) + "1234",
        "PRIVATE_DNS": "appliance" + chr(46) + "house" + chr(46) + "local",
        "HOME_PATH": "/" + "Users/person/private.txt",
        "CREDENTIAL_URL": "https://user:" + "pass" + chr(64) + "example.net/data",
        "SECRET_ASSIGNMENT": (
            "access_token " + chr(61) + " " + chr(34) + "never-print-this" + chr(34)
        ),
        "SERIAL_ASSIGNMENT": (
            "serialNumber " + chr(61) + " " + chr(34) + "device-123456" + chr(34)
        ),
        "REAL_TIMESTAMP": "2026-08-02" + "T12:34:56Z",
        "QR_PAYLOAD": "qr_" + "payload = value",
        "UUID": "12345678-1234-4234-9234-" + "123456789abc",
    }
    for rule_id, value in cases.items():
        findings = check_share_safety.scan_text("candidate.txt", value)
        rendered = "\n".join(finding.render() for finding in findings)
        assert f"candidate.txt:1:{rule_id}" in rendered
        assert value not in rendered


def test_binary_and_archive_inputs_are_rejected(tmp_path):
    binary = tmp_path / "fixture.bin"
    binary.write_bytes(b"before\x00after")
    capture = tmp_path / "fixture.pcap"
    capture.write_text("text-looking content")

    assert check_share_safety.scan_file(binary, "fixture.bin") == [
        check_share_safety.Finding("fixture.bin", 0, "BINARY_CONTENT")
    ]
    assert check_share_safety.scan_file(capture, "fixture.pcap") == [
        check_share_safety.Finding("fixture.pcap", 0, "FORBIDDEN_FILE_TYPE")
    ]


def test_changed_paths_include_staged_unstaged_and_untracked_files(
    tmp_path, monkeypatch
):
    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=" + "test" + chr(64) + "example.invalid",
                *args,
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
            text=True,
        )

    git("init", "--quiet")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("before\n")
    git("add", "baseline.txt")
    git("commit", "--quiet", "-m", "baseline")
    base = git("rev-parse", "HEAD").stdout.strip()

    staged = tmp_path / "staged.txt"
    staged.write_text("staged\n")
    git("add", "staged.txt")
    baseline.write_text("after\n")
    (tmp_path / "untracked.txt").write_text("untracked\n")

    monkeypatch.chdir(tmp_path)
    assert check_share_safety._changed_paths(base) == [
        "baseline.txt",
        "staged.txt",
        "untracked.txt",
    ]


def test_committed_scan_ignores_unchanged_findings_but_checks_added_lines(
    tmp_path, monkeypatch
):
    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=" + "test" + chr(64) + "example.invalid",
                *args,
            ],
            cwd=tmp_path,
            capture_output=True,
            check=True,
            text=True,
        )

    candidate = tmp_path / "candidate.txt"
    private_one = "10." + "24.8.9"
    private_two = "10." + "24.8.10"
    git("init", "--quiet")
    candidate.write_text(f"existing {private_one}\n")
    git("add", "candidate.txt")
    git("commit", "--quiet", "-m", "baseline")
    base = git("rev-parse", "HEAD").stdout.strip()

    candidate.write_text(f"existing {private_one}\nsafe addition\n")
    git("add", "candidate.txt")
    git("commit", "--quiet", "-m", "safe change")
    monkeypatch.chdir(tmp_path)
    assert check_share_safety.check_changed(base) == []

    candidate.write_text(
        f"existing {private_one}\nsafe addition\nintroduced {private_two}\n"
    )
    git("add", "candidate.txt")
    git("commit", "--quiet", "-m", "unsafe change")
    assert check_share_safety.check_changed(base) == [
        check_share_safety.Finding("candidate.txt", 3, "NON_DOCUMENTATION_IPV4")
    ]
