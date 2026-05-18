#!/usr/bin/env bash
set -euo pipefail

#==============================================================================
# Creates a self-signed Code Signing certificate for go-ocr.
# Run ONCE. Afterwards build.sh will sign with this stable identity, so
# macOS TCC permissions (Screen Recording, Accessibility) survive rebuilds.
#==============================================================================

CERT_NAME="go-ocr-codesign"
LOGIN_KC="$HOME/Library/Keychains/login.keychain-db"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

KEY_FILE="$TMP_DIR/key.pem"
CERT_FILE="$TMP_DIR/cert.pem"
P12_FILE="$TMP_DIR/cert.p12"
CFG_FILE="$TMP_DIR/codesign.cnf"

if security find-identity -p codesigning 2>/dev/null | grep -q "$CERT_NAME"; then
    echo "Code signing identity '$CERT_NAME' already exists. Nothing to do."
    security find-identity -p codesigning | grep "$CERT_NAME"
    exit 0
fi

cat > "$CFG_FILE" <<EOF
[req]
distinguished_name = req_dn
x509_extensions    = v3_req
prompt             = no

[req_dn]
CN = $CERT_NAME

[v3_req]
basicConstraints    = critical, CA:false
keyUsage            = critical, digitalSignature
extendedKeyUsage    = critical, codeSigning
EOF

echo "==> Generating 2048-bit RSA self-signed code signing certificate"
openssl req -x509 -newkey rsa:2048 \
    -keyout "$KEY_FILE" -out "$CERT_FILE" -nodes \
    -config "$CFG_FILE" \
    -days 3650 2>&1 | tail -3

P12_PASS="go-ocr-p12"

echo "==> Packaging into PKCS12 (legacy PBE for macOS Keychain compatibility)"
openssl pkcs12 -export \
    -keypbe PBE-SHA1-3DES \
    -certpbe PBE-SHA1-3DES \
    -macalg sha1 \
    -inkey "$KEY_FILE" -in "$CERT_FILE" -out "$P12_FILE" \
    -passout "pass:$P12_PASS" \
    -name "$CERT_NAME"

echo "==> Importing into login keychain (allow codesign access)"
security import "$P12_FILE" \
    -k "$LOGIN_KC" \
    -P "$P12_PASS" \
    -T /usr/bin/codesign \
    -T /usr/bin/security

if security find-identity -p codesigning 2>/dev/null | grep -q "$CERT_NAME"; then
    echo ""
    echo "Done. Code signing identity installed:"
    security find-identity -p codesigning | grep "$CERT_NAME"
    echo ""
    echo "Next: ./build.sh"
    echo "First codesign will prompt 'codesign wants to use key XXX' --"
    echo "click 'Always Allow' so subsequent builds run unattended."
else
    echo "ERROR: import did not register a codesigning identity." >&2
    exit 1
fi
