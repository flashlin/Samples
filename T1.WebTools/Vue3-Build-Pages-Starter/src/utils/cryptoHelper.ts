import CryptoJS from "crypto-js/core.js"
import sha256 from "crypto-js/sha256.js"
import Base64 from "crypto-js/enc-base64.js"


const UUID_V4_TEMPLATE = "10000000-1000-4000-8000-100000000000"

/**
 * @internal
 */
export class CryptoUtils {
  private static _randomWord(): number {
    return CryptoJS.lib.WordArray.random(1).words[0]
  }

  /**
   * Generates RFC4122 version 4 guid
   */
  public static generateUUIDv4(): string {
    const uuid = UUID_V4_TEMPLATE.replace(/[018]/g, c =>
      (+c ^ CryptoUtils._randomWord() & 15 >> +c / 4).toString(16),
    )
    return uuid.replace(/-/g, "")
  }

  /**
   * PKCE: Generate a code verifier
   */
  public static generateCodeVerifier(): string {
    return CryptoUtils.generateUUIDv4() + CryptoUtils.generateUUIDv4() + CryptoUtils.generateUUIDv4()
  }

  /**
   * PKCE: Generate a code challenge
   */
  public static generateCodeChallenge(code_verifier: string): string {
    const hashed = sha256(code_verifier)
    return Base64.stringify(hashed).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
  }
}
