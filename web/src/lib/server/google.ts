export interface IdToken {
  iss: string;
  aud: string;
  exp: number;
  sub: string;
  email?: string;
  name?: string;
}

/** Claims of an id_token received straight from Google's token endpoint over TLS. */
export function checkIdToken(claims: IdToken, clientId: string, nowS: number): boolean {
  return (
    (claims.iss === "https://accounts.google.com" || claims.iss === "accounts.google.com") &&
    claims.aud === clientId &&
    claims.exp > nowS &&
    Boolean(claims.sub)
  );
}

