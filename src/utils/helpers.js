export function normalizeAttackName(name) {
  const s = String(name).trim().toLowerCase().replace(/_/g, ' ').replace(/-/g, ' ').replace(/:/g, ' ');
  return s.split(/\s+/).join(' ');
}

export function normalizeColumns(cols) {
  return cols.map(c => String(c).replace(/\t/g, ' ').trim().toLowerCase());
}

export function explainAttackType(name) {
  const ATTACK_INFO = {
    "dos slowloris": "DoS using partial HTTP requests to exhaust server sockets.",
    "dos slowhttptest": "DoS with slow req/resp to keep many connections open.",
    "dos goldeneye": "Layer-7 HTTP DoS with rapid connections & requests.",
    "dos hulk": "Volumetric HTTP flood causing resource exhaustion.",
    "ddos": "Distributed DoS from many hosts.",
    "portscan": "Probing ports/services to enumerate open services.",
    "bot": "Compromised host contacting C2.",
    "infiltration": "Unauthorized internal access/data exfiltration.",
    "ftp-patator": "Brute-force FTP login.",
    "ssh-patator": "Brute-force SSH login.",
    "web attack xss": "Cross-Site Scripting injection.",
    "web attack sql injection": "SQL injection against backend DB.",
    "web attack brute force": "Password guessing on web login.",
    "heartbleed": "Exploit of TLS heartbeat to read server memory."
  };

  const n = normalizeAttackName(name);
  if (ATTACK_INFO[n]) return ATTACK_INFO[n];
  for (const [key, val] of Object.entries(ATTACK_INFO)) {
    if (key.includes(n)) return val;
  }
  return "Malicious traffic pattern (CICIDS).";
}