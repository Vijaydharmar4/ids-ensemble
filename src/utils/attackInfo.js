// Comprehensive attack information with descriptions, prevention tips, and visual data

export const attackInfo = {
  'DoS Hulk': {
    name: 'DoS Hulk',
    category: 'Denial of Service',
    severity: 'High',
    icon: '💥',
    color: '#ff6b6b',
    description: 'DoS Hulk is a high-volume HTTP flood attack that overwhelms web servers by sending massive amounts of HTTP requests, causing service disruption.',
    howItWorks: 'The attacker sends a large number of HTTP GET/POST requests to the target server, consuming server resources and bandwidth until the service becomes unavailable.',
    impact: 'Server becomes unresponsive, legitimate users cannot access services, potential revenue loss, and damage to reputation.',
    preventionTips: [
      'Implement rate limiting on web servers',
      'Use Content Delivery Network (CDN) to distribute traffic',
      'Configure web application firewalls (WAF)',
      'Enable DDoS protection services (Cloudflare, AWS Shield)',
      'Monitor traffic patterns and set up alerts',
      'Scale server resources dynamically during traffic spikes',
      'Block suspicious IP addresses automatically'
    ],
    detectionSigns: [
      'Sudden spike in HTTP requests',
      'High CPU and memory usage',
      'Slow response times',
      'Connection timeouts',
      'Unusual traffic from single or multiple IPs'
    ]
  },
  'DoS Slowloris': {
    name: 'DoS Slowloris',
    category: 'Denial of Service',
    severity: 'High',
    icon: '🐌',
    color: '#ff9800',
    description: 'Slowloris is a low-bandwidth DoS attack that keeps HTTP connections open for as long as possible, exhausting server connection pools.',
    howItWorks: 'The attacker opens multiple HTTP connections and sends partial HTTP requests, keeping connections alive to consume server resources without using much bandwidth.',
    impact: 'Server connection pool exhaustion, legitimate users cannot connect, service becomes unavailable with minimal attacker bandwidth usage.',
    preventionTips: [
      'Configure connection timeouts on web servers',
      'Limit concurrent connections per IP address',
      'Use reverse proxy servers (Nginx, Apache)',
      'Enable mod_reqtimeout for Apache servers',
      'Implement connection rate limiting',
      'Monitor connection pool usage',
      'Use load balancers with connection limits'
    ],
    detectionSigns: [
      'Many incomplete HTTP connections',
      'Connection pool exhaustion',
      'Slow response times',
      'High number of TIME_WAIT connections',
      'Unusual connection patterns'
    ]
  },
  'DoS Slowhttptest': {
    name: 'DoS Slowhttptest',
    category: 'Denial of Service',
    severity: 'High',
    icon: '⏳',
    color: '#ff9800',
    description: 'SlowHTTPTest is a tool that performs slow HTTP attacks by sending HTTP requests very slowly to exhaust server resources.',
    howItWorks: 'The attacker sends HTTP requests at an extremely slow rate, keeping connections open for extended periods and consuming server resources.',
    impact: 'Server resource exhaustion, denial of service to legitimate users, potential server crashes.',
    preventionTips: [
      'Set HTTP request timeout limits',
      'Configure slow connection handling',
      'Use web application firewalls',
      'Implement connection rate limiting',
      'Monitor for slow HTTP connections',
      'Enable request timeout modules',
      'Use CDN services for protection'
    ],
    detectionSigns: [
      'Very slow HTTP request processing',
      'Long connection durations',
      'Server resource exhaustion',
      'Unusual request patterns',
      'High number of open connections'
    ]
  },
  'DoS GoldenEye': {
    name: 'DoS GoldenEye',
    category: 'Denial of Service',
    severity: 'High',
    icon: '👁️',
    color: '#ff6b6b',
    description: 'GoldenEye is a Layer 7 HTTP DoS attack tool that floods web servers with HTTP requests, similar to Hulk but with different request patterns.',
    howItWorks: 'The attacker uses GoldenEye tool to send HTTP GET requests in multiple threads, overwhelming the target server with traffic.',
    impact: 'Web server overload, service unavailability, impact on business operations and user experience.',
    preventionTips: [
      'Implement HTTP request rate limiting',
      'Use DDoS mitigation services',
      'Configure web server limits',
      'Enable traffic filtering',
      'Monitor HTTP request patterns',
      'Use load balancers',
      'Implement auto-scaling'
    ],
    detectionSigns: [
      'Massive HTTP GET request volume',
      'Server performance degradation',
      'High CPU and memory usage',
      'Response time increases',
      'Connection failures'
    ]
  },
  'DDoS': {
    name: 'DDoS',
    category: 'Distributed Denial of Service',
    severity: 'Critical',
    icon: '🌐',
    color: '#f44336',
    description: 'Distributed Denial of Service (DDoS) attack uses multiple compromised systems to flood a target with traffic, making it unavailable.',
    howItWorks: 'Attackers use a botnet (network of compromised devices) to simultaneously send massive amounts of traffic to overwhelm the target server.',
    impact: 'Complete service outage, significant financial losses, reputation damage, potential data breaches during downtime.',
    preventionTips: [
      'Use DDoS protection services (Cloudflare, AWS Shield, Akamai)',
      'Implement network-level filtering',
      'Configure rate limiting and traffic shaping',
      'Use Content Delivery Networks (CDN)',
      'Enable automatic traffic scrubbing',
      'Monitor network traffic patterns',
      'Have an incident response plan ready',
      'Use multiple data centers for redundancy'
    ],
    detectionSigns: [
      'Massive traffic spike from multiple sources',
      'Traffic from distributed IP addresses',
      'Complete service unavailability',
      'Network bandwidth saturation',
      'Unusual traffic patterns'
    ]
  },
  'PortScan': {
    name: 'PortScan',
    category: 'Reconnaissance',
    severity: 'Medium',
    icon: '🔍',
    color: '#ff9800',
    description: 'Port scanning is a reconnaissance technique used to discover open ports and services on a target system, often a precursor to attacks.',
    howItWorks: 'The attacker systematically scans target IP addresses to identify which ports are open, revealing available services and potential vulnerabilities.',
    impact: 'Information disclosure about system services, identification of potential attack vectors, preparation for targeted attacks.',
    preventionTips: [
      'Use firewall rules to block unnecessary ports',
      'Implement port knocking for sensitive services',
      'Monitor and log port scan attempts',
      'Use intrusion detection systems (IDS)',
      'Limit exposed services to public networks',
      'Use VPN for remote access',
      'Implement network segmentation',
      'Block IPs showing scan patterns'
    ],
    detectionSigns: [
      'Multiple connection attempts to various ports',
      'Rapid sequential port access attempts',
      'Connection attempts from single IP',
      'Unusual network activity patterns',
      'Failed connection attempts'
    ]
  },
  'FTP-Brute Force': {
    name: 'FTP-Brute Force',
    category: 'Brute Force',
    severity: 'High',
    icon: '🔐',
    color: '#ff6b6b',
    description: 'FTP Brute Force attack attempts to gain unauthorized access by systematically trying different username/password combinations on FTP servers.',
    howItWorks: 'The attacker uses automated tools to try thousands of password combinations until finding the correct credentials to access the FTP server.',
    impact: 'Unauthorized access to file systems, potential data theft, malware upload, system compromise, data breach.',
    preventionTips: [
      'Use strong, complex passwords',
      'Implement account lockout after failed attempts',
      'Enable two-factor authentication (2FA)',
      'Use SFTP instead of FTP (encrypted)',
      'Limit FTP access to specific IP addresses',
      'Monitor failed login attempts',
      'Disable anonymous FTP access',
      'Use VPN for FTP connections',
      'Regularly update FTP server software'
    ],
    detectionSigns: [
      'Multiple failed login attempts',
      'Repeated authentication failures',
      'Unusual login patterns',
      'Access attempts from suspicious IPs',
      'High number of authentication requests'
    ]
  },
  'SSH-Brute Force': {
    name: 'SSH-Brute Force',
    category: 'Brute Force',
    severity: 'High',
    icon: '🔑',
    color: '#ff6b6b',
    description: 'SSH Brute Force attack attempts to gain unauthorized access to SSH servers by trying multiple username/password combinations.',
    howItWorks: 'Attackers use automated scripts to systematically try different credentials until finding valid ones to gain remote server access.',
    impact: 'Unauthorized server access, potential complete system compromise, data theft, malware installation, lateral movement in network.',
    preventionTips: [
      'Disable password authentication, use SSH keys instead',
      'Change default SSH port (22)',
      'Implement fail2ban to block repeated attempts',
      'Use strong, unique passwords if keys not possible',
      'Limit SSH access to specific IP addresses',
      'Enable two-factor authentication',
      'Monitor SSH access logs',
      'Disable root login via SSH',
      'Use VPN for SSH access'
    ],
    detectionSigns: [
      'Multiple failed SSH login attempts',
      'Repeated authentication failures',
      'Login attempts from various IPs',
      'Unusual SSH connection patterns',
      'High volume of authentication requests'
    ]
  },
  'Bot': {
    name: 'Bot',
    category: 'Malware',
    severity: 'Critical',
    icon: '🤖',
    color: '#f44336',
    description: 'Bot traffic indicates compromised systems that are part of a botnet, used for coordinated attacks or malicious activities.',
    howItWorks: 'Malware infects systems, turning them into bots that can be remotely controlled to perform attacks, send spam, or steal data.',
    impact: 'System compromise, participation in DDoS attacks, data theft, spam distribution, credential theft, cryptocurrency mining.',
    preventionTips: [
      'Keep all software and systems updated',
      'Use antivirus and anti-malware solutions',
      'Implement network monitoring',
      'Educate users about phishing and malware',
      'Use application whitelisting',
      'Monitor network traffic for bot communication',
      'Implement endpoint detection and response (EDR)',
      'Regular security audits',
      'Block known malicious IPs and domains'
    ],
    detectionSigns: [
      'Unusual outbound network connections',
      'High CPU usage without user activity',
      'Suspicious network traffic patterns',
      'Communication with known C&C servers',
      'Unexpected system behavior'
    ]
  },
  'Infiltration': {
    name: 'Infiltration',
    category: 'Intrusion',
    severity: 'Critical',
    icon: '🚪',
    color: '#f44336',
    description: 'Infiltration attack involves unauthorized access to a network or system, often through exploiting vulnerabilities or stolen credentials.',
    howItWorks: 'Attackers gain unauthorized access through various means: exploiting vulnerabilities, using stolen credentials, or social engineering.',
    impact: 'Complete system compromise, data breach, lateral movement, potential data exfiltration, system damage, compliance violations.',
    preventionTips: [
      'Implement strong access controls',
      'Regular security vulnerability assessments',
      'Use multi-factor authentication',
      'Implement network segmentation',
      'Monitor for unusual access patterns',
      'Keep all systems patched and updated',
      'Use intrusion detection/prevention systems',
      'Implement least privilege access',
      'Regular security awareness training',
      'Encrypt sensitive data'
    ],
    detectionSigns: [
      'Unauthorized access attempts',
      'Unusual user activity',
      'Access from unexpected locations',
      'Privilege escalation attempts',
      'Suspicious file access patterns'
    ]
  },
  'Web Attack': {
    name: 'Web Attack',
    category: 'Web Application',
    severity: 'High',
    icon: '🌐',
    color: '#ff6b6b',
    description: 'Web attacks target web applications through various techniques like SQL injection, XSS, or other web vulnerabilities.',
    howItWorks: 'Attackers exploit web application vulnerabilities to gain unauthorized access, steal data, or compromise the application.',
    impact: 'Data breach, unauthorized access, website defacement, customer data theft, compliance violations, reputation damage.',
    preventionTips: [
      'Use Web Application Firewalls (WAF)',
      'Implement input validation and sanitization',
      'Use parameterized queries to prevent SQL injection',
      'Regular security testing and code reviews',
      'Keep web frameworks and libraries updated',
      'Implement Content Security Policy (CSP)',
      'Use HTTPS for all communications',
      'Regular penetration testing',
      'Implement rate limiting',
      'Monitor web application logs'
    ],
    detectionSigns: [
      'Suspicious SQL queries',
      'Unusual HTTP request patterns',
      'Attempts to access admin panels',
      'Suspicious file uploads',
      'Unusual error messages'
    ]
  },
  'Heartbleed': {
    name: 'Heartbleed',
    category: 'Vulnerability Exploit',
    severity: 'Critical',
    icon: '💔',
    color: '#f44336',
    description: 'Heartbleed is a critical vulnerability in OpenSSL that allows attackers to read memory contents, potentially exposing sensitive data.',
    howItWorks: 'The vulnerability allows attackers to request more data than should be returned, potentially exposing private keys, passwords, and other sensitive information.',
    impact: 'Exposure of private keys, passwords, and sensitive data, complete compromise of encrypted communications, potential data breach.',
    preventionTips: [
      'Update OpenSSL to patched versions',
      'Revoke and regenerate SSL certificates',
      'Change all passwords after patching',
      'Monitor for unusual SSL/TLS activity',
      'Use updated TLS/SSL libraries',
      'Regular security updates',
      'Implement certificate pinning',
      'Monitor SSL/TLS connections'
    ],
    detectionSigns: [
      'Unusual SSL/TLS handshake patterns',
      'Suspicious memory read attempts',
      'Unusual OpenSSL library usage',
      'Abnormal SSL connection patterns'
    ]
  },
  'benign': {
    name: 'Benign',
    category: 'Normal Traffic',
    severity: 'None',
    icon: '✅',
    color: '#4caf50',
    description: 'Normal, legitimate network traffic with no malicious activity detected.',
    howItWorks: 'Regular network communication between legitimate users and services.',
    impact: 'No security impact - this is expected normal traffic.',
    preventionTips: [
      'Continue monitoring for anomalies',
      'Maintain security best practices',
      'Keep systems updated'
    ],
    detectionSigns: [
      'Normal traffic patterns',
      'Expected user behavior',
      'Legitimate service requests'
    ]
  }
};

// Get attack info with fallback for unknown attacks
export function getAttackInfo(attackName) {
  const normalizedName = attackName?.toString().trim() || 'Unknown';
  return attackInfo[normalizedName] || {
    name: normalizedName,
    category: 'Unknown',
    severity: 'Medium',
    icon: '⚠️',
    color: '#ff9800',
    description: `Unknown attack type: ${normalizedName}. This may be a new or uncommon attack pattern.`,
    howItWorks: 'Attack pattern details are not available for this attack type.',
    impact: 'Potential security risk - requires investigation.',
    preventionTips: [
      'Monitor network traffic closely',
      'Review security logs',
      'Implement general security best practices',
      'Consult security experts'
    ],
    detectionSigns: [
      'Unusual network activity',
      'Suspicious patterns',
      'Requires further investigation'
    ]
  };
}

// Get severity color
export function getSeverityColor(severity) {
  const colors = {
    'Critical': '#f44336',
    'High': '#ff6b6b',
    'Medium': '#ff9800',
    'Low': '#ffc107',
    'None': '#4caf50'
  };
  return colors[severity] || '#ff9800';
}
