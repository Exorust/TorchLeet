export const WELCOME_MESSAGE = `Welcome to TorchLeet.
The best collection of PyTorch practice problems for ML/AI interviews based off real engineer interviews.

90 questions across 3 sets tagged with companies who ask them.
(rumor has it there's more hidden in v3 if you know where to look)

Type help to get started or exit for webview.`;

export const HELP_TEXT = `
Available commands:

  help                Show this help message
  list, ls            List all question sets
  list <set>          List questions in a set (v1, v2, v3)
  open <id>           Open a question (e.g. open v3-14)
  company <name>      Filter by company (e.g. company anthropic)
  filter <difficulty>  Filter by difficulty (basic/easy/medium/hard/expert)
  github              Open the GitHub repo
  aboutme             About the creator
  feedback <message>  Send feedback to the creator
  clear               Clear the terminal
  exit                Switch to web mode
`;

export const ABOUT_ME = `
  Chandrahas Aroori

  ML engineer building tools for the AI interview grind.
  TorchLeet started as my own study notes and grew into
  a collection of 90 PyTorch problems sourced from real
  interviews at companies like Google, Meta, and Anthropic.

  I like to hide things in plain sight.

  GitHub:   github.com/Exorust
  Project:  github.com/Exorust/TorchLeet
`;

export const GHOST_QUESTION = `
  ┌─────────────────────────────────────────────┐
  │  v3-31: ██████████████████████████           │
  │                                              │
  │  Difficulty: ███████                         │
  │  Companies:  ██████, ██████, ██████          │
  │                                              │
  │  Implement a ████████ ████████ that          │
  │  ████████ the ██████ of a ████████           │
  │  ████████ system using ██████████            │
  │  ████████████████████████████████.           │
  │                                              │
  │  Status: [CLASSIFIED]                        │
  │  Access: requires root privileges            │
  └─────────────────────────────────────────────┘
`;

export const SUDO_DENIED = `
  [sudo] password for torch: ********
  Sorry, try again.
  [sudo] password for torch: ********
  Sorry, try again.
  [sudo] password for torch: ********
  sudo: 3 incorrect password attempts
`;

export const SUDO_ROOT = `
  [sudo] password for torch: ********
  Access granted.

  ┌──────────────────────────────────────────────┐
  │  root@torchleet:~#                            │
  │                                               │
  │  Welcome, root.                               │
  │                                               │
  │  v4 is coming.                                │
  │  30 new questions. Harder. Deeper.            │
  │  Multi-GPU. Distributed training.             │
  │  The kind of problems that make               │
  │  senior engineers sweat.                      │
  │                                               │
  │  Stay tuned: github.com/Exorust/TorchLeet    │
  └──────────────────────────────────────────────┘
`;

export const MATRIX_LINES = [
  "  Wake up, engineer...",
  "  The interview is a simulation.",
  "  Follow the gradient.",
  "",
  "  ...or type 'help' to return to reality.",
];

export const CAT_SECRETS = `
  $ cat /home/torch/.secret

  ---BEGIN TRANSMISSION---
  The 31st question exists outside the index.
  You've already found it if you're reading this.
  But the real secret? There is no v4.
  There's only what you build next.
  ---END TRANSMISSION---
`;

export const LS_HOME = `
  drwxr-xr-x  torch  torch  4096  Jun 07  .
  drwxr-xr-x  root   root   4096  Jun 07  ..
  -rw-------  torch  torch    42  Jun 07  .secret
  -rw-r--r--  torch  torch   512  Jun 07  .bash_history
  drwxr-xr-x  torch  torch  4096  Jun 07  questions/
  drwxr-xr-x  torch  torch  4096  Jun 07  notebooks/
`;
