#!/usr/bin/env node
/**
 * Stop Hook: Display basic status + AI summary prompt (Cross-platform)
 *
 * Event: Stop
 * Purpose: Display Git status, change statistics and temp files when session stops
 */

const common = require('./hook-common');

// Read stdin input
let input = {};
try {
  const stdinData = require('fs').readFileSync(0, 'utf8');
  if (stdinData.trim()) {
    input = JSON.parse(stdinData);
  }
} catch {
  // Use default empty object
}

const cwd = input.cwd || process.cwd();
const reason = input.reason || 'task_complete';
const binding = common.getProjectMemoryBinding(cwd);

// Build message
function buildMessage() {
  let msg = '\n---\n';
  msg += '✅ Session ended\n\n';

  // Git info
  const gitInfo = common.getGitInfo(cwd);

  if (gitInfo.is_repo) {
    msg += '📁 Git repository\n';
    msg += `  Branch: ${gitInfo.branch}\n`;

    if (gitInfo.has_changes) {
      const changesDetails = common.getChangesDetails(cwd);

      msg += '  Changes:\n';
      if (changesDetails.added > 0) msg += `    增加: ${changesDetails.added} files\n`;
      if (changesDetails.modified > 0) msg += `    修改: ${changesDetails.modified} files\n`;
      if (changesDetails.deleted > 0) msg += `    删除: ${changesDetails.deleted} files\n`;
    } else {
      msg += '  Status: clean\n';
    }
  } else {
    msg += '📁 Not a Git repository\n';
  }

  msg += '\n';

  // Temp file detection
  const tempInfo = common.detectTempFiles(cwd);

  if (tempInfo.count > 0) {
    msg += `🧹 Temp files: ${tempInfo.count}\n`;

    const grouped = {};
    for (const file of tempInfo.files) {
      const dir = require('path').dirname(file);
      if (!grouped[dir]) grouped[dir] = [];
      grouped[dir].push(require('path').basename(file));
    }

    for (const [dir, files] of Object.entries(grouped)) {
      msg += `  📂 ${dir}/ (${files.length})\n`;
      for (let i = 0; i < Math.min(3, files.length); i++) {
        msg += `    • ${files[i]}\n`;
      }
      if (files.length > 3) {
        msg += `    ... and ${files.length - 3} more\n`;
      }
    }
  } else {
    msg += '✅ No temp files\n';
  }

  if (binding.bound) {
    msg += '\n🧠 Bound Obsidian KB\n';
    msg += `  Project: ${binding.projectId || 'unknown'}\n`;
    msg += '  Minimum maintenance after research-state turns:\n';
    msg += '    • Daily/YYYY-MM-DD.md\n';
    msg += `    • ${binding.memoryPath || '.claude/project-memory/<project_id>.md'}\n`;
    msg += '    • 00-Hub.md (only when top-level project status changes)\n';
  }

  msg += '---';

  return msg;
}

// Build and return
const systemMessage = buildMessage();

const result = {
  continue: true,
  systemMessage: systemMessage
};

console.log(JSON.stringify(result));

process.exit(0);
