/**
 * Cross-platform Hook shared utility library
 * Provides cross-platform compatible shared functions for Claude Code Hooks
 *
 * @module hook-common
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

/**
 * Get Git status information
 * @param {string} cwd - Current working directory
 * @returns {Object} Git info object
 */
function getGitInfo(cwd) {
  try {
    // Check if this is a Git repository
    execSync('git rev-parse --git-dir', {
      cwd,
      stdio: 'pipe'
    });

    // Get branch name
    let branch = 'unknown';
    try {
      branch = execSync('git branch --show-current', {
        cwd,
        encoding: 'utf8',
        stdio: 'pipe'
      }).trim();
    } catch {
      branch = 'unknown';
    }

    // Get changed files
    let changes = '';
    try {
      changes = execSync('git status --porcelain', {
        cwd,
        encoding: 'utf8',
        stdio: 'pipe'
      });
    } catch {
      changes = '';
    }

    const changeList = changes.split('\n').filter(line => line.trim());
    const hasChanges = changeList.length > 0;

    return {
      is_repo: true,
      branch,
      changes_count: changeList.length,
      has_changes: hasChanges,
      changes: changeList
    };
  } catch {
    return {
      is_repo: false,
      branch: 'unknown',
      changes_count: 0,
      has_changes: false,
      changes: []
    };
  }
}

/**
 * Get todo item information
 * @param {string} cwd - Current working directory
 * @returns {Object} Todo item information
 */
function getTodoInfo(cwd) {
  const todoFiles = [
    path.join(cwd, 'docs', 'todo.md'),
    path.join(cwd, 'TODO.md'),
    path.join(cwd, '.claude', 'todos.md'),
    path.join(cwd, 'TODO'),
    path.join(cwd, 'notes', 'todo.md')
  ];

  for (const file of todoFiles) {
    if (fs.existsSync(file)) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        const totalMatches = content.match(/^[\-\*] \[[ x]\]/gim) || [];
        const total = totalMatches.length;

        const doneMatches = content.match(/^[\-\*] \[x\]/gim) || [];
        const done = doneMatches.length;

        const pending = total - done;

        return {
          found: true,
          file: path.basename(file),
          path: file,
          total,
          done,
          pending
        };
      } catch {
        continue;
      }
    }
  }

  return {
    found: false,
    file: null,
    path: null,
    total: 0,
    done: 0,
    pending: 0
  };
}


/**
 * Get Git change details
 * @param {string} cwd - Current working directory
 * @returns {Object} Change statistics
 */
function getChangesDetails(cwd) {
  try {
    let added = 0;
    let modified = 0;
    let deleted = 0;

    // 解析 name-status 输出，每行格式: "A\tfilename" 或 "M\tfilename"
    const parseNameStatus = (output) => {
      for (const line of output.trim().split('\n').filter(Boolean)) {
        const status = line.charAt(0);
        if (status === 'A') added++;
        else if (status === 'M') modified++;
        else if (status === 'D') deleted++;
      }
    };

    // 工作区变更（unstaged）
    const unstaged = execSync('git diff --name-status', {
      cwd,
      encoding: 'utf8',
      stdio: 'pipe'
    });
    parseNameStatus(unstaged);

    // 暂存区变更（staged）
    const staged = execSync('git diff --cached --name-status', {
      cwd,
      encoding: 'utf8',
      stdio: 'pipe'
    });
    parseNameStatus(staged);

    // 未跟踪文件算作 added
    const untracked = execSync('git status --porcelain', {
      cwd,
      encoding: 'utf8',
      stdio: 'pipe'
    });
    for (const line of untracked.trim().split('\n').filter(Boolean)) {
      if (line.startsWith('??')) added++;
    }

    return { added, modified, deleted };
  } catch {
    return { added: 0, modified: 0, deleted: 0 };
  }
}

/**
 * Detect temporary files
 * @param {string} cwd - Current working directory
 * @returns {Object} Temporary file information
 */
function detectTempFiles(cwd) {
  const tempFiles = [];
  const gitInfo = getGitInfo(cwd);

  // Find temp files from Git untracked files
  if (gitInfo.is_repo) {
    for (const change of gitInfo.changes) {
      if (change.startsWith('??')) {
        const file = change.substring(3).trim();
        if (/plan|draft|tmp|temp|scratch/i.test(file)) {
          tempFiles.push(file);
        }
      }
    }
  }

  // Check known temporary directories
  const tempDirs = ['plan', 'docs/plans', '.claude/temp', 'tmp', 'temp'];
  for (const dir of tempDirs) {
    const dirPath = path.join(cwd, dir);
    if (fs.existsSync(dirPath)) {
      try {
        const files = getAllFiles(dirPath);
        for (const file of files) {
          tempFiles.push(path.relative(cwd, file));
        }
      } catch {
        // Ignore errors
      }
    }
  }

  return {
    files: tempFiles,
    count: tempFiles.length
  };
}

/**
 * Recursively get all files in a directory
 * @param {string} dirPath - Directory path
 * @returns {Array<string>} List of file paths
 */
function getAllFiles(dirPath) {
  const files = [];
  const items = fs.readdirSync(dirPath);

  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      files.push(...getAllFiles(fullPath));
    } else {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Get list of enabled plugins
 * @param {string} homeDir - User home directory
 * @returns {Array<Object>} Plugin list
 */
function getEnabledPlugins(homeDir) {
  const settingsFile = path.join(homeDir, '.claude', 'settings.json');

  if (!fs.existsSync(settingsFile)) {
    return [];
  }

  try {
    const settings = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
    const enabledPlugins = settings.enabledPlugins || {};

    const plugins = [];
    for (const [pluginId, enabled] of Object.entries(enabledPlugins)) {
      if (enabled) {
        const pluginName = pluginId.split('@')[0];
        plugins.push({
          id: pluginId,
          name: pluginName
        });
      }
    }

    return plugins;
  } catch {
    return [];
  }
}

/**
 * Get list of available commands
 * @param {string} homeDir - User home directory
 * @returns {Array<Object>} Command list
 */
function getAvailableCommands(homeDir) {
  const commands = [];

  // Only collect local commands, not plugin commands
  const localCommandsDir = path.join(homeDir, '.claude', 'commands');
  if (fs.existsSync(localCommandsDir)) {
    const commandFiles = fs.readdirSync(localCommandsDir)
      .filter(f => f.endsWith('.md'));

    for (const cmdFile of commandFiles) {
      const cmdName = cmdFile.replace('.md', '');
      commands.push({
        plugin: 'local',
        name: cmdName,
        path: path.join(localCommandsDir, cmdFile)
      });
    }
  }

  return commands;
}

/**
 * Get command description
 * @param {string} cmdPath - Command file path
 * @returns {string} Command description
 */
function getCommandDescription(cmdPath) {
  try {
    const content = fs.readFileSync(cmdPath, 'utf8');
    const lines = content.split('\n');

    // Try to get description from frontmatter
    let inFrontmatter = false;
    for (const line of lines) {
      if (line.trim() === '---') {
        if (!inFrontmatter) {
          inFrontmatter = true;
        } else {
          break;
        }
        continue;
      }

      if (inFrontmatter && line.trim().startsWith('description:')) {
        const match = line.match(/description:\s*["']?(.+?)["']?$/);
        if (match) {
          return match[1].trim();
        }
      }
    }

    // Try to get from heading
    for (const line of lines) {
      const match = line.match(/^#+\s*(.+)$/);
      if (match) {
        return match[1].trim().substring(0, 50);
      }
    }

    return '';
  } catch {
    return '';
  }
}

/**
 * Format date and time
 * @param {Date} date - Date object
 * @returns {string} Formatted date-time string
 */
function formatDateTime(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');

  return `${year}/${month}/${day} ${hours}:${minutes}:${seconds}`;
}

// Export all functions
module.exports = {
  getGitInfo,
  getTodoInfo,
  getChangesDetails,
  detectTempFiles,
  getEnabledPlugins,
  getAvailableCommands,
  getCommandDescription,
  formatDateTime
};
