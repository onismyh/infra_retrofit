#!/usr/bin/env node
/**
 * UserPromptSubmit Hook: Forced skill activation flow (cross-platform version)
 *
 * Event: UserPromptSubmit
 * Function: Force AI to evaluate available skills and begin implementation after activation
 */

const path = require('path');
const fs = require('fs');
const os = require('os');

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

const userPrompt = input.user_prompt || '';

// Check if it is a slash command (escape)
if (userPrompt.startsWith('/')) {
  // Distinguish commands from paths:
  // - Commands: /commit, /update-github (no second slash after the first)
  // - Paths: /Users/xxx, /path/to/file (contains path separators)
  const rest = userPrompt.substring(1);
  if (rest.includes('/')) {
    // This is a path, continue with skill scanning
  } else {
    // This is a command, skip skill evaluation
    console.log(JSON.stringify({ continue: true }));
    process.exit(0);
  }
}

const homeDir = os.homedir();

// Dynamically collect skill list
function collectSkills() {
  const skills = [];
  const skillsDir = path.join(homeDir, '.claude', 'skills');

  // 1. Collect local skills
  if (fs.existsSync(skillsDir)) {
    const skillDirs = fs.readdirSync(skillsDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => d.name);

    for (const skillName of skillDirs) {
      skills.push(skillName);
    }
  }

  // 2. Collect plugin skills
  const pluginsCache = path.join(homeDir, '.claude', 'plugins', 'cache');

  if (fs.existsSync(pluginsCache)) {
    const marketplaces = fs.readdirSync(pluginsCache, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .map(d => d.name);

    for (const marketplace of marketplaces) {
      const marketplacePath = path.join(pluginsCache, marketplace);
      const plugins = fs.readdirSync(marketplacePath, { withFileTypes: true })
        .filter(d => d.isDirectory() && !d.name.startsWith('.'))
        .map(d => d.name);

      for (const plugin of plugins) {
        const pluginPath = path.join(marketplacePath, plugin);
        const versions = fs.readdirSync(pluginPath, { withFileTypes: true })
          .filter(d => d.isDirectory())
          .map(d => d.name)
          .sort()
          .reverse();

        if (versions.length > 0) {
          const latestVersion = versions[0];
          const skillsDirPath = path.join(pluginPath, latestVersion, 'skills');

          if (fs.existsSync(skillsDirPath)) {
            const skillDirs = fs.readdirSync(skillsDirPath, { withFileTypes: true })
              .filter(d => d.isDirectory())
              .map(d => d.name);

            for (const skillName of skillDirs) {
              skills.push(`${plugin}:${skillName}`);
            }
          }
        }
      }
    }
  }

  // Deduplicate
  return [...new Set(skills)].sort();
}

// Categorize skills into groups
function categorizeSkills(skills) {
  const categories = {
    'Research & Writing': /research|paper|writing|citation|rebuttal|doc-coauthoring|latex|results-analysis|results-report|brainstorm/,
    'Development': /coding|git|code-review|bug|architecture|verification|tdd|driven-development|development-branch|planning|dispatching|executing|using-superpowers/,
    'Plugin Dev': /skill-|command-|hook-|mcp-/,
    'Design & UI': /frontend|ui-ux|web-design|canvas|brand|theme|algorithmic-art|slack-gif|figma/,
    'Documents': /docx|xlsx|pptx|pdf|internal-comms|web-artifacts/,
  };

  const grouped = {};
  for (const cat of Object.keys(categories)) {
    grouped[cat] = [];
  }
  grouped['Other'] = [];

  for (const skill of skills) {
    let matched = false;
    for (const [cat, regex] of Object.entries(categories)) {
      if (regex.test(skill)) {
        grouped[cat].push(skill);
        matched = true;
        break;
      }
    }
    if (!matched) {
      grouped['Other'].push(skill);
    }
  }

  return grouped;
}

// Keyword-to-skill mapping for pre-matching
// Note: \b doesn't work with CJK characters, so we use separate patterns
// English keywords use \b for precision; Chinese keywords match without \b
const KEYWORD_SKILL_MAP = [
  { keywords: /\b(git|github|commit|push|pull|merge|rebase|branch|tag|stash|cherry.?pick|develop|master|main)\b|分支|合并|推送|提交代码|上传.*分支|主分支/i, skills: ['git-workflow'] },
  { keywords: /\b(debug|bug|error|broken|failing|traceback|exception)\b|排查|调试|报错|出错|不工作/i, skills: ['bug-detective'] },
  { keywords: /\b(tdd|test.?driven)\b|写测试|测试驱动/i, skills: ['superpowers:test-driven-development'] },
  { keywords: /\b(code.?review|review code)\b|审查代码|代码审查/i, skills: ['code-review-excellence'] },
  { keywords: /\b(ablation)\b|实验结果|results.*analysis|统计检验|消融实验|科学绘图|实验统计/i, skills: ['results-analysis'] },
  { keywords: /\b(experiment.?report|results.?report|retrospective|wrap.?up)\b|实验报告|实验总结|实验复盘|结果总结/i, skills: ['results-report'] },
  { keywords: /\b(plan|planning)\b|规划|计划/i, skills: ['planning-with-files'] },
  { keywords: /\b(verify|verification)\b|验证/i, skills: ['verification-loop'] },
  { keywords: /\b(implement|write code|add feature|modify|refactor)\b|写代码|改代码|实现|添加功能|修改|重构/i, skills: ['daily-coding'] },
];

// Pre-match user prompt against keyword map
function suggestSkills(prompt) {
  const suggested = new Set();
  for (const { keywords, skills } of KEYWORD_SKILL_MAP) {
    if (keywords.test(prompt)) {
      for (const s of skills) suggested.add(s);
    }
  }
  return [...suggested];
}

// Generate skill list
const SKILL_LIST = collectSkills();
const SKILL_GROUPS = categorizeSkills(SKILL_LIST);
const suggestedSkills = suggestSkills(userPrompt);

// Format grouped skills (skip empty groups)
const groupedDisplay = Object.entries(SKILL_GROUPS)
  .filter(([, skills]) => skills.length > 0)
  .map(([cat, skills]) => `[${cat}] ${skills.join(', ')}`)
  .join('\n');

// Build suggested skills hint
const suggestedHint = suggestedSkills.length > 0
  ? `\n**Pre-matched skills (MUST activate these)**: ${suggestedSkills.join(', ')}\nThese skills matched keywords in the user's prompt. You MUST activate them via Skill tool.\n`
  : '';

// Generate output
const output = `## Instruction: Forced Skill Activation (Mandatory)

Silently scan the user's request against available skills. Do NOT list every skill with Yes/No.

Available skills:
${groupedDisplay}
${suggestedHint}
**Action**:
- If any skill matches → Activate via Skill tool, then output: "Activating: [skill-name] — [reason]"
- If no skill matches → Output: "No skills needed"
- Begin implementation only after activation is complete.
- When multiple skills match, activate all of them.
`;

console.log(output);

process.exit(0);
