/**
 * pianochart-dsh — turn one audio link into a piano score.
 *
 * The scoring pipeline is a Python program that already documents its own
 * command line (`src/pianochart/cli.py`). This plugin does not reimplement any
 * of it: the tool resolves the link to a local audio file, runs that CLI in a
 * child process, and reads the `report.json` it writes. Keeping the pipeline
 * out of this process matters — Demucs wants a few gigabytes and minutes of
 * CPU, which must not run inside the Harness Node runtime.
 *
 * @module pianochart-dsh
 */

import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const name = 'pianochart-dsh'
export const inject = ['tools', 'skills']

const HERE = dirname(fileURLToPath(import.meta.url))
const BRIDGE = join(HERE, 'lib', 'pianochart_bridge.py')
const TOOL_TIMEOUT_MS = 1_800_000

// The skill carries the driving instructions (which tier, preview long audio,
// what to do with a page link); the tool schema stays the single source of
// truth for arguments. Both halves ship in one bundle so `skills` is optional:
// a profile without the service still gets the tool.
const SKILL_DIR = join(HERE, 'skill')
const SKILL_BODY = join(SKILL_DIR, 'pianochart.md')
// `@deepseek-ai/dsh-skill` exports BUNDLED_SKILL_RANK = 600, but that package is
// supplied by the Harness installation rather than by every profile, so keep the
// number local and import nothing from it.
const BUNDLED_SKILL_RANK = 600

function skillCandidate(manifest) {
  return {
    name: manifest.name,
    description: manifest.description,
    invocation: { modelInvocable: true, userInvocable: true },
    provider: name,
    source: 'bundled',
    resourceBase: { kind: 'directory', path: SKILL_DIR },
    rank: BUNDLED_SKILL_RANK,
    locator: pathToFileURL(SKILL_BODY),
  }
}

async function registerSkill(ctx) {
  if (typeof ctx.skills?.registerProvider !== 'function') return
  const manifest = JSON.parse(await readFile(join(SKILL_DIR, 'manifest.json'), 'utf8'))
  const candidate = skillCandidate(manifest)
  ctx.skills.registerProvider(() => ({
    name,
    list: () => Promise.resolve([candidate]),
    async get() {
      return {
        name: candidate.name,
        description: candidate.description,
        invocation: candidate.invocation,
        provider: candidate.provider,
        source: candidate.source,
        resourceBase: candidate.resourceBase,
        content: await readFile(SKILL_BODY, 'utf8'),
      }
    },
  }))
}

/** Where the pipeline may live. First hit wins.
 *
 *  The bundle is normally a symlink or a file shipped *inside* the pipeline
 *  checkout (`<repo>/dsh-bundle`), so walking up from this file finds the
 *  checkout with no configuration at all — that is what makes a clone in any
 *  directory work. A globally installed copy (node_modules/<pkg>) has no such
 *  ancestor, so config, the env var, and the usual checkout names follow. */
function findRepo(config) {
  const candidates = [config?.repo, process.env.PIANOCHART_REPO]
  for (let dir = HERE; ; dir = dirname(dir)) {
    candidates.push(dir)
    if (dirname(dir) === dir) break
  }
  candidates.push(join(homedir(), 'piano-rescue'), join(homedir(), 'piano'))
  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || candidate.length === 0) continue
    const absolute = resolve(candidate)
    if (existsSync(join(absolute, 'src', 'pianochart', 'cli.py'))) return absolute
  }
  return ''
}

function findPython(repo, config) {
  const candidates = [
    config?.python,
    process.env.PIANOCHART_PYTHON,
    repo && join(repo, '.venv', 'bin', 'python'),
    repo && join(repo, '.venv', 'bin', 'python3'),
  ].filter((value) => typeof value === 'string' && value.length > 0)
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate
  }
  return ''
}

const PARAMETERS = {
  type: 'object',
  properties: {
    source: {
      type: 'string',
      description:
        'Audio to transcribe: an http(s) link to the audio file, a page hosting it, or a local path. '
        + 'A bare player page is fetched with yt-dlp; if that fails, read the page first '
        + 'and pass the direct audio URL it exposes.',
    },
    tier: {
      type: 'string',
      enum: ['easy', 'medium', 'hard'],
      description:
        'Difficulty. The three tiers are one arrangement in three keys: the same notes, rhythm and '
        + 'left-hand voicing, transposed so easy has the fewest black keys and hard the most, which '
        + 'puts a tier a few semitones above or below the others. Pick easy when the point is to play '
        + 'it. Playback restores the original pitch. Default easy.',
    },
    seconds: {
      type: 'number',
      description:
        'Only transcribe the first N seconds. Use it to preview a long track cheaply: the '
        + 'separation stage costs about a tenth of the audio length on CPU.',
    },
    fingering: {
      type: 'string',
      enum: ['key', 'all', 'none'],
      description:
        'Which notes carry a finger number. key marks only the few places where the hand has to '
        + 'be told how to get there (the default); all numbers every note; none leaves the staff '
        + 'unmarked.',
    },
    transpose: {
      type: 'integer',
      description: 'Shift the whole arrangement by this many semitones. Omit to keep the written key.',
    },
    out: {
      type: 'string',
      description: 'Directory for the score files. Default a new directory under the pipeline checkout.',
    },
    jianpuChars: {
      type: 'integer',
      description: 'How much numbered notation to return inline. Default 1800 characters; the file holds the rest.',
    },
  },
  required: ['source'],
  additionalProperties: false,
}

function runBridge(args) {
  return new Promise((resolvePromise) => {
    const child = spawn(args.python, [BRIDGE, ...args.argv], {
      cwd: args.repo,
      env: {
        ...process.env,
        PYTHONPATH: join(args.repo, 'src'),
        DYLD_FALLBACK_LIBRARY_PATH:
          process.env.DYLD_FALLBACK_LIBRARY_PATH || '/opt/homebrew/lib',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    let stdout = ''
    let stderr = ''
    let settled = false
    const finish = (value) => {
      if (settled) return
      settled = true
      resolvePromise(value)
    }

    const timer = setTimeout(() => {
      child.kill('SIGTERM')
      finish({ ok: false, error: `gave up after ${Math.round(TOOL_TIMEOUT_MS / 60_000)} minutes` })
    }, TOOL_TIMEOUT_MS)

    child.stdout.on('data', (chunk) => { stdout += chunk })
    child.stderr.on('data', (chunk) => {
      // Progress only; kept for the failure message and otherwise discarded.
      stderr = (stderr + chunk).slice(-4000)
    })
    child.on('error', (error) => {
      clearTimeout(timer)
      finish({ ok: false, error: `could not start the pipeline: ${error.message}` })
    })
    child.on('close', () => {
      clearTimeout(timer)
      const line = stdout.trim().split('\n').filter(Boolean).pop()
      if (!line) {
        finish({ ok: false, error: stderr.trim().slice(-800) || 'the pipeline printed nothing' })
        return
      }
      try {
        finish(JSON.parse(line))
      } catch {
        finish({ ok: false, error: `unreadable pipeline output: ${line.slice(0, 400)}` })
      }
    })
  })
}

function format(result) {
  if (!result.ok) {
    const lines = [`piano · nothing scored`, String(result.error || 'unknown failure')]
    if (result.hint) lines.push(String(result.hint))
    if (result.seconds) lines.push(`${result.seconds}s`)
    return lines.join('\n')
  }

  const evalBlock = result.similarity || {}
  const lines = []
  const title = result.tier === 'easy' ? '简单' : result.tier === 'medium' ? '中等' : '困难'
  lines.push(`piano · ${title} 档 · ${result.key_signature || '?'}`
    + (result.transpose ? `（写谱 ${result.transpose > 0 ? '+' : ''}${result.transpose} 半音，回放还原）` : ''))
  if (typeof evalBlock.chroma_cos_mean === 'number') {
    lines.push(`与原音频相似度 ${evalBlock.chroma_cos_mean.toFixed(3)}`
      + (typeof evalBlock.chroma_cos_p10 === 'number' ? `（最差 10% ${evalBlock.chroma_cos_p10.toFixed(3)}）` : ''))
  }
  lines.push(`右手 ${result.right_notes} 音 · 左手 ${result.left_notes} 音 · ${result.duration}s`
    + (result.audio_seconds ? ` · 音频 ${Math.round(result.audio_seconds)}s` : ''))
  const fing = result.fingering
  if (fing && fing.mode === 'none') {
    lines.push('指法 未标注')
  } else if (fing && fing.mode) {
    const marked = (fing.right || 0) + (fing.left || 0)
    const total = (fing.right_total || 0) + (fing.left_total || 0)
    lines.push(`指法 ${marked} 处${total ? ` / 共 ${total} 个音` : ''}`
      + (fing.mode === 'key' ? '（只标关键处）' : ''))
    const chk = result.fingering_check
    if (chk && typeof chk.in_staff_file === 'number') {
      lines.push(chk.agrees
        ? `谱面已写入 ${chk.in_staff_file} 个，左右手顺序都对`
        : `谱面只写入 ${chk.in_staff_file} 个，编配里是 ${chk.in_arrangement} 个，对不上`)
    }
  }
  if (result.staff_pages?.length) lines.push(`五线谱 ${result.staff_pages.length} 页：${result.staff_pages[0]}`)
  if (result.musicxml) lines.push(`MusicXML ${result.musicxml}`)
  if (result.midi) lines.push(`MIDI ${result.midi}`)
  if (result.jianpu_path) lines.push(`简谱 ${result.jianpu_path}`)
  if (result.jianpu) {
    lines.push('')
    lines.push(result.jianpu)
    if (result.jianpu_truncated) lines.push(`（还有更多，见上面简谱文件）`)
  }
  lines.push('')
  lines.push(`全部产物 ${result.out_dir} · ${result.seconds}s`)
  return lines.join('\n')
}

export function apply(ctx, config = {}) {
  ctx.effect(() => {
    registerSkill(ctx).catch((error) => {
      ctx.logger?.warn?.('pianochart-dsh: skill not registered', error)
    })
  })
  ctx.effect(() => ctx.tools.register({
    name: 'piano_score',
    description:
      'Turn one audio link into a piano score: five-line staff images, numbered (jianpu) notation, '
      + 'MusicXML and MIDI. Only background music is transcribed — vocals are separated out and '
      + 'dropped, and non-melodic instruments are folded into the left hand. '
      + 'This runs a local CPU pipeline and takes minutes: separation alone is about a tenth of the '
      + 'audio length. Pass `seconds` for a quick preview. Files are written to disk; the paths are '
      + 'in the result.',
    parameters: PARAMETERS,
    timeoutMs: TOOL_TIMEOUT_MS,
    output: {
      schema: {
        type: 'object',
        properties: {
          ok: { type: 'boolean' },
          tier: { type: 'string' },
          key_signature: { type: 'string' },
          transpose: { type: 'integer' },
          fingering: {
            type: 'object',
            properties: {
              mode: { type: 'string' },
              right: { type: 'integer' },
              left: { type: 'integer' },
              right_total: { type: 'integer' },
              left_total: { type: 'integer' },
            },
            additionalProperties: false,
          },
          fingering_check: {
            type: 'object',
            properties: {
              mode: { type: 'string' },
              in_arrangement: { type: 'integer' },
              in_staff_file: { type: 'integer' },
              marked_strikes: { type: 'integer' },
              non_monotone_strikes: { type: 'integer' },
              agrees: { type: 'boolean' },
            },
            additionalProperties: false,
          },
          duration: { type: 'number' },
          right_notes: { type: 'integer' },
          left_notes: { type: 'integer' },
          similarity: {
            type: 'object',
            properties: {
              chroma_cos_mean: { type: 'number' },
              chroma_cos_p10: { type: 'number' },
            },
            additionalProperties: false,
          },
          staff_pages: { type: 'array', items: { type: 'string' } },
          musicxml: { type: 'string' },
          midi: { type: 'string' },
          jianpu_path: { type: 'string' },
          jianpu: { type: 'string' },
          jianpu_truncated: { type: 'boolean' },
          out_dir: { type: 'string' },
          seconds: { type: 'number' },
          error: { type: 'string' },
          hint: { type: 'string' },
        },
        required: ['ok'],
        additionalProperties: true,
      },
      render: (_args, value) => [{ type: 'text', text: format(value) }],
    },
    execute: async (args, execution) => {
      const repo = findRepo(config)
      if (!repo) {
        return {
          ok: false,
          error: 'the pianochart pipeline is not on this machine',
          hint: 'Set PIANOCHART_REPO to its directory (the one holding src/pianochart/cli.py).',
        }
      }
      const python = findPython(repo, config)
      if (!python) {
        return {
          ok: false,
          error: `no Python environment in ${repo}`,
          hint: 'Set PIANOCHART_PYTHON to an interpreter that has the pipeline dependencies.',
        }
      }

      const outDir = typeof args.out === 'string' && args.out.trim()
        ? resolve(args.out)
        : join(repo, 'out', `link-${new Date().toISOString().replace(/[^0-9]/g, '').slice(0, 14)}`)

      const argv = [
        '--source', String(args.source),
        '--repo', repo,
        '--out', outDir,
        '--tier', args.tier || 'easy',
        '--jianpu-limit', String(args.jianpuChars ?? 1800),
      ]
      if (typeof args.seconds === 'number' && args.seconds > 0) argv.push('--seconds', String(args.seconds))
      if (Number.isInteger(args.transpose) && args.transpose !== 0) argv.push('--transpose', String(args.transpose))
      if (args.fingering) argv.push('--fingering', String(args.fingering))

      const result = await runBridge({ python, repo, argv })

      // Canceled turns must not keep a Demucs run alive.
      if (execution?.signal?.aborted && !result.ok) {
        return { ok: false, error: 'canceled' }
      }

      if (!result.ok) {
        return {
          ok: false,
          error: result.error,
          hint: result.hint,
          seconds: result.seconds,
          out_dir: outDir,
        }
      }
      return {
        ok: true,
        tier: result.tier,
        key_signature: result.key_signature ?? undefined,
        transpose: result.transpose ?? undefined,
        duration: result.duration ?? undefined,
        right_notes: result.right_notes ?? undefined,
        left_notes: result.left_notes ?? undefined,
        fingering: result.fingering && result.fingering.mode ? result.fingering : undefined,
        fingering_check: result.fingering_check && result.fingering_check.mode
          ? result.fingering_check
          : undefined,
        similarity: result.similarity?.chroma_cos_mean !== undefined
          ? {
              chroma_cos_mean: result.similarity.chroma_cos_mean,
              chroma_cos_p10: result.similarity.chroma_cos_p10,
            }
          : undefined,
        staff_pages: Array.isArray(result.staff_pages) ? result.staff_pages : undefined,
        musicxml: result.musicxml ?? undefined,
        midi: result.midi ?? undefined,
        jianpu_path: result.jianpu_path ?? undefined,
        jianpu: result.jianpu || undefined,
        jianpu_truncated: result.jianpu_truncated || undefined,
        out_dir: result.out_dir || outDir,
        seconds: result.seconds,
      }
    },
  }))
}
