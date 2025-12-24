import { MessageV2 } from "./message-v2"
import { Log } from "@/util/log"
import { Identifier } from "@/id/id"
import { Session } from "."
import { Agent } from "@/agent/agent"
import { Permission } from "@/permission"
import { Snapshot } from "@/snapshot"
import { SessionSummary } from "./summary"
import { Bus } from "@/bus"
import { SessionRetry } from "./retry"
import { SessionStatus } from "./status"
import { Plugin } from "@/plugin"
import type { Provider } from "@/provider/provider"
import { LLM } from "./llm"
import { Config } from "@/config/config"

export namespace SessionProcessor {
  const DOOM_LOOP_THRESHOLD = 3
  const log = Log.create({ service: "session.processor" })

  export type Info = Awaited<ReturnType<typeof create>>
  export type Result = Awaited<ReturnType<Info["process"]>>

  export function create(input: {
    assistantMessage: MessageV2.Assistant
    sessionID: string
    model: Provider.Model
    abort: AbortSignal
  }) {
    const toolcalls: Record<string, MessageV2.ToolPart> = {}
    const preserveReasoning =
      typeof input.model.capabilities.interleaved === "object" &&
      input.model.capabilities.interleaved.field === "reasoning_content"
    let snapshot: string | undefined
    let blocked = false
    let attempt = 0

    const result = {
      get message() {
        return input.assistantMessage
      },
      partFromToolCall(toolCallID: string) {
        return toolcalls[toolCallID]
      },
      async process(streamInput: LLM.StreamInput) {
        log.info("process")
        const shouldBreak = (await Config.get()).experimental?.continue_loop_on_deny !== true
        while (true) {
          try {
            let currentText: MessageV2.TextPart | undefined
            let reasoningMap: Record<string, MessageV2.ReasoningPart> = {}
            let sawReasoning = false
            let sawSyntheticToolCall = false

            const parseTaggedToolCalls = (
              text: string,
            ): { cleaned: string; calls: Array<{ name: string; args: any }> } => {
              const openTag = "<tool_call>"
              const closeTag = "</tool_call>"
              if (!text.includes(openTag)) return { cleaned: text, calls: [] }

              const calls: Array<{ name: string; args: any }> = []
              let cleaned = text

              const re = /<tool_call>([\s\S]*?)<\/tool_call>/g
              let match: RegExpExecArray | null
              while ((match = re.exec(text)) !== null) {
                const inner = (match[1] ?? "").trim()
                if (!inner) continue

                let name = ""
                let args: any = {}

                // Try JSON-first
                if (inner.startsWith("{")) {
                  try {
                    const parsed = JSON.parse(inner)
                    if (parsed && typeof parsed === "object") {
                      if (typeof (parsed as any).name === "string") {
                        name = (parsed as any).name
                        args = (parsed as any).arguments ?? {}
                      } else if (typeof (parsed as any).tool === "string") {
                        name = (parsed as any).tool
                        args = (parsed as any).arguments ?? {}
                      }
                    }
                  } catch {
                    // fall back to token parsing
                  }
                }

                if (!name) {
                  const firstSpace = inner.search(/\s/)
                  if (firstSpace === -1) {
                    name = inner
                  } else {
                    name = inner.slice(0, firstSpace)
                    const rest = inner.slice(firstSpace).trim()
                    if (rest) {
                      try {
                        args = JSON.parse(rest)
                      } catch {
                        args = {}
                      }
                    }
                  }
                }

                if (name) {
                  calls.push({ name, args })
                }
              }

              cleaned = cleaned.replace(re, "").trim()
              return { cleaned, calls }
            }

            const resolveToolName = (name: string) => {
              const candidates = [
                name,
                name.toLowerCase(),
                name.replace(/[^a-zA-Z0-9_]/g, "_"),
                name.toLowerCase().replace(/[^a-zA-Z0-9_]/g, "_"),
              ]
              return candidates.find((c) => c in streamInput.tools)
            }

            const executeSyntheticToolCalls = async (calls: Array<{ name: string; args: any }>) => {
              for (const call of calls) {
                const toolName = resolveToolName(call.name)
                if (!toolName) continue

                const toolCallId = "call_" + crypto.randomUUID().replace(/-/g, "")
                const start = Date.now()

                const part = await Session.updatePart({
                  id: Identifier.ascending("part"),
                  messageID: input.assistantMessage.id,
                  sessionID: input.assistantMessage.sessionID,
                  type: "tool",
                  tool: toolName,
                  callID: toolCallId,
                  state: {
                    status: "running",
                    input: call.args ?? {},
                    time: {
                      start,
                    },
                  },
                })

                toolcalls[toolCallId] = part as MessageV2.ToolPart

                try {
                  const output = await (streamInput.tools as any)[toolName].execute(call.args ?? {}, {
                    toolCallId,
                    abortSignal: input.abort,
                  })

                  await Session.updatePart({
                    ...(part as any),
                    state: {
                      status: "completed",
                      input: call.args ?? {},
                      output: output.output,
                      metadata: output.metadata,
                      title: output.title,
                      time: {
                        start,
                        end: Date.now(),
                      },
                      attachments: output.attachments,
                    },
                  })

                  sawSyntheticToolCall = true
                } catch (error) {
                  await Session.updatePart({
                    ...(part as any),
                    state: {
                      status: "error",
                      input: call.args ?? {},
                      error: (error as any).toString(),
                      metadata: error instanceof Permission.RejectedError ? error.metadata : undefined,
                      time: {
                        start,
                        end: Date.now(),
                      },
                    },
                  })

                  if (error instanceof Permission.RejectedError) {
                    blocked = shouldBreak
                  }
                } finally {
                  delete toolcalls[toolCallId]
                }
              }
            }

            const stream = await LLM.stream(streamInput)

            for await (const value of stream.fullStream) {
              input.abort.throwIfAborted()
              switch (value.type) {
                case "start":
                  SessionStatus.set(input.sessionID, { type: "busy" })
                  break

                case "reasoning-start":
                  sawReasoning = true
                  if (value.id in reasoningMap) {
                    continue
                  }
                  reasoningMap[value.id] = {
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "reasoning",
                    text: "",
                    time: {
                      start: Date.now(),
                    },
                    metadata: value.providerMetadata,
                  }
                  break

                case "reasoning-delta":
                  sawReasoning = true
                  if (value.id in reasoningMap) {
                    const part = reasoningMap[value.id]
                    part.text += value.text
                    if (value.providerMetadata) part.metadata = value.providerMetadata
                    if (part.text) await Session.updatePart({ part, delta: value.text })
                  }
                  break

                case "reasoning-end":
                  sawReasoning = true
                  if (value.id in reasoningMap) {
                    const part = reasoningMap[value.id]

                    const tagged = parseTaggedToolCalls(part.text)
                    if (tagged.calls.length) {
                      await executeSyntheticToolCalls(tagged.calls)
                      part.text = tagged.cleaned
                    }

                    if (!preserveReasoning) {
                      part.text = part.text.trimEnd()
                    }

                    part.time = {
                      ...part.time,
                      end: Date.now(),
                    }
                    if (value.providerMetadata) part.metadata = value.providerMetadata
                    await Session.updatePart(part)
                    delete reasoningMap[value.id]
                  }
                  break

                case "tool-input-start":
                  const part = await Session.updatePart({
                    id: toolcalls[value.id]?.id ?? Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "tool",
                    tool: value.toolName,
                    callID: value.id,
                    state: {
                      status: "pending",
                      input: {},
                      raw: "",
                    },
                  })
                  toolcalls[value.id] = part as MessageV2.ToolPart
                  break

                case "tool-input-delta":
                  break

                case "tool-input-end":
                  break

                case "tool-call": {
                  const match = toolcalls[value.toolCallId]
                  if (match) {
                    const part = await Session.updatePart({
                      ...match,
                      tool: value.toolName,
                      state: {
                        status: "running",
                        input: value.input,
                        time: {
                          start: Date.now(),
                        },
                      },
                      metadata: value.providerMetadata,
                    })
                    toolcalls[value.toolCallId] = part as MessageV2.ToolPart

                    const parts = await MessageV2.parts(input.assistantMessage.id)
                    const lastThree = parts.slice(-DOOM_LOOP_THRESHOLD)

                    if (
                      lastThree.length === DOOM_LOOP_THRESHOLD &&
                      lastThree.every(
                        (p) =>
                          p.type === "tool" &&
                          p.tool === value.toolName &&
                          p.state.status !== "pending" &&
                          JSON.stringify(p.state.input) === JSON.stringify(value.input),
                      )
                    ) {
                      const permission = await Agent.get(input.assistantMessage.mode).then((x) => x.permission)
                      if (permission.doom_loop === "ask") {
                        await Permission.ask({
                          type: "doom_loop",
                          pattern: value.toolName,
                          sessionID: input.assistantMessage.sessionID,
                          messageID: input.assistantMessage.id,
                          callID: value.toolCallId,
                          title: `Possible doom loop: "${value.toolName}" called ${DOOM_LOOP_THRESHOLD} times with identical arguments`,
                          metadata: {
                            tool: value.toolName,
                            input: value.input,
                          },
                        })
                      } else if (permission.doom_loop === "deny") {
                        throw new Permission.RejectedError(
                          input.assistantMessage.sessionID,
                          "doom_loop",
                          value.toolCallId,
                          {
                            tool: value.toolName,
                            input: value.input,
                          },
                          `You seem to be stuck in a doom loop, please stop repeating the same action`,
                        )
                      }
                    }
                  }
                  break
                }
                case "tool-result": {
                  const match = toolcalls[value.toolCallId]
                  if (match && match.state.status === "running") {
                    await Session.updatePart({
                      ...match,
                      state: {
                        status: "completed",
                        input: value.input,
                        output: value.output.output,
                        metadata: value.output.metadata,
                        title: value.output.title,
                        time: {
                          start: match.state.time.start,
                          end: Date.now(),
                        },
                        attachments: value.output.attachments,
                      },
                    })

                    delete toolcalls[value.toolCallId]
                  }
                  break
                }

                case "tool-error": {
                  const match = toolcalls[value.toolCallId]
                  if (match && match.state.status === "running") {
                    await Session.updatePart({
                      ...match,
                      state: {
                        status: "error",
                        input: value.input,
                        error: (value.error as any).toString(),
                        metadata: value.error instanceof Permission.RejectedError ? value.error.metadata : undefined,
                        time: {
                          start: match.state.time.start,
                          end: Date.now(),
                        },
                      },
                    })

                    if (value.error instanceof Permission.RejectedError) {
                      blocked = shouldBreak
                    }
                    delete toolcalls[value.toolCallId]
                  }
                  break
                }
                case "error":
                  throw value.error

                case "start-step":
                  snapshot = await Snapshot.track()
                  await Session.updatePart({
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.sessionID,
                    snapshot,
                    type: "step-start",
                  })
                  break

                case "finish-step":
                  const usage = Session.getUsage({
                    model: input.model,
                    usage: value.usage,
                    metadata: value.providerMetadata,
                  })
                  const finishReason = sawSyntheticToolCall ? "tool-calls" : value.finishReason

                  input.assistantMessage.finish = finishReason
                  input.assistantMessage.cost += usage.cost
                  input.assistantMessage.tokens = usage.tokens
                  await Session.updatePart({
                    id: Identifier.ascending("part"),
                    reason: finishReason,

                    snapshot: await Snapshot.track(),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "step-finish",
                    tokens: usage.tokens,
                    cost: usage.cost,
                    metadata: value.providerMetadata,
                  })
                  await Session.updateMessage(input.assistantMessage)
                  if (snapshot) {
                    const patch = await Snapshot.patch(snapshot)
                    if (patch.files.length) {
                      await Session.updatePart({
                        id: Identifier.ascending("part"),
                        messageID: input.assistantMessage.id,
                        sessionID: input.sessionID,
                        type: "patch",
                        hash: patch.hash,
                        files: patch.files,
                      })
                    }
                    snapshot = undefined
                  }
                  SessionSummary.summarize({
                    sessionID: input.sessionID,
                    messageID: input.assistantMessage.parentID,
                  })
                  break

                case "text-start":
                  currentText = {
                    id: Identifier.ascending("part"),
                    messageID: input.assistantMessage.id,
                    sessionID: input.assistantMessage.sessionID,
                    type: "text",
                    text: "",
                    time: {
                      start: Date.now(),
                    },
                    metadata: value.providerMetadata,
                  }
                  break

                case "text-delta":
                  if (currentText) {
                    currentText.text += value.text
                    if (value.providerMetadata) currentText.metadata = value.providerMetadata
                    if (currentText.text)
                      await Session.updatePart({
                        part: currentText,
                        delta: value.text,
                      })
                  }
                  break

                case "text-end":
                  if (currentText) {
                    if (!sawReasoning && input.model.capabilities.reasoning && currentText.text.includes("</think>")) {
                      const openTag = "<think>"
                      const closeTag = "</think>"
                      const openIdx = currentText.text.indexOf(openTag)
                      const closeIdx = currentText.text.indexOf(closeTag)
                      if (closeIdx !== -1) {
                        const reasoningText =
                          openIdx !== -1 && openIdx < closeIdx
                            ? currentText.text.slice(openIdx + openTag.length, closeIdx)
                            : currentText.text.slice(0, closeIdx)
                        const remainingText =
                          openIdx !== -1 && openIdx < closeIdx
                            ? currentText.text.slice(0, openIdx) + currentText.text.slice(closeIdx + closeTag.length)
                            : currentText.text.slice(closeIdx + closeTag.length)
                        if (reasoningText) {
                          await Session.updatePart({
                            id: Identifier.ascending("part"),
                            messageID: input.assistantMessage.id,
                            sessionID: input.assistantMessage.sessionID,
                            type: "reasoning",
                            text: reasoningText,
                            time: {
                              start: currentText.time?.start ?? Date.now(),
                              end: Date.now(),
                            },
                          })
                          currentText.text = remainingText
                          sawReasoning = true
                        }
                      }
                    }
                    const tagged = parseTaggedToolCalls(currentText.text)
                    if (tagged.calls.length) {
                      await executeSyntheticToolCalls(tagged.calls)
                      currentText.text = tagged.cleaned
                    }

                    currentText.text = currentText.text.trimEnd()
                    const textOutput = await Plugin.trigger(
                      "experimental.text.complete",
                      {
                        sessionID: input.sessionID,
                        messageID: input.assistantMessage.id,
                        partID: currentText.id,
                      },
                      { text: currentText.text },
                    )
                    currentText.text = textOutput.text
                    currentText.time = {
                      start: Date.now(),
                      end: Date.now(),
                    }
                    if (value.providerMetadata) currentText.metadata = value.providerMetadata
                    await Session.updatePart(currentText)
                  }
                  currentText = undefined
                  break

                case "finish":
                  break

                default:
                  log.info("unhandled", {
                    ...value,
                  })
                  continue
              }
            }
          } catch (e: any) {
            log.error("process", {
              error: e,
              stack: JSON.stringify(e.stack),
            })
            const error = MessageV2.fromError(e, { providerID: input.model.providerID })
            const retry = SessionRetry.retryable(error)
            if (retry !== undefined) {
              attempt++
              const delay = SessionRetry.delay(attempt, error.name === "APIError" ? error : undefined)
              SessionStatus.set(input.sessionID, {
                type: "retry",
                attempt,
                message: retry,
                next: Date.now() + delay,
              })
              await SessionRetry.sleep(delay, input.abort).catch(() => {})
              continue
            }
            input.assistantMessage.error = error
            Bus.publish(Session.Event.Error, {
              sessionID: input.assistantMessage.sessionID,
              error: input.assistantMessage.error,
            })
          }
          if (snapshot) {
            const patch = await Snapshot.patch(snapshot)
            if (patch.files.length) {
              await Session.updatePart({
                id: Identifier.ascending("part"),
                messageID: input.assistantMessage.id,
                sessionID: input.sessionID,
                type: "patch",
                hash: patch.hash,
                files: patch.files,
              })
            }
            snapshot = undefined
          }
          const p = await MessageV2.parts(input.assistantMessage.id)
          for (const part of p) {
            if (part.type === "tool" && part.state.status !== "completed" && part.state.status !== "error") {
              await Session.updatePart({
                ...part,
                state: {
                  ...part.state,
                  status: "error",
                  error: "Tool execution aborted",
                  time: {
                    start: Date.now(),
                    end: Date.now(),
                  },
                },
              })
            }
          }
          input.assistantMessage.time.completed = Date.now()
          await Session.updateMessage(input.assistantMessage)
          if (blocked) return "stop"
          if (input.assistantMessage.error) return "stop"
          return "continue"
        }
      },
    }
    return result
  }
}
