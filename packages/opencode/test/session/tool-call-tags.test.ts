import { beforeEach, describe, expect, test, mock } from "bun:test"
import path from "path"

let capturedConversationToolKeys: string[] = []
let conversationCallCount = 0
let toolExecuteCount = 0

beforeEach(() => {
  capturedConversationToolKeys = []
  conversationCallCount = 0
  toolExecuteCount = 0
})

// Avoid spawning background summary/title model calls from affecting test behavior.
mock.module("../../src/session/summary", () => ({
  SessionSummary: {
    summarize: async () => {},
  },
}))

mock.module("../../src/mcp/index", () => ({
  MCP: {
    tools: async () => ({
      // Intentionally includes '-' so GLM normalization can map it to '_' key.
      "qbraid-slurm-mcp_get_user_context": {
        description: "mock tool",
        execute: async (_args: unknown, _opts: unknown) => {
          toolExecuteCount++
          return {
            metadata: {},
            content: [{ type: "text", text: "ok" }],
          }
        },
      } as any,
    }),
  },
}))

mock.module("../../src/session/llm", () => ({
  LLM: {
    stream: async (input: any) => {
      const tools = input?.tools ?? {}
      const hasConversationTool =
        typeof tools === "object" &&
        tools !== null &&
        ("qbraid_slurm_mcp_get_user_context" in tools || "qbraid-slurm-mcp_get_user_context" in tools)

      if (hasConversationTool) {
        conversationCallCount++
        if (conversationCallCount === 1) {
          capturedConversationToolKeys = Object.keys(tools)
        }

        const text =
          conversationCallCount === 1 ? "Before <tool_call>qbraid_slurm_mcp_get_user_context</tool_call> after" : "Done"

        const fullStream = (async function* () {
          yield { type: "start" as const }
          yield { type: "start-step" as const }
          yield { type: "text-start" as const }
          yield { type: "text-delta" as const, text }
          yield { type: "text-end" as const }
          yield {
            type: "finish-step" as const,
            finishReason: "stop" as const,
            usage: { inputTokens: 0, outputTokens: 0 },
            providerMetadata: undefined,
          }
          yield { type: "finish" as const }
        })()

        return {
          fullStream,
          text: Promise.resolve(text),
        } as any
      }

      // Default behavior for other internal calls (e.g. title generation).
      const text = "Test title"
      const fullStream = (async function* () {
        yield { type: "start" as const }
        yield { type: "start-step" as const }
        yield { type: "text-start" as const }
        yield { type: "text-delta" as const, text }
        yield { type: "text-end" as const }
        yield {
          type: "finish-step" as const,
          finishReason: "stop" as const,
          usage: { inputTokens: 0, outputTokens: 0 },
          providerMetadata: undefined,
        }
        yield { type: "finish" as const }
      })()

      return {
        fullStream,
        text: Promise.resolve(text),
      } as any
    },
  },
}))

describe("GLM tool-call tag fallback", () => {
  test("normalizes MCP tool names and executes <tool_call> tag", async () => {
    const { Instance } = await import("../../src/project/instance")
    const { Session } = await import("../../src/session")
    const { SessionPrompt } = await import("../../src/session/prompt")
    const { tmpdir } = await import("../fixture/fixture")

    await using tmp = await tmpdir({
      init: async (dir) => {
        await Bun.write(
          path.join(dir, "opencode.json"),
          JSON.stringify({
            $schema: "https://opencode.ai/config.json",
            enabled_providers: ["qbraid"],
            provider: {
              qbraid: {
                name: "qBraid",
                npm: "@ai-sdk/openai-compatible",
                env: [],
                models: {
                  "glm-4.7": {
                    name: "GLM 4.7",
                    tool_call: true,
                    limit: { context: 8192, output: 2048 },
                  },
                },
                options: {
                  apiKey: "test-key",
                  baseURL: "https://sim.qbraid.com/api/vllm/v1",
                },
              },
            },
          }),
        )
      },
    })

    await Instance.provide({
      directory: tmp.path,
      fn: async () => {
        const session = await Session.create({})

        await SessionPrompt.prompt({
          sessionID: session.id,
          model: {
            providerID: "qbraid",
            modelID: "glm-4.7",
          },
          parts: [{ type: "text", text: "hi" }],
        })

        const messages = await Session.messages({ sessionID: session.id })
        const toolParts = messages.flatMap((m) => m.parts).filter((p) => p.type === "tool") as any[]

        expect(capturedConversationToolKeys).toContain("qbraid_slurm_mcp_get_user_context")
        expect(capturedConversationToolKeys).not.toContain("qbraid-slurm-mcp_get_user_context")

        const match = toolParts.find((p) => p.tool === "qbraid_slurm_mcp_get_user_context")
        expect(match).toBeDefined()
        expect(match.state.status).toBe("completed")
        expect(match.state.output).toBe("ok")
        expect(toolExecuteCount).toBe(1)

        const allText = messages
          .flatMap((m) => m.parts)
          .filter((p) => p.type === "text")
          .map((p: any) => p.text)
          .join("\n")
        expect(allText).not.toContain("<tool_call>")

        await Session.remove(session.id)
      },
    })
  })
})
