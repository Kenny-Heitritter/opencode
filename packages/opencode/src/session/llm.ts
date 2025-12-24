import { Provider } from "@/provider/provider"
import { Log } from "@/util/log"
import {
  generateText,
  streamText,
  wrapLanguageModel,
  type ModelMessage,
  type StreamTextResult,
  type Tool,
  type ToolSet,
} from "ai"
import { clone, mergeDeep, pipe } from "remeda"
import { ProviderTransform } from "@/provider/transform"
import { Config } from "@/config/config"
import { Instance } from "@/project/instance"
import type { Agent } from "@/agent/agent"
import type { MessageV2 } from "./message-v2"
import { Plugin } from "@/plugin"
import { SystemPrompt } from "./system"
import { ToolRegistry } from "@/tool/registry"
import { Flag } from "@/flag/flag"

export namespace LLM {
  const log = Log.create({ service: "llm" })

  export const OUTPUT_TOKEN_MAX = Flag.OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX || 32_000

  export type StreamInput = {
    user: MessageV2.User
    sessionID: string
    model: Provider.Model
    agent: Agent.Info
    system: string[]
    abort: AbortSignal
    messages: ModelMessage[]
    small?: boolean
    tools: Record<string, Tool>
    retries?: number

    /**
     * For OpenAI Responses API multi-step flows, pass the previous response ID
     * so tool outputs can be attached to the correct tool calls.
     */
    previousResponseId?: string
  }

  export type StreamOutput = StreamTextResult<ToolSet, unknown>

  export async function stream(input: StreamInput) {
    const l = log
      .clone()
      .tag("providerID", input.model.providerID)
      .tag("modelID", input.model.id)
      .tag("sessionID", input.sessionID)
      .tag("small", (input.small ?? false).toString())
      .tag("agent", input.agent.name)
    l.info("stream", {
      modelID: input.model.id,
      providerID: input.model.providerID,
    })
    const [language, cfg] = await Promise.all([Provider.getLanguage(input.model), Config.get()])

    const system = SystemPrompt.header(input.model.providerID)
    system.push(
      [
        // use agent prompt otherwise provider prompt
        ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
        // any custom prompt passed into this call
        ...input.system,
        // any custom prompt from last user message
        ...(input.user.system ? [input.user.system] : []),
      ]
        .filter((x) => x)
        .join("\n"),
    )

    const header = system[0]
    const original = clone(system)
    await Plugin.trigger("experimental.chat.system.transform", {}, { system })
    if (system.length === 0) {
      system.push(...original)
    }
    // rejoin to maintain 2-part structure for caching if header unchanged
    if (system.length > 2 && system[0] === header) {
      const rest = system.slice(1)
      system.length = 0
      system.push(header, rest.join("\n"))
    }

    const provider = await Provider.getProvider(input.model.providerID)

    const params = await Plugin.trigger(
      "chat.params",
      {
        sessionID: input.sessionID,
        agent: input.agent,
        model: input.model,
        provider: Provider.getProvider(input.model.providerID),
        message: input.user,
      },
      {
        temperature: input.model.capabilities.temperature
          ? (input.agent.temperature ?? ProviderTransform.temperature(input.model))
          : undefined,
        topP: input.agent.topP ?? ProviderTransform.topP(input.model),
        topK: ProviderTransform.topK(input.model),
        options: (() => {
          const mergeAny = mergeDeep as unknown as (a: any, b: any) => any
          let options: any = {}

          options = mergeAny(options, ProviderTransform.options(input.model, input.sessionID, provider.options))

          // Disable parallel tool calls for GLM models to improve reliability
          if (input.model.id.includes("glm-") || input.model.api.id.includes("glm-")) {
            options = mergeAny(options, { parallelToolCalls: false })
          }

          if (input.small) {
            options = mergeAny(options, ProviderTransform.smallOptions(input.model))
          }

          options = mergeAny(options, input.model.options)
          options = mergeAny(options, input.agent.options)
          return options
        })(),
      },
    )

    l.info("params", {
      params,
    })

    if (
      input.previousResponseId &&
      (input.model.api.npm === "@ai-sdk/openai" || input.model.api.npm === "@ai-sdk/azure")
    ) {
      // Avoid remeda.mergeDeep here to prevent TS deep instantiation
      ;(params.options as any).previousResponseId = input.previousResponseId
      ;(params.options as any).store = true
    }

    const maxOutputTokens = ProviderTransform.maxOutputTokens(
      input.model.api.npm,
      params.options,
      input.model.limit.output,
      OUTPUT_TOKEN_MAX,
    )

    const tools = await resolveTools(input)

    const messages: ModelMessage[] = [
      ...system.map(
        (x): ModelMessage => ({
          role: "system",
          content: x,
        }),
      ),
      ...input.messages,
    ]

    const providerOptions = ProviderTransform.providerOptions(input.model, params.options)
    const activeTools = Object.keys(tools).filter((x) => x !== "invalid")

    const wrappedModel = wrapLanguageModel({
      model: language,
      middleware: [
        {
          async transformParams(args) {
            if (args.type === "stream" || args.type === "generate") {
              // @ts-expect-error
              args.params.prompt = ProviderTransform.message(args.params.prompt, input.model)
            }
            return args.params
          },
        },
      ],
    })

    // vLLM OpenAI-compatible endpoints sometimes do not include tool calls in streaming mode
    // even when the model emits a tool call marker. Fall back to a single non-streaming request
    // and let SessionProcessor extract/execute tool calls from the returned text.
    const baseURL = (provider.options as any)?.baseURL
    const useNonStreaming =
      input.model.api.npm === "@ai-sdk/openai-compatible" &&
      typeof baseURL === "string" &&
      baseURL.includes("/api/vllm/")

    if (useNonStreaming) {
      const result = await generateText({
        temperature: params.temperature,
        topP: params.topP,
        topK: params.topK,
        providerOptions,
        activeTools,
        tools,
        maxOutputTokens,
        abortSignal: input.abort,
        headers: {
          ...(input.model.providerID.startsWith("opencode")
            ? {
                "x-opencode-project": Instance.project.id,
                "x-opencode-session": input.sessionID,
                "x-opencode-request": input.user.id,
                "x-opencode-client": Flag.OPENCODE_CLIENT,
              }
            : undefined),
          ...input.model.headers,
        },
        maxRetries: input.retries ?? 0,
        messages,
        model: wrappedModel,
        experimental_telemetry: { isEnabled: cfg.experimental?.openTelemetry },
      })

      const fullStream = (async function* () {
        yield { type: "start" as const }
        yield { type: "start-step" as const }

        if (result.reasoningText) {
          const id = "reasoning_" + crypto.randomUUID().replace(/-/g, "")
          yield { type: "reasoning-start" as const, id }
          yield { type: "reasoning-delta" as const, id, text: result.reasoningText }
          yield { type: "reasoning-end" as const, id }
        }

        yield { type: "text-start" as const }
        if (result.text) {
          yield { type: "text-delta" as const, text: result.text }
        }
        yield { type: "text-end" as const }

        yield {
          type: "finish-step" as const,
          finishReason: result.finishReason,
          usage: result.usage,
          providerMetadata: result.providerMetadata,
        }
        yield { type: "finish" as const }
      })()

      return { fullStream } as any
    }

    return streamText({
      onError(error) {
        l.error("stream error", {
          error,
        })
      },
      async experimental_repairToolCall(failed) {
        const candidates = [
          failed.toolCall.toolName,
          failed.toolCall.toolName.toLowerCase(),
          failed.toolCall.toolName.replace(/[^a-zA-Z0-9_]/g, "_"),
          failed.toolCall.toolName.toLowerCase().replace(/[^a-zA-Z0-9_]/g, "_"),
        ]

        for (const candidate of candidates) {
          if (candidate !== failed.toolCall.toolName && tools[candidate]) {
            l.info("repairing tool call", {
              tool: failed.toolCall.toolName,
              repaired: candidate,
            })
            return {
              ...failed.toolCall,
              toolName: candidate,
            }
          }
        }

        return {
          ...failed.toolCall,
          input: JSON.stringify({
            tool: failed.toolCall.toolName,
            error: failed.error.message,
          }),
          toolName: "invalid",
        }
      },
      temperature: params.temperature,
      topP: params.topP,
      topK: params.topK,
      providerOptions,
      activeTools,
      tools,
      maxOutputTokens,
      abortSignal: input.abort,
      headers: {
        ...(input.model.providerID.startsWith("opencode")
          ? {
              "x-opencode-project": Instance.project.id,
              "x-opencode-session": input.sessionID,
              "x-opencode-request": input.user.id,
              "x-opencode-client": Flag.OPENCODE_CLIENT,
            }
          : undefined),
        ...input.model.headers,
      },
      maxRetries: input.retries ?? 0,
      messages,
      model: wrappedModel,
      experimental_telemetry: { isEnabled: cfg.experimental?.openTelemetry },
    })
  }

  async function resolveTools(input: Pick<StreamInput, "tools" | "agent" | "user">) {
    const enabled = pipe(
      input.agent.tools,
      mergeDeep(await ToolRegistry.enabled(input.agent)),
      mergeDeep(input.user.tools ?? {}),
    )
    for (const [key, value] of Object.entries(enabled)) {
      if (value === false) delete input.tools[key]
    }
    return input.tools
  }
}
