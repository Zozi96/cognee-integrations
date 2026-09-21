/**
 * Cognee-backed chat memory for the n8n AI Agent, built on @n8n/ai-node-sdk.
 *
 * Every agent turn is stored as a Cognee session Q&A entry
 * (POST /v1/remember/entry) and read back from the session detail endpoint
 * (GET /v1/sessions/{sessionId}). Cognee cognifies those entries into the
 * knowledge graph server-side, so the same conversation is later reachable
 * through Recall / Search as well as through this memory.
 *
 * The HTTP transport is injected so the classes stay free of n8n runtime
 * imports and can be unit-tested with a fake request function.
 */
import { BaseChatHistory, BaseChatMemory } from '@n8n/ai-node-sdk';
import type { Message } from '@n8n/ai-node-sdk';

import { buildRememberEntryPayload } from '../Cognee/payloads';

/** Minimal request contract: path is relative to `{baseUrl}/api`. */
export interface CogneeRequestOptions {
	method: 'GET' | 'POST';
	url: string;
	body?: Record<string, unknown>;
	/** Resolve with `undefined` instead of throwing when the server answers 404. */
	allowNotFound?: boolean;
}

export type CogneeRequest = (options: CogneeRequestOptions) => Promise<unknown>;

/** Error raised by the transport so callers can branch on the HTTP status. */
export class CogneeRequestError extends Error {
	constructor(
		message: string,
		public readonly statusCode?: number,
	) {
		super(message);
		this.name = 'CogneeRequestError';
	}
}

/**
 * Pull an HTTP status out of the error shapes n8n's request helpers throw
 * (NodeApiError.httpCode, axios-style response.status, plain statusCode).
 */
export function toCogneeRequestError(error: unknown): CogneeRequestError {
	if (error instanceof CogneeRequestError) return error;
	const err = (error ?? {}) as Record<string, unknown>;
	const candidates: unknown[] = [
		err.httpCode,
		err.statusCode,
		(err.response as Record<string, unknown> | undefined)?.status,
		(err.response as Record<string, unknown> | undefined)?.statusCode,
		(err.cause as Record<string, unknown> | undefined)?.statusCode,
		(err.cause as Record<string, unknown> | undefined)?.httpCode,
	];
	let statusCode: number | undefined;
	for (const candidate of candidates) {
		const parsed = Number(candidate);
		if (Number.isFinite(parsed) && parsed > 0) {
			statusCode = parsed;
			break;
		}
	}
	const message =
		typeof err.message === 'string' && err.message ? err.message : 'Cognee request failed';
	return new CogneeRequestError(message, statusCode);
}

/** The server keeps the recent tail of a session; GET /v1/sessions/{id} returns at most this many Q&A pairs. */
export const MAX_SESSION_PAIRS = 20;

const NO_QUESTION = '(no user message)';
const NO_ANSWER = '(no assistant message)';

export const CLEAR_UNSUPPORTED_MESSAGE =
	'Cognee Memory cannot clear a session: Cognee has no endpoint that deletes a single session. This affects the Chat Memory Manager operations that wipe memory first — Delete Messages, and Insert Messages with Override All Messages. Use a new Session ID to start a fresh conversation, or the Cognee node (Memory → Forget) to remove the dataset the session was remembered into.';

function textOf(message: Message): string {
	return message.content
		.map((part) => ('text' in part && typeof part.text === 'string' ? part.text : ''))
		.filter((text) => text.length > 0)
		.join('\n');
}

function textMessage(role: 'user' | 'assistant', text: string): Message {
	return { role, content: [{ type: 'text', text }] };
}

/** Q&A entry shape returned inside `qas` by GET /v1/sessions/{sessionId}. */
export interface SessionQaEntry {
	question?: unknown;
	answer?: unknown;
	time?: unknown;
	qa_id?: unknown;
}

/**
 * Turn the session detail response into alternating user/assistant messages,
 * oldest first. Entries with no usable text are skipped so a malformed row
 * never produces an empty message.
 */
export function sessionEntriesToMessages(body: unknown): Message[] {
	if (!body || typeof body !== 'object') return [];
	const qas = (body as { qas?: unknown }).qas;
	if (!Array.isArray(qas)) return [];

	const messages: Message[] = [];
	for (const entry of qas as SessionQaEntry[]) {
		if (!entry || typeof entry !== 'object') continue;
		const question = typeof entry.question === 'string' ? entry.question : '';
		const answer = typeof entry.answer === 'string' ? entry.answer : '';
		if (!question && !answer) continue;
		messages.push(textMessage('user', question || NO_QUESTION));
		messages.push(textMessage('assistant', answer || NO_ANSWER));
	}
	return messages;
}

/** Keep only the most recent `windowSize` user/assistant pairs. */
export function windowMessages(messages: Message[], windowSize: number): Message[] {
	const pairs = Math.max(1, Math.floor(windowSize));
	const max = pairs * 2;
	return messages.length > max ? messages.slice(-max) : messages;
}

export interface CogneeMemoryConfig {
	sessionId: string;
	datasetName: string;
	windowSize: number;
	request: CogneeRequest;
}

/**
 * ChatHistory over a Cognee session. Used directly by n8n's Chat Memory
 * Manager node (insert / get / delete) and as the `chatHistory` of
 * {@link CogneeChatMemory}.
 */
export class CogneeChatHistory extends BaseChatHistory {
	constructor(
		private readonly sessionId: string,
		private readonly datasetName: string,
		private readonly request: CogneeRequest,
	) {
		super();
	}

	async getMessages(): Promise<Message[]> {
		// 404 = no session record yet (first turn) → the transport yields undefined → [].
		const body = await this.request({
			method: 'GET',
			url: `/v1/sessions/${encodeURIComponent(this.sessionId)}`,
			allowNotFound: true,
		});
		return sessionEntriesToMessages(body);
	}

	/**
	 * Write one message. Nothing is buffered between calls: n8n's Chat Memory
	 * Manager adds messages one at a time and then discards this instance, so a
	 * message held back waiting for its counterpart would be lost silently.
	 */
	async addMessage(message: Message): Promise<void> {
		await this.addMessages([message]);
	}

	/**
	 * Write a batch as Cognee question/answer entries.
	 *
	 * A user message immediately followed by an assistant message is the normal
	 * agent turn and becomes one paired entry. Every other message is written on
	 * its own with a placeholder on the missing side — never dropped — so a lone
	 * user message, a trailing question, or a system/tool message still reaches
	 * Cognee and shows up in Recall and Session → Get.
	 */
	async addMessages(messages: Message[]): Promise<void> {
		let index = 0;
		while (index < messages.length) {
			const current = messages[index];
			const next = messages[index + 1];

			if (current.role === 'user' && next?.role === 'assistant') {
				await this.storeTurn(textOf(current), textOf(next));
				index += 2;
				continue;
			}
			if (current.role === 'user') {
				await this.storeTurn(textOf(current), NO_ANSWER);
			} else if (current.role === 'assistant') {
				await this.storeTurn(NO_QUESTION, textOf(current));
			} else {
				// system / tool messages have no Q&A shape; keep them as context-only rows.
				await this.storeTurn(`[${current.role}]`, textOf(current));
			}
			index += 1;
		}
	}

	async clear(): Promise<void> {
		throw new CogneeRequestError(CLEAR_UNSUPPORTED_MESSAGE);
	}

	/** POST one Q&A entry into the session cache (and, via Cognee, the graph). */
	async storeTurn(question: string, answer: string): Promise<void> {
		const payload = buildRememberEntryPayload({
			entryType: 'qa',
			sessionId: this.sessionId,
			datasetName: this.datasetName,
			fields: { question: question || NO_QUESTION, answer: answer || NO_ANSWER },
		});
		await this.request({ method: 'POST', url: '/v1/remember/entry', body: payload });
	}
}

/**
 * ChatMemory handed to the AI Agent via supplyMemory(). Loads the last
 * `windowSize` turns from Cognee and writes each new turn as a Q&A entry.
 */
export class CogneeChatMemory extends BaseChatMemory {
	readonly chatHistory: CogneeChatHistory;
	private readonly windowSize: number;

	constructor(config: CogneeMemoryConfig) {
		super();
		this.chatHistory = new CogneeChatHistory(config.sessionId, config.datasetName, config.request);
		this.windowSize = Math.min(Math.max(1, Math.floor(config.windowSize || 10)), MAX_SESSION_PAIRS);
	}

	async loadMessages(): Promise<Message[]> {
		return windowMessages(await this.chatHistory.getMessages(), this.windowSize);
	}

	async saveTurn(input: string, output: string): Promise<void> {
		await this.chatHistory.storeTurn(input, output);
	}

	async clear(): Promise<void> {
		await this.chatHistory.clear();
	}
}
