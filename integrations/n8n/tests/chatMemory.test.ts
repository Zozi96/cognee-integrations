import { describe, expect, it } from 'vitest';
import type { ISupplyDataFunctions } from 'n8n-workflow';

import {
	CLEAR_UNSUPPORTED_MESSAGE,
	CogneeChatHistory,
	CogneeChatMemory,
	CogneeRequestError,
	sessionEntriesToMessages,
	toCogneeRequestError,
	windowMessages,
} from '../nodes/CogneeMemory/memory';
import type { CogneeRequestOptions } from '../nodes/CogneeMemory/memory';
import { CogneeMemory } from '../nodes/CogneeMemory/CogneeMemory.node';

type Call = CogneeRequestOptions;

function fakeRequest(responses: Record<string, unknown | (() => unknown)> = {}) {
	const calls: Call[] = [];
	const request = async (options: Call) => {
		calls.push(options);
		const key = `${options.method} ${options.url}`;
		if (!(key in responses)) return {};
		const value = responses[key];
		return typeof value === 'function' ? (value as () => unknown)() : value;
	};
	return { calls, request };
}

const sessionBody = {
	session_id: 'chat-1',
	qas: [
		{ time: '2026-01-01T00:00:00Z', question: 'Hi', answer: 'Hello!', qa_id: 'q1' },
		{
			time: '2026-01-01T00:01:00Z',
			question: 'Where was Einstein born?',
			answer: 'Ulm.',
			qa_id: 'q2',
		},
		{ time: '2026-01-01T00:02:00Z', question: '', answer: '' },
	],
	traces: [],
};

describe('sessionEntriesToMessages / windowMessages', () => {
	it('maps Q&A entries to alternating user/assistant messages, oldest first, skipping empty rows', () => {
		expect(sessionEntriesToMessages(sessionBody)).toEqual([
			{ role: 'user', content: [{ type: 'text', text: 'Hi' }] },
			{ role: 'assistant', content: [{ type: 'text', text: 'Hello!' }] },
			{ role: 'user', content: [{ type: 'text', text: 'Where was Einstein born?' }] },
			{ role: 'assistant', content: [{ type: 'text', text: 'Ulm.' }] },
		]);
	});

	it('tolerates unexpected shapes', () => {
		expect(sessionEntriesToMessages(null)).toEqual([]);
		expect(sessionEntriesToMessages({ qas: 'nope' })).toEqual([]);
		expect(sessionEntriesToMessages({ qas: [null, 5] })).toEqual([]);
	});

	it('keeps only the last N pairs', () => {
		const messages = sessionEntriesToMessages(sessionBody);
		expect(windowMessages(messages, 1)).toEqual(messages.slice(-2));
		expect(windowMessages(messages, 10)).toEqual(messages);
	});
});

describe('toCogneeRequestError', () => {
	it('reads the status from n8n, axios and plain error shapes', () => {
		expect(toCogneeRequestError({ message: 'x', httpCode: '404' }).statusCode).toBe(404);
		expect(toCogneeRequestError({ message: 'x', response: { status: 500 } }).statusCode).toBe(500);
		expect(toCogneeRequestError({ message: 'x', cause: { statusCode: 401 } }).statusCode).toBe(401);
		expect(toCogneeRequestError(new Error('boom')).statusCode).toBeUndefined();
		expect(toCogneeRequestError(undefined).message).toBe('Cognee request failed');
	});
});

describe('CogneeChatMemory', () => {
	it('loads the windowed history from GET /v1/sessions/{id}', async () => {
		const { calls, request } = fakeRequest({ 'GET /v1/sessions/chat%201': sessionBody });
		const memory = new CogneeChatMemory({
			sessionId: 'chat 1',
			datasetName: 'main_dataset',
			windowSize: 1,
			request,
		});
		const messages = await memory.loadMessages();
		expect(calls).toEqual([{ method: 'GET', url: '/v1/sessions/chat%201', allowNotFound: true }]);
		expect(messages.map((m) => m.role)).toEqual(['user', 'assistant']);
		expect(messages[1].content).toEqual([{ type: 'text', text: 'Ulm.' }]);
	});

	it('treats a missing session (transport yields undefined on 404) as empty and propagates other errors', async () => {
		const notFound = fakeRequest({ 'GET /v1/sessions/new': () => undefined });
		const memory = new CogneeChatMemory({
			sessionId: 'new',
			datasetName: 'main_dataset',
			windowSize: 5,
			request: notFound.request,
		});
		expect(await memory.loadMessages()).toEqual([]);

		const failing = fakeRequest({
			'GET /v1/sessions/new': () => {
				throw new CogneeRequestError('unauthorized', 401);
			},
		});
		const broken = new CogneeChatMemory({
			sessionId: 'new',
			datasetName: 'main_dataset',
			windowSize: 5,
			request: failing.request,
		});
		await expect(broken.loadMessages()).rejects.toThrow(/unauthorized/);
	});

	it('saves each turn as one qa entry via POST /v1/remember/entry', async () => {
		const { calls, request } = fakeRequest();
		const memory = new CogneeChatMemory({
			sessionId: 'chat-1',
			datasetName: 'support',
			windowSize: 10,
			request,
		});
		await memory.saveTurn('What is Cognee?', 'A memory engine.');
		expect(calls).toEqual([
			{
				method: 'POST',
				url: '/v1/remember/entry',
				body: {
					entry: {
						type: 'qa',
						question: 'What is Cognee?',
						answer: 'A memory engine.',
						context: '',
					},
					session_id: 'chat-1',
					dataset_name: 'support',
				},
			},
		]);
	});

	it('clamps the window to what the server returns and refuses clear() with a clear message', async () => {
		const { request } = fakeRequest();
		const memory = new CogneeChatMemory({
			sessionId: 's',
			datasetName: 'd',
			windowSize: 999,
			request,
		});
		expect((memory as unknown as { windowSize: number }).windowSize).toBe(20);
		await expect(memory.clear()).rejects.toThrow(CLEAR_UNSUPPORTED_MESSAGE);
	});
});

describe('CogneeChatHistory (Chat Memory Manager path)', () => {
	it('pairs user and assistant messages into qa entries and pads unpaired ones', async () => {
		const { calls, request } = fakeRequest();
		const history = new CogneeChatHistory('chat-1', 'main_dataset', request);
		await history.addMessages([
			{ role: 'user', content: [{ type: 'text', text: 'Q1' }] },
			{ role: 'assistant', content: [{ type: 'text', text: 'A1' }] },
			{ role: 'user', content: [{ type: 'text', text: 'Q2 (never answered)' }] },
		]);
		const entries = calls.map(
			(c) => (c.body as { entry: { question: string; answer: string } }).entry,
		);
		expect(entries).toEqual([
			{ type: 'qa', question: 'Q1', answer: 'A1', context: '' },
			{
				type: 'qa',
				question: 'Q2 (never answered)',
				answer: '(no assistant message)',
				context: '',
			},
		]);
	});

	// Regression: the Chat Memory Manager node adds messages one at a time and
	// never calls addMessages, so anything buffered waiting for a counterpart
	// was silently dropped when the instance went away.
	it('writes a lone user message immediately instead of buffering it', async () => {
		const { calls, request } = fakeRequest();
		const history = new CogneeChatHistory('chat-1', 'main_dataset', request);
		await history.addMessage({ role: 'user', content: [{ type: 'text', text: 'remember this' }] });
		expect(calls).toHaveLength(1);
		expect((calls[0].body as { entry: unknown }).entry).toEqual({
			type: 'qa',
			question: 'remember this',
			answer: '(no assistant message)',
			context: '',
		});
	});

	it('loses nothing when messages arrive one at a time', async () => {
		const { calls, request } = fakeRequest();
		const history = new CogneeChatHistory('chat-1', 'main_dataset', request);
		for (const message of [
			{ role: 'user' as const, content: [{ type: 'text' as const, text: 'Q' }] },
			{ role: 'assistant' as const, content: [{ type: 'text' as const, text: 'A' }] },
			{ role: 'system' as const, content: [{ type: 'text' as const, text: 'be terse' }] },
		]) {
			await history.addMessage(message);
		}
		const entries = calls.map(
			(c) => (c.body as { entry: { question: string; answer: string } }).entry,
		);
		expect(entries).toEqual([
			{ type: 'qa', question: 'Q', answer: '(no assistant message)', context: '' },
			{ type: 'qa', question: '(no user message)', answer: 'A', context: '' },
			{ type: 'qa', question: '[system]', answer: 'be terse', context: '' },
		]);
	});

	it('rejects clear() naming the Chat Memory Manager operations it blocks', async () => {
		const { calls, request } = fakeRequest();
		const history = new CogneeChatHistory('chat-1', 'main_dataset', request);
		await expect(history.clear()).rejects.toThrow(/Delete Messages.*Override All Messages/s);
		expect(calls).toHaveLength(0);
	});
});

describe('CogneeMemory node', () => {
	it('declares itself as an AI memory sub-node', () => {
		const node = new CogneeMemory();
		expect(node.description.outputs).toEqual(['ai_memory']);
		expect(node.description.inputs).toEqual([]);
		expect(node.description.credentials).toEqual([{ name: 'cogneeApi', required: true }]);
		expect(node.description.codex?.subcategories?.AI).toContain('Memory');
	});

	it('supplies a memory that calls the Cognee API through the credential', async () => {
		const http: Array<Record<string, unknown>> = [];
		const params: Record<string, unknown> = {
			sessionId: 'chat-7',
			options: { datasetName: 'support', windowSize: 2 },
		};
		const ctx = {
			getNodeParameter: (name: string, _i: number, fallback?: unknown) =>
				name in params ? params[name] : fallback,
			getNode: () => ({
				name: 'Cognee Memory',
				type: 'cogneeMemory',
				typeVersion: 1,
				position: [0, 0],
				parameters: {},
			}),
			getCredentials: async () => ({ baseUrl: 'https://tenant.example.cognee.ai/', apiKey: 'k' }),
			addInputData: () => ({ index: 0 }),
			addOutputData: () => undefined,
			helpers: {
				httpRequestWithAuthentication: async (_type: string, options: Record<string, unknown>) => {
					http.push(options);
					if (options.method === 'GET') return sessionBody;
					return { entry_type: 'qa', entry_id: 'q9' };
				},
			},
		} as unknown as ISupplyDataFunctions;

		const supplied = await new CogneeMemory().supplyData.call(ctx, 0);
		const lcMemory = supplied.response as {
			loadMemoryVariables: (v: Record<string, unknown>) => Promise<{ chat_history: unknown[] }>;
			saveContext: (i: Record<string, unknown>, o: Record<string, unknown>) => Promise<void>;
		};

		const { chat_history } = await lcMemory.loadMemoryVariables({ input: 'x' });
		expect(chat_history).toHaveLength(4); // window of 2 pairs
		expect(http[0]).toMatchObject({
			method: 'GET',
			url: 'https://tenant.example.cognee.ai/api/v1/sessions/chat-7',
			json: true,
		});

		await lcMemory.saveContext({ input: 'New question' }, { output: 'New answer' });
		expect(http[1]).toMatchObject({
			method: 'POST',
			url: 'https://tenant.example.cognee.ai/api/v1/remember/entry',
			body: {
				entry: { type: 'qa', question: 'New question', answer: 'New answer', context: '' },
				session_id: 'chat-7',
				dataset_name: 'support',
			},
		});
	});

	it('turns a 404 on the session lookup into an empty history and wraps other failures as NodeApiError', async () => {
		const make = (fail: () => never) =>
			({
				getNodeParameter: (name: string, _i: number, fallback?: unknown) =>
					name === 'sessionId' ? 'fresh' : fallback,
				getNode: () => ({
					name: 'Cognee Memory',
					type: 'cogneeMemory',
					typeVersion: 1,
					position: [0, 0],
					parameters: {},
				}),
				getCredentials: async () => ({ baseUrl: 'https://c.example', apiKey: 'k' }),
				addInputData: () => ({ index: 0 }),
				addOutputData: () => undefined,
				helpers: { httpRequestWithAuthentication: async () => fail() },
			}) as unknown as ISupplyDataFunctions;

		const missing = await new CogneeMemory().supplyData.call(
			make(() => {
				throw Object.assign(new Error('not found'), { httpCode: '404' });
			}),
			0,
		);
		const asLc = (m: unknown) =>
			m as {
				loadMemoryVariables: (v: Record<string, unknown>) => Promise<{ chat_history: unknown[] }>;
			};
		expect((await asLc(missing.response).loadMemoryVariables({})).chat_history).toEqual([]);

		const broken = await new CogneeMemory().supplyData.call(
			make(() => {
				throw Object.assign(new Error('forbidden'), { httpCode: '403' });
			}),
			0,
		);
		await expect(asLc(broken.response).loadMemoryVariables({})).rejects.toThrow(/Forbidden/);
	});

	it('rejects an empty session id', async () => {
		const ctx = {
			getNodeParameter: (_n: string, _i: number, fallback?: unknown) => fallback,
			getNode: () => ({
				name: 'Cognee Memory',
				type: 'cogneeMemory',
				typeVersion: 1,
				position: [0, 0],
				parameters: {},
			}),
		} as unknown as ISupplyDataFunctions;
		await expect(new CogneeMemory().supplyData.call(ctx, 0)).rejects.toThrow(
			/Session ID is required/,
		);
	});
});
