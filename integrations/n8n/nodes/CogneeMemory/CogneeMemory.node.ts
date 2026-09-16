import type {
	INodeType,
	INodeTypeDescription,
	ISupplyDataFunctions,
	JsonObject,
	SupplyData,
} from 'n8n-workflow';
import { NodeApiError, NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import { supplyMemory } from '@n8n/ai-node-sdk';

import { CogneeChatMemory, MAX_SESSION_PAIRS, toCogneeRequestError } from './memory';
import type { CogneeRequest, CogneeRequestOptions } from './memory';

type MemoryOptions = {
	datasetName?: string;
	windowSize?: number;
};

/**
 * AI Agent memory sub-node: persists the conversation in a Cognee session so
 * it is both the agent's chat history and searchable, cognified memory.
 */
export class CogneeMemory implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Cognee Memory',
		name: 'cogneeMemory',
		icon: { light: 'file:cognee.svg', dark: 'file:cognee.dark.svg' },
		group: ['transform'],
		version: [1],
		subtitle: 'Session memory in Cognee',
		description:
			'Store AI Agent conversation history in Cognee so it becomes searchable knowledge-graph memory',
		defaults: {
			name: 'Cognee Memory',
		},
		codex: {
			categories: ['assistant'],
			subcategories: {
				AI: ['Memory', 'Root Nodes'],
				Memory: ['Other memories'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://github.com/topoteretes/cognee-integrations/tree/main/integrations/n8n#sub-node-cognee-memory',
					},
				],
			},
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiMemory],
		outputNames: ['Memory'],
		credentials: [
			{
				name: 'cogneeApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				default: '={{ $json.sessionId }}',
				required: true,
				description:
					'Cognee session the conversation is stored under. Each Q&A turn becomes a session entry; the same value works with the Cognee node (Recall with Session ID, Session → Get).',
				placeholder: 'user-123',
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options for memory management',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Dataset Name',
						name: 'datasetName',
						type: 'string',
						default: 'main_dataset',
						description:
							'Cognee dataset the session is attributed to. Cognee cognifies the session into this dataset’s knowledge graph, so Recall and Search over it can find past conversations.',
					},
					{
						displayName: 'Window Size',
						name: 'windowSize',
						type: 'number',
						default: 10,
						description: 'Number of recent question/answer pairs to load into the agent context',
						typeOptions: {
							minValue: 1,
							maxValue: MAX_SESSION_PAIRS,
						},
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const sessionId = String(this.getNodeParameter('sessionId', itemIndex, '') ?? '').trim();
		if (!sessionId) {
			throw new NodeOperationError(this.getNode(), 'Session ID is required', { itemIndex });
		}
		const options = this.getNodeParameter('options', itemIndex, {}) as MemoryOptions;
		const datasetName = (options.datasetName ?? '').trim() || 'main_dataset';
		const windowSize = options.windowSize ?? 10;

		const credentials = await this.getCredentials('cogneeApi');
		const baseUrl = String(credentials.baseUrl ?? '').replace(/\/+$/, '');
		if (!baseUrl) {
			throw new NodeOperationError(this.getNode(), 'The Cognee API credential has no Base URL', {
				itemIndex,
			});
		}

		// The credential's authenticate block adds the X-Api-Key header.
		const request: CogneeRequest = async (requestOptions: CogneeRequestOptions) => {
			try {
				return await this.helpers.httpRequestWithAuthentication.call(this, 'cogneeApi', {
					method: requestOptions.method,
					url: `${baseUrl}/api${requestOptions.url}`,
					body: requestOptions.body,
					headers: { Accept: 'application/json' },
					json: true,
					timeout: 300_000,
				});
			} catch (error) {
				if (requestOptions.allowNotFound && toCogneeRequestError(error).statusCode === 404) {
					return undefined;
				}
				throw new NodeApiError(this.getNode(), error as JsonObject);
			}
		};

		const memory = new CogneeChatMemory({ sessionId, datasetName, windowSize, request });
		// @n8n/ai-utilities pins its own n8n-workflow copy, so the two ISupplyDataFunctions
		// declarations are structurally identical but nominally distinct to tsc.
		return supplyMemory(this as unknown as Parameters<typeof supplyMemory>[0], memory);
	}
}
