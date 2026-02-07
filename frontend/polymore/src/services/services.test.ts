import client from '../api/client';
import { predictTier1, predictTier2, getTaskStatus } from '../services/services';

jest.mock('../api/client', () => ({
    post: jest.fn(),
    get: jest.fn(),
}));

describe('API Services', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });

    describe('predictTier1', () => {
        it('should call client.post with correct arguments and return data', async () => {
            const mockResponse = {
                data: {
                    status: 200,
                    message: 'Success',
                    data: { strength: 1.0, flexibility: 2.0, degradability: 3.0, sustainability: 4.0, sas_score: 5.0, meta: {} },
                },
            };
            (client.post as jest.Mock).mockResolvedValue(mockResponse);

            const result = await predictTier1('C');
            expect(client.post).toHaveBeenCalledWith('/predict/tier-1', { smiles: 'C' });
            expect(result).toEqual(mockResponse.data);
        });

        it('should handle errors correctly', async () => {
            const errorResponse = {
                response: {
                    data: {
                        status: 400,
                        message: 'Bad Request',
                        error: 'Invalid SMILES'
                    }
                }
            };
            (client.post as jest.Mock).mockRejectedValue(errorResponse);
            await expect(predictTier1('invalid')).rejects.toThrow('Invalid SMILES');
        });
    });

    describe('predictTier2', () => {
        it('should call client.post with correct arguments', async () => {
            const mockResponse = {
                data: {
                    status: 202,
                    message: 'Submitted',
                    data: { task_id: '123', status: 'submitted', message: 'Analysis submitted successfully' }
                }
            };
            (client.post as jest.Mock).mockResolvedValue(mockResponse);

            const result = await predictTier2('C');
            expect(client.post).toHaveBeenCalledWith('/predict/tier-2', { smiles: 'C' });
            expect(result).toEqual(mockResponse.data);
        });
    });

    describe('getTaskStatus', () => {
        it('should call client.get with correct arguments', async () => {
            const mockResponse = {
                data: {
                    status: 200,
                    message: 'Status check',
                    data: { task_id: '123', status: 'PENDING' }
                }
            };
            (client.get as jest.Mock).mockResolvedValue(mockResponse);

            const result = await getTaskStatus('123');
            expect(client.get).toHaveBeenCalledWith('/tasks/123');
            expect(result).toEqual(mockResponse.data);
        });
    });
});
