import client from './client';

describe('Axios Client', () => {
    it('should be defined', () => {
        expect(client).toBeDefined();
    });

    it('should have the correct base URL', () => {
        const baseURL = client.defaults.baseURL;
        expect(baseURL).toBeDefined();
        const expected = process.env.REACT_APP_API_BASE_URL || 'https://api.spotme.life';
        expect(baseURL).toBe(expected);
    });

    it('should have correct default headers', () => {
        expect(client.defaults.headers['Content-Type']).toBe('application/json');
    });
});
