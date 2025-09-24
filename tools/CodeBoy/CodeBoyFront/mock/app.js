// Mock data for app-related endpoints
export default [
  {
    url: '/api/app/health',
    method: 'get',
    response: () => {
      return {
        code: 200,
        message: 'OK',
        data: {
          status: 'healthy',
          timestamp: new Date().toISOString(),
        },
      };
    },
  },
  {
    url: '/api/app/info',
    method: 'get',
    response: () => {
      return {
        code: 200,
        message: 'OK',
        data: {
          name: 'CodeBoy Frontend',
          version: '1.0.0',
          description: 'A Vue3 SPA for CodeBoy',
        },
      };
    },
  },
];
