import pkg from 'mockjs';
const { Random } = pkg;

// Mock data for user-related endpoints
export default [
  {
    url: '/api/user/login',
    method: 'post',
    response: ({ body }) => {
      const { username, password } = body;
      if (username === 'admin' && password === 'admin') {
        return {
          code: 200,
          message: 'Login successful',
          data: {
            token: Random.guid(),
            user: {
              id: 1,
              username: 'admin',
              email: 'admin@example.com',
              role: 'admin',
            },
          },
        };
      }
      return {
        code: 401,
        message: 'Invalid credentials',
        data: null,
      };
    },
  },
  {
    url: '/api/user/profile',
    method: 'get',
    response: () => {
      return {
        code: 200,
        message: 'OK',
        data: {
          id: 1,
          username: 'admin',
          email: 'admin@example.com',
          role: 'admin',
          avatar: Random.image('100x100', '#4A90E2', '#FFF', 'Avatar'),
          createdAt: Random.datetime(),
        },
      };
    },
  },
];
