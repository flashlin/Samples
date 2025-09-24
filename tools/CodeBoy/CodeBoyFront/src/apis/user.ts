import request from './request';

export interface LoginParams {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user: {
    id: number;
    username: string;
    email: string;
    role: string;
  };
}

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  role: string;
  avatar?: string;
  createdAt: string;
}

// User API
export const userApi = {
  // Login
  login(params: LoginParams): Promise<LoginResponse> {
    return request.post('/user/login', params);
  },

  // Get user profile
  getProfile(): Promise<UserProfile> {
    return request.get('/user/profile');
  },
};
