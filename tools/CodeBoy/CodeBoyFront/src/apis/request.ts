import axios, { type AxiosInstance, type InternalAxiosRequestConfig, type AxiosResponse } from 'axios';

// Create axios instance
const request: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_APP_API_BASE_URL || '/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
request.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
request.interceptors.response.use(
  (response: AxiosResponse) => {
    const { data } = response;
    
    // Handle common response structure
    if (data.code === 200) {
      return data.data;
    } else {
      console.error('API Error:', data.message);
      return Promise.reject(new Error(data.message || 'Unknown error'));
    }
  },
  (error) => {
    console.error('Request Error:', error.message);
    
    // Handle common error cases
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

export default request;
