import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

// Define types locally since we no longer have user API
interface LoginParams {
  username: string;
  password: string;
}

interface UserProfile {
  id: number;
  username: string;
  email: string;
  role: string;
  avatar?: string;
  createdAt: string;
}

export const useUserStore = defineStore('user', () => {
  // State
  const token = ref<string>(localStorage.getItem('token') || '');
  const userInfo = ref<UserProfile | null>(null);
  const loading = ref(false);

  // Getters
  const isLoggedIn = computed(() => !!token.value);

  // Actions
  const login = async (params: LoginParams) => {
    loading.value = true;
    try {
      // Mock login for now - replace with actual API call when needed
      const mockResult = {
        token: 'mock-token-' + Date.now(),
        user: {
          id: 1,
          username: params.username,
          email: `${params.username}@example.com`,
          role: 'user'
        }
      };
      
      token.value = mockResult.token;
      userInfo.value = { ...mockResult.user, createdAt: new Date().toISOString() };
      localStorage.setItem('token', mockResult.token);
      return mockResult;
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    } finally {
      loading.value = false;
    }
  };

  const logout = () => {
    token.value = '';
    userInfo.value = null;
    localStorage.removeItem('token');
  };

  const fetchProfile = async () => {
    if (!token.value) return;
    
    loading.value = true;
    try {
      // Mock profile fetch - replace with actual API call when needed
      const mockProfile: UserProfile = {
        id: 1,
        username: 'admin',
        email: 'admin@example.com',
        role: 'admin',
        avatar: 'https://via.placeholder.com/100',
        createdAt: new Date().toISOString(),
      };
      
      userInfo.value = mockProfile;
      return mockProfile;
    } catch (error) {
      console.error('Failed to fetch profile:', error);
      throw error;
    } finally {
      loading.value = false;
    }
  };

  return {
    // State
    token,
    userInfo,
    loading,
    // Getters
    isLoggedIn,
    // Actions
    login,
    logout,
    fetchProfile,
  };
});
