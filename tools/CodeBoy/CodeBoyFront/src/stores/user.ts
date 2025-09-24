import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { userApi, type LoginParams, type UserProfile } from '@/apis/user';

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
      const result = await userApi.login(params);
      token.value = result.token;
      userInfo.value = { ...result.user, createdAt: new Date().toISOString() };
      localStorage.setItem('token', result.token);
      return result;
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
      const profile = await userApi.getProfile();
      userInfo.value = profile;
      return profile;
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
