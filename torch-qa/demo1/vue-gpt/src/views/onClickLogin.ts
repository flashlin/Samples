import authService from '@/libs/authService';
import { data } from './LoginView.vue';

export const onClickLogin = () => {
await authService.login({
username: data.loginName,
password: data.password
});
};
