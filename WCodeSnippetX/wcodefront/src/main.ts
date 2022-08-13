import {createApp} from 'vue';
import App from './App.vue';
import PrimeVue from 'primevue/config';

//import 'primevue/resources/themes/saga-blue/theme.css';
//import 'primevue/resources/themes/md-light-indigo/theme.css';
import 'primevue/resources/themes/bootstrap4-dark-blue/theme.css';
import 'primevue/resources/primevue.min.css';
import 'primeicons/primeicons.css';

import DialogService from 'primevue/dialogservice';

const app = createApp(App);
app.use(PrimeVue);
app.use(DialogService);
app.mount('#app');
