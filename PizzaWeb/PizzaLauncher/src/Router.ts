import { createRouter, createWebHistory } from "vue-router";
import TemplateEditor from "./views/TemplateEditor.vue";

const basePath = "/dist/";
const router = createRouter({
    history: createWebHistory(basePath),
    routes: [
        {path: "/", component: TemplateEditor },
    ]
});

// router.beforeEach((to, from, next)=>{
//     const publicPages = ['/'];
//     const authRequired = !publicPages.includes(to.path);
//     const appContext = useAppContext();
//     const logged = appContext.loginName.value != "";
//     if( authRequired && !logged){
//         next('/');
//     } else {
//         next();
//     }
// });

export default router;