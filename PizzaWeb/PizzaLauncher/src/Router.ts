import { createRouter, createWebHistory } from "vue-router";
import TemplateEditor from "./views/TemplateEditor.vue";
import BannerEditor from "./views/BannerEditor.vue";

const basePath = "/dist/";
const router = createRouter({
    history: createWebHistory(basePath),
    routes: [
        {path: "/", component: TemplateEditor },
        {path: "/banners", component: BannerEditor },
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