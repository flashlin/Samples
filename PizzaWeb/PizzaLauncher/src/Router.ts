import { createRouter, createWebHistory } from "vue-router";
import TemplateShelves from "./views/TemplateShelves";

const basePath = "/dist/";
const router = createRouter({
    history: createWebHistory(basePath),
    routes: [
        {path: "/", component: TemplateShelves },
        {path: "/temp", component: TemplateShelves },
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