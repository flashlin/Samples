import { createRouter, createWebHistory, createWebHashHistory } from "vue-router";
import { useAppStore } from "@/stores/appStore";
import HomeView from "../views/HomeView.vue";
import LoginView from "@/views/Login.vue";
import { storeToRefs } from 'pinia';

const router = createRouter({
  //history: createWebHistory(import.meta.env.BASE_URL),
  history: createWebHashHistory(),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
      meta: {
        requiresAuth: true,
      },
    },
    {
      path: "/login",
      name: "login",
      component: LoginView,
    },
    {
      path: "/about",
      name: "about",
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import("../views/AboutView.vue"),
    },
  ],
});

router.beforeEach((to, from, next) => {
  if (!to.matched.some((record) => record.meta.requiresAuth)) {
    next();
    return;
  }
  const appStore = useAppStore();
  const isAuthenticated = appStore.IsAuthenticated;
  if (!isAuthenticated) {
    next({
      name: "login",
      query: { redirect: to.fullPath },
    });
    return;
  }
  next();
});

export default router;
