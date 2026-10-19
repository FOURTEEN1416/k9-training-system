import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/LoginView.vue'),
    meta: { public: true, title: '登录' },
  },
  {
    path: '/',
    redirect: '/upload',
  },
  {
    path: '/upload',
    name: 'upload',
    component: () => import('@/views/UploadView.vue'),
    meta: { title: '视频上传', icon: 'cloud-upload' },
  },
  {
    path: '/report/:id?',
    name: 'report',
    component: () => import('@/views/ReportView.vue'),
    meta: { title: '评分报告', icon: 'document-text' },
  },
  {
    path: '/history',
    name: 'history',
    component: () => import('@/views/HistoryView.vue'),
    meta: { title: '历史记录', icon: 'time' },
  },
  {
    path: '/compare',
    name: 'compare',
    component: () => import('@/views/CompareView.vue'),
    meta: { title: '训练对比', icon: 'stats-chart', roles: ['ADMIN', 'MANAGER', 'RESEARCHER'] },
  },
  {
    path: '/admin',
    name: 'admin',
    component: () => import('@/views/AdminView.vue'),
    meta: { title: '系统管理', icon: 'settings', roles: ['ADMIN', 'MANAGER'] },
  },
  {
    // P1-2：标注审核（必须位于 catch-all 之前）
    // 注意：MainLayout 菜单未包含本入口（布局不在本线白名单），当前仅 URL 直达。
    // meta.roles 不被 router 守卫执行（守卫只查 token），服务端 403 才是安全边界。
    path: '/annotations/review',
    name: 'annotation-review',
    component: () => import('@/views/AnnotationReviewView.vue'),
    meta: {
      title: '标注审核',
      icon: 'checkmark-done',
      roles: ['ADMIN', 'MANAGER', 'RESEARCHER', 'HANDLER', 'VIEWER'],
    },
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/upload',
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// P0-6 登录守卫：非 public 路由必须有会话 token
router.beforeEach((to) => {
  if (to.meta.public) return true
  const token = sessionStorage.getItem('k9_access_token')
  if (!token) {
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  return true
})

router.afterEach((to) => {
  const title = (to.meta?.title as string) || ''
  document.title = title ? `${title} · K9 训练识别系统` : 'K9 训练识别系统'
})

export default router
