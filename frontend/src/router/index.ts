import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
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
    path: '/admin',
    name: 'admin',
    component: () => import('@/views/AdminView.vue'),
    meta: { title: '系统管理', icon: 'settings' },
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

router.afterEach((to) => {
  const title = (to.meta?.title as string) || ''
  document.title = title
    ? `${title} - 工作犬训练机器视觉识别系统`
    : '工作犬训练机器视觉识别系统'
})

export default router
