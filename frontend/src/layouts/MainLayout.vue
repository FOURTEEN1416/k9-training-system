<script setup lang="ts">
import { computed, h, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NLayout,
  NLayoutHeader,
  NLayoutSider,
  NLayoutContent,
  NMenu,
  NIcon,
  NText,
  NTag,
  NButton,
  NDropdown,
  NSpace,
  type MenuOption,
} from 'naive-ui'
import {
  CloudUploadOutline,
  DocumentTextOutline,
  TimeOutline,
  SettingsOutline,
  RefreshOutline,
  StatsChartOutline,
  PersonCircleOutline,
  LogOutOutline,
} from '@vicons/ionicons5'
import { useAppStore } from '@/stores/app'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const appStore = useAppStore()
const authStore = useAuthStore()
const collapsed = ref(false)

function renderIcon(icon: typeof CloudUploadOutline) {
  return () => h(NIcon, null, { default: () => h(icon) })
}

// P0-6：全量菜单 + 角色可见性过滤（meta.roles 缺省 = 全员可见）
const allMenuOptions: MenuOption[] = [
  { label: '视频上传', key: '/upload', icon: renderIcon(CloudUploadOutline) },
  { label: '评分报告', key: '/report', icon: renderIcon(DocumentTextOutline) },
  { label: '历史记录', key: '/history', icon: renderIcon(TimeOutline) },
  { label: '训练对比', key: '/compare', icon: renderIcon(StatsChartOutline), roles: ['ADMIN', 'MANAGER', 'RESEARCHER'] },
  { label: '系统管理', key: '/admin', icon: renderIcon(SettingsOutline), roles: ['ADMIN', 'MANAGER'] },
]

const menuOptions = computed<MenuOption[]>(() =>
  allMenuOptions.filter(
    (m) => !(m as { roles?: string[] }).roles || (m as { roles?: string[] }).roles!.includes(authStore.role),
  ),
)

const activeKey = computed(() => {
  const path = route.path
  if (path.startsWith('/report')) return '/report'
  if (path.startsWith('/upload')) return '/upload'
  if (path.startsWith('/history')) return '/history'
  if (path.startsWith('/compare')) return '/compare'
  if (path.startsWith('/admin')) return '/admin'
  return path
})

function handleMenuSelect(key: string) {
  router.push(key)
}

async function refreshHealth() {
  await appStore.fetchHealth()
}

// 用户菜单
const userOptions = [
  { label: '退出登录', key: 'logout', icon: renderIcon(LogOutOutline) },
]

function handleUserSelect(key: string) {
  if (key === 'logout') {
    authStore.logout()
    router.push('/login')
  }
}

// 页头：真实版本 + 识别模式（P0-6 去除「Phase 0 · 骨架」误导文案）
const headerMeta = computed(() => {
  const v = appStore.health
  if (!v) return ''
  return `v${v.version} · ${v.environment}${v.deploy_mode ? ` · 识别模式: ${v.deploy_mode}` : ''}`
})

onMounted(async () => {
  // 刷新后 token 仍在 → 恢复用户信息；401 由 http 拦截器统一处理
  if (authStore.isLoggedIn && !authStore.user) {
    try {
      await authStore.fetchMe()
    } catch {
      // 会话失效：拦截器已跳登录
    }
  }
  refreshHealth()
})
</script>

<template>
  <NLayout style="height: 100vh">
    <NLayoutHeader bordered style="height: 56px; padding: 0 24px; display: flex; align-items: center; gap: 16px;">
      <NText strong style="font-size: 16px;">工作犬训练机器视觉识别系统</NText>
      <NTag :type="appStore.backendOnline ? 'success' : 'error'" size="small" round>
        后端：{{ appStore.backendOnline ? '在线' : '离线' }}
      </NTag>
      <NButton size="tiny" quaternary circle @click="refreshHealth">
        <template #icon>
          <NIcon :component="RefreshOutline" />
        </template>
      </NButton>
      <div style="flex: 1"></div>
      <NText depth="3" style="font-size: 12px;">{{ headerMeta }}</NText>
      <NDropdown :options="userOptions" @select="handleUserSelect">
        <NSpace align="center" style="cursor: pointer;">
          <NIcon :component="PersonCircleOutline" size="20" />
          <NText>{{ authStore.userName || '未登录' }}</NText>
          <NTag v-if="authStore.role" size="tiny" type="info" round>{{ authStore.role }}</NTag>
        </NSpace>
      </NDropdown>
    </NLayoutHeader>
    <NLayout has-sider style="height: calc(100vh - 56px)">
      <NLayoutSider
        bordered
        :collapsed="collapsed"
        :collapsed-width="64"
        :width="220"
        show-trigger
        @collapse="collapsed = true"
        @expand="collapsed = false"
      >
        <NMenu
          v-model:value="activeKey"
          :options="menuOptions"
          :collapsed="collapsed"
          :collapsed-width="64"
          :collapsed-icon-size="22"
          @update:value="handleMenuSelect"
        />
      </NLayoutSider>
      <NLayoutContent content-style="padding: 24px;" :native-scrollbar="false">
        <RouterView />
      </NLayoutContent>
    </NLayout>
  </NLayout>
</template>
