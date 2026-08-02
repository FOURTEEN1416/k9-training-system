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
  type MenuOption,
} from 'naive-ui'
import {
  CloudUploadOutline,
  DocumentTextOutline,
  TimeOutline,
  SettingsOutline,
  RefreshOutline,
  StatsChartOutline,
} from '@vicons/ionicons5'
import { useAppStore } from '@/stores/app'

const router = useRouter()
const route = useRoute()
const appStore = useAppStore()
const collapsed = ref(false)

function renderIcon(icon: typeof CloudUploadOutline) {
  return () => h(NIcon, null, { default: () => h(icon) })
}

const menuOptions: MenuOption[] = [
  { label: '视频上传', key: '/upload', icon: renderIcon(CloudUploadOutline) },
  { label: '评分报告', key: '/report', icon: renderIcon(DocumentTextOutline) },
  { label: '历史记录', key: '/history', icon: renderIcon(TimeOutline) },
  { label: '训练对比', key: '/compare', icon: renderIcon(StatsChartOutline) },
  { label: '系统管理', key: '/admin', icon: renderIcon(SettingsOutline) },
]

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

onMounted(() => {
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
      <NText depth="3" style="font-size: 12px;">Phase 0 · 骨架</NText>
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
