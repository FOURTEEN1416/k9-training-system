<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { NConfigProvider, NMessageProvider, NDialogProvider, NLoadingBarProvider, zhCN, dateZhCN } from 'naive-ui'
import MainLayout from '@/layouts/MainLayout.vue'

const route = useRoute()
// P0-6：登录页不套主布局（无侧边导航）
const isLoginPage = computed(() => route.path === '/login')
</script>

<template>
  <NConfigProvider :locale="zhCN" :date-locale="dateZhCN">
    <NLoadingBarProvider>
      <NDialogProvider>
        <NMessageProvider>
          <MainLayout v-if="!isLoginPage" />
          <RouterView v-else />
        </NMessageProvider>
      </NDialogProvider>
    </NLoadingBarProvider>
  </NConfigProvider>
</template>
