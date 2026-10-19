<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NCard, NSpace, NForm, NFormItem, NInput, NButton, NText, NAlert, useMessage,
} from 'naive-ui'
import { useAuthStore } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const authStore = useAuthStore()

const email = ref('')
const password = ref('')
const loading = ref(false)

// 已登录直接进入目标页
onMounted(() => {
  if (authStore.isLoggedIn) {
    router.replace((route.query.redirect as string) || '/upload')
  }
  if (route.query.expired) {
    message.warning('会话已过期，请重新登录')
  }
})

async function handleLogin() {
  if (!email.value || !password.value) {
    message.warning('请输入邮箱和密码')
    return
  }
  loading.value = true
  try {
    await authStore.login(email.value.trim(), password.value)
    message.success(`欢迎，${authStore.userName}`)
    router.replace((route.query.redirect as string) || '/upload')
  } catch (e) {
    message.error(`登录失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <NCard title="工作犬训练机器视觉识别系统" style="width: 400px;" bordered>
      <NSpace vertical :size="16">
        <NText depth="3">请使用管理员分配的账号登录</NText>
        <NForm @submit.prevent="handleLogin">
          <NFormItem label="邮箱">
            <NInput
              v-model:value="email"
              type="text"
              placeholder="name@example.com"
              :input-props="{ autocomplete: 'username' }"
            />
          </NFormItem>
          <NFormItem label="密码">
            <NInput
              v-model:value="password"
              type="password"
              placeholder="请输入密码"
              show-password-on="click"
              :input-props="{ autocomplete: 'current-password' }"
              @keyup.enter="handleLogin"
            />
          </NFormItem>
          <NButton
            type="primary"
            attr-type="submit"
            :loading="loading"
            block
            @click="handleLogin"
          >
            登录
          </NButton>
        </NForm>
        <NAlert type="info" :bordered="false">
          识别结果仅供训练参考；未过验收精度时报告须人工复核。
        </NAlert>
      </NSpace>
    </NCard>
  </div>
</template>

<style scoped>
.login-page {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
}
</style>
