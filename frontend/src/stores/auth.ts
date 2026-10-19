import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { api, type HandlerInfo } from '@/api'

const TOKEN_KEY = 'k9_access_token'

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(sessionStorage.getItem(TOKEN_KEY))
  const user = ref<HandlerInfo | null>(null)

  const isLoggedIn = computed(() => !!token.value)
  const role = computed(() => user.value?.role ?? '')
  const userName = computed(() => user.value?.name ?? '')

  async function login(email: string, password: string) {
    const resp = await api.login(email, password)
    token.value = resp.access_token
    user.value = resp.handler
    sessionStorage.setItem(TOKEN_KEY, resp.access_token)
  }

  /** 刷新会话中的用户信息（页面刷新后 token 仍在但内存 user 丢失） */
  async function fetchMe() {
    if (!token.value) return
    user.value = await api.me()
  }

  function logout() {
    token.value = null
    user.value = null
    sessionStorage.removeItem(TOKEN_KEY)
  }

  return { token, user, isLoggedIn, role, userName, login, fetchMe, logout }
})
