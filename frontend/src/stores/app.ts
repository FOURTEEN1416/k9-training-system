import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, type HealthResponse } from '@/api'

export const useAppStore = defineStore('app', () => {
  const health = ref<HealthResponse | null>(null)
  const lastError = ref<string | null>(null)
  const backendOnline = ref<boolean>(false)

  async function fetchHealth() {
    try {
      health.value = await api.health()
      backendOnline.value = health.value?.status === 'ok'
      lastError.value = null
    } catch (err) {
      lastError.value = err instanceof Error ? err.message : String(err)
      backendOnline.value = false
    }
    return backendOnline.value
  }

  return {
    health,
    lastError,
    backendOnline,
    fetchHealth,
  }
})
