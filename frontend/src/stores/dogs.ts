import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, type Dog } from '@/api'

export const useDogsStore = defineStore('dogs', () => {
  const dogs = ref<Dog[]>([])
  const loading = ref(false)
  const lastError = ref<string | null>(null)

  async function fetchAll() {
    loading.value = true
    try {
      dogs.value = await api.listDogs()
      lastError.value = null
    } catch (err) {
      lastError.value = err instanceof Error ? err.message : String(err)
      dogs.value = []
    } finally {
      loading.value = false
    }
  }

  return { dogs, loading, lastError, fetchAll }
})
