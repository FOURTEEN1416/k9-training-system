import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios'

const baseURL = '/'

const http: AxiosInstance = axios.create({
  baseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

const TOKEN_KEY = 'k9_access_token'

// 请求拦截：注入 JWT（P0-6）
http.interceptors.request.use((config) => {
  const token = sessionStorage.getItem(TOKEN_KEY)
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// 响应拦截：401 → 清理会话并跳登录（P0-6 错误态：未登录/会话过期可读）
http.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status
    if (status === 401 && !window.location.pathname.startsWith('/login')) {
      sessionStorage.removeItem(TOKEN_KEY)
      const redirect = encodeURIComponent(window.location.pathname + window.location.search)
      window.location.href = `/login?expired=1&redirect=${redirect}`
    }
    const message =
      error?.response?.data?.detail ||
      error?.message ||
      '请求失败'
    return Promise.reject(new Error(message))
  },
)

export async function request<T>(config: AxiosRequestConfig): Promise<T> {
  const response = await http.request<T>(config)
  return response.data
}

export default http
