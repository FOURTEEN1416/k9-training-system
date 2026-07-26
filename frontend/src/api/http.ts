import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios'

const baseURL = '/'

const http: AxiosInstance = axios.create({
  baseURL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

http.interceptors.response.use(
  (response) => response,
  (error) => {
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
