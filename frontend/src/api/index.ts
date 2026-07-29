import { request } from './http'

// ============================================================
// 类型定义
// ============================================================

export interface HealthResponse {
  status: string
  version: string
  environment: string
}

export interface Dog {
  id: number
  name: string
  breed?: string | null
  gender?: string | null
  chip_id?: string | null
  handler_id?: number | null
  training_stage: string
  created_at: string
}

export type VideoStatus = 'uploaded' | 'processing' | 'completed' | 'failed'
export type Scene = 'puppy_selection' | 'obedience_trial'

export interface Video {
  id: number
  dog_id?: number | null
  handler_id?: number | null
  original_filename: string
  status: VideoStatus
  scene: Scene
  duration_sec?: number | null
  fps?: number | null
  width?: number | null
  height?: number | null
  uploaded_at: string
  processed_at?: string | null
  error_message?: string | null
  report_path?: string | null
}

export interface VideoStatusRead {
  id: number
  status: VideoStatus
  scene: Scene
  error_message?: string | null
  report_path?: string | null
  processed_at?: string | null
}

export interface MlModel {
  id: number
  name: string
  version: string
  type: string  // pose / behavior / scoring
  framework: string  // pytorch / onnx / tensorrt
  storage_path: string
  is_active: boolean
  metrics_json?: Record<string, unknown> | null
  description?: string | null
  created_at: string
}

export interface ScoringConfig {
  scene: Scene
  content: string
}

export interface ScoringResult {
  total_score: number
  verdict: string  // pass / borderline / fail
  passed: boolean
  dimension_labels: Record<string, string>
  dimension_scores: Record<string, number>
  explanation: string[]
  scene: Scene
  card_name: string
  card_version: string
}

export interface Score {
  id: number
  video_id: number
  standard: string
  accuracy: number
  response_latency?: number | null
  duration: number
  search_efficiency?: number | null
  attention: number
  courage?: number | null
  gait_quality?: number | null
  overall: number
  scoring_engine_version: string
  created_at: string
}

// ============================================================
// API 封装
// ============================================================

export const api = {
  // === 健康 ===
  health: () => request<HealthResponse>({ url: '/health' }),

  // === 犬只 ===
  listDogs: () => request<Dog[]>({ url: '/api/dogs' }),
  getDog: (id: number) => request<Dog>({ url: `/api/dogs/${id}` }),
  createDog: (data: Partial<Dog>) =>
    request<Dog>({ url: '/api/dogs', method: 'POST', data }),
  updateDog: (id: number, data: Partial<Dog>) =>
    request<Dog>({ url: `/api/dogs/${id}`, method: 'PUT', data }),
  deleteDog: (id: number) =>
    request<void>({ url: `/api/dogs/${id}`, method: 'DELETE' }),

  // === 视频 ===
  listVideos: () => request<Video[]>({ url: '/api/videos' }),
  getVideo: (id: number) => request<Video>({ url: `/api/videos/${id}` }),
  uploadVideo: (
    file: File,
    scene: Scene,
    dogId?: number,
    handlerId?: number,
  ) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('scene', scene)
    if (dogId) formData.append('dog_id', String(dogId))
    if (handlerId) formData.append('handler_id', String(handlerId))
    return request<Video>({
      url: '/api/videos/upload',
      method: 'POST',
      data: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
  },
  getVideoStatus: (id: number) =>
    request<VideoStatusRead>({ url: `/api/videos/${id}/status` }),
  getReportUrl: (id: number) => `/api/videos/${id}/report`,

  // === 模型 ===
  listModels: (type?: string) =>
    request<MlModel[]>({ url: '/api/models', params: type ? { type } : undefined }),
  getCurrentModel: (type: string) =>
    request<MlModel>({ url: '/api/models/current', params: { type } }),
  registerModel: (data: Partial<MlModel>) =>
    request<MlModel>({ url: '/api/models/register', method: 'POST', data }),
  activateModel: (id: number) =>
    request<MlModel>({ url: `/api/models/${id}/activate`, method: 'POST' }),

  // === 评分卡 ===
  listScoringConfigs: () =>
    request<ScoringConfig[]>({ url: '/api/scoring/configs' }),
  getScoringConfig: (scene: Scene) =>
    request<ScoringConfig>({ url: `/api/scoring/configs/${scene}` }),
  updateScoringConfig: (scene: Scene, content: string) =>
    request<ScoringConfig>({
      url: `/api/scoring/configs/${scene}`,
      method: 'PUT',
      data: { content },
    }),
  evaluateScore: (scene: Scene, signals: Record<string, number | string | boolean>) =>
    request<ScoringResult>({
      url: '/api/scoring/evaluate',
      method: 'POST',
      data: { scene, signals },
    }),

  // === 评分查询 ===
  listScores: (videoId?: number) =>
    request<Score[]>({
      url: '/api/scores',
      params: videoId ? { video_id: videoId } : undefined,
    }),
  getScore: (id: number) =>
    request<Score>({ url: `/api/scores/${id}` }),
}
