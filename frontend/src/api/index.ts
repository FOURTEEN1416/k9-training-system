import { request } from './http'

export interface HealthResponse {
  status: string
  version: string
  environment: string
}

export interface Dog {
  id: number
  name: string
  breed?: string | null
  birth_date?: string | null
  gender?: 'MALE' | 'FEMALE' | null
  chip_id?: string | null
  handler_id?: number | null
  training_stage: 'P0' | 'P1' | 'P2'
  notes?: string | null
  created_at: string
  updated_at: string
}

export interface Video {
  id: number
  session_id?: number | null
  dog_id?: number | null
  handler_id?: number | null
  original_filename: string
  storage_path: string
  duration_sec?: number | null
  fps?: number | null
  width?: number | null
  height?: number | null
  size_bytes?: number | null
  status: 'UPLOADED' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  error_message?: string | null
  uploaded_at: string
  processed_at?: string | null
}

export interface Score {
  id: number
  video_id: number
  standard: 'GA_T' | 'USPCA' | 'FCI_IGP' | 'CUSTOM'
  accuracy: number
  response_latency?: number | null
  duration: number
  search_efficiency?: number | null
  attention: number
  courage?: number | null
  gait_quality?: number | null
  overall: number
  scoring_engine_version: string
  notes?: string | null
}

export interface MlModel {
  id: number
  name: string
  version: string
  type: 'POSE' | 'BEHAVIOR' | 'SCORING'
  framework: 'PYTORCH' | 'ONNX' | 'TENSORRT'
  storage_path: string
  is_active: boolean
  metrics_json?: Record<string, unknown> | null
  description?: string | null
}

export const api = {
  health: () => request<HealthResponse>({ url: '/health' }),

  listDogs: () => request<Dog[]>({ url: '/api/dogs' }),
  getDog: (id: number) => request<Dog>({ url: `/api/dogs/${id}` }),
  createDog: (data: Partial<Dog>) =>
    request<Dog>({ url: '/api/dogs', method: 'POST', data }),

  listVideos: () => request<Video[]>({ url: '/api/videos' }),
  getVideo: (id: number) => request<Video>({ url: `/api/videos/${id}` }),

  listScores: (videoId?: number) =>
    request<Score[]>({
      url: '/api/scores',
      params: videoId ? { video_id: videoId } : undefined,
    }),

  listModels: () => request<MlModel[]>({ url: '/api/models' }),
}
