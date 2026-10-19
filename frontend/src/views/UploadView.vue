<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  NCard, NSpace, NText, NUpload, NButton, NIcon, NSelect,
  NTag, NProgress, NAlert, useMessage,
  type UploadFileInfo, type SelectOption,
} from 'naive-ui'
import { CloudUploadOutline, DocumentTextOutline, DownloadOutline } from '@vicons/ionicons5'
import { api, type Scene, type Dog, type VideoStatusRead } from '@/api'

const message = useMessage()
const router = useRouter()

// === 状态 ===
const fileList = ref<UploadFileInfo[]>([])
const scene = ref<Scene>('obedience_trial')
const dogs = ref<Dog[]>([])
const selectedDogId = ref<number | null>(null)
const uploading = ref(false)
const currentVideoId = ref<number | null>(null)
const currentStatus = ref<OpsVideoStatus | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

// P1-6 服务端新增字段（api/index.ts 不在本线白名单，本地交叉类型扩展）
type OpsVideoStatus = VideoStatusRead & {
  uploaded_at?: string | null
  processing_stage?: string | null
  processing_percent?: number | null
  processing_started_at?: string | null
  processing_duration_ms?: number | null
  processing_elapsed_ms?: number | null
  error_code?: string | null
  error_hint?: string | null
  processing_stale?: boolean
  retry_scheduled?: boolean
}

// 阶段中文映射 + 旧数据 percent=null 时的回退下限（与服务端阶段表一致）
const STAGE_LABELS: Record<string, string> = {
  preparing: '准备中',
  pose_inference: '姿态推理',
  keypoints: '关键点写入',
  object_detection: '物体检测',
  behavior_inference: '行为识别',
  scoring: '评分计算',
  report: '生成报告',
  persist: '结果写入',
  completed: '已完成',
  retrying: '等待重试',
}

const STAGE_FALLBACK_PERCENT: Record<string, number> = {
  preparing: 0,
  pose_inference: 5,
  keypoints: 90,
  object_detection: 92,
  behavior_inference: 94,
  scoring: 96,
  report: 98,
  persist: 99,
  completed: 100,
  retrying: 1,
}

const stageLabel = computed(() => {
  const stage = currentStatus.value?.processing_stage
  if (!stage) return '处理中'
  return STAGE_LABELS[stage] || stage
})

const clampedPercent = computed(() => {
  const value = currentStatus.value?.processing_percent
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.min(100, Math.max(0, Math.round(value)))
  }
  const stage = currentStatus.value?.processing_stage
  return stage ? (STAGE_FALLBACK_PERCENT[stage] ?? 0) : 0
})

// 已运行时长（每秒刷新，卸载时清理）
const nowMs = ref(Date.now())
let clockTimer: ReturnType<typeof setInterval> | null = null

const elapsedText = computed(() => {
  const status = currentStatus.value
  if (status?.status !== 'processing') return ''
  if (typeof status.processing_elapsed_ms === 'number' && status.processing_elapsed_ms >= 0) {
    return formatDuration(status.processing_elapsed_ms)
  }
  if (!status.processing_started_at) return ''
  const started = new Date(status.processing_started_at).getTime()
  if (!Number.isFinite(started)) return ''
  return formatDuration(Math.max(0, nowMs.value - started))
})

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const min = Math.floor(totalSec / 60)
  const sec = totalSec % 60
  return min > 0 ? `${min} 分 ${sec} 秒` : `${sec} 秒`
}

const retryCount = ref(0)
watch(
  () => currentStatus.value?.retry_scheduled === true,
  (isRetrying, wasRetrying) => {
    if (isRetrying && !wasRetrying) retryCount.value += 1
  },
)

const retryText = computed(() =>
  retryCount.value > 0 ? `第 ${retryCount.value} 次重试中` : '等待重试',
)

// === 场景选项 ===
const sceneOptions: SelectOption[] = [
  { label: '科目测评（坐/卧/立/吠叫等 8 类行为）', value: 'obedience_trial' },
  { label: '幼犬选育（食物/玩具/胆量 3 维）', value: 'puppy_selection' },
]

// === 犬只选项 ===
const dogOptions = computed<SelectOption[]>(() =>
  dogs.value.map(d => ({
    label: `${d.name}${d.breed ? '（' + d.breed + '）' : ''}`,
    value: d.id,
  })),
)

// === 文件选择 ===
function handleChange(options: { fileList: UploadFileInfo[] }) {
  fileList.value = options.fileList.slice(-1) // 只保留最后一个
}

const selectedFile = computed(() => fileList.value[0]?.file ?? null)

// === 上传 ===
async function handleUpload() {
  if (!selectedFile.value) {
    message.warning('请先选择视频文件')
    return
  }

  uploading.value = true
  currentStatus.value = null
  retryCount.value = 0
  try {
    const video = await api.uploadVideo(
      selectedFile.value,
      scene.value,
      selectedDogId.value ?? undefined,
    )
    currentVideoId.value = video.id
    message.success(`上传成功，video_id=${video.id}，开始推理...`)
    startPolling(video.id)
  } catch (e) {
    message.error(`上传失败: ${(e as Error).message}`)
  } finally {
    uploading.value = false
  }
}

// === 状态轮询 ===
function startPolling(videoId: number) {
  stopPolling()
  pollTimer = setInterval(async () => {
    try {
      const status = await api.getVideoStatus(videoId)
      currentStatus.value = status as OpsVideoStatus
      if (status.status === 'completed') {
        stopPolling()
        message.success('推理完成，可查看报告')
      } else if (status.status === 'failed') {
        stopPolling()
        const detail = (status as OpsVideoStatus).error_hint
          || status.error_message
          || '未知错误'
        message.error(`推理失败: ${detail}`)
      }
      // processing（含 retrying）继续轮询，不在首次失败时停止
    } catch {
      // 轮询失败忽略，下次重试
    }
  }, 2000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// === 状态展示 ===
const statusTagType = computed(() => {
  const s = currentStatus.value?.status
  if (s === 'completed') return 'success'
  if (s === 'failed') return 'error'
  if (s === 'processing') return 'info'
  return 'default'
})

const statusText = computed(() => {
  const s = currentStatus.value?.status
  const map: Record<string, string> = {
    uploaded: '已上传',
    processing: '推理中',
    completed: '已完成',
    failed: '失败',
  }
  return s ? (map[s] || s) : ''
})

const isProcessing = computed(() => currentStatus.value?.status === 'processing')
const isCompleted = computed(() => currentStatus.value?.status === 'completed')

// === 跳转报告 ===
function goReport() {
  if (currentVideoId.value) {
    router.push(`/report/${currentVideoId.value}`)
  }
}

// === 下载报告 ===
function downloadReport() {
  if (currentVideoId.value) {
    window.open(api.getReportUrl(currentVideoId.value), '_blank')
  }
}

// === 清理 ===
function resetUpload() {
  fileList.value = []
  currentVideoId.value = null
  currentStatus.value = null
  retryCount.value = 0
  stopPolling()
}

onMounted(async () => {
  clockTimer = setInterval(() => {
    nowMs.value = Date.now()
  }, 1000)
  try {
    dogs.value = await api.listDogs()
  } catch {
    // 犬只列表加载失败不阻塞
  }
})

onUnmounted(() => {
  stopPolling()
  if (clockTimer) {
    clearInterval(clockTimer)
    clockTimer = null
  }
})
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="视频上传" size="medium" bordered>
      <NSpace vertical :size="16">
        <!-- 场景选择 -->
        <div>
          <NText strong>测试场景</NText>
          <NText depth="3" style="font-size: 12px; margin-left: 8px;">
            决定走哪条推理 + 评分管线
          </NText>
        </div>
        <NSelect
          v-model:value="scene"
          :options="sceneOptions"
          style="max-width: 500px;"
        />

        <!-- 犬只选择（可选） -->
        <div>
          <NText strong>关联犬只</NText>
          <NText depth="3" style="font-size: 12px; margin-left: 8px;">
            可选，用于报告归档
          </NText>
        </div>
        <NSelect
          v-model:value="selectedDogId"
          :options="dogOptions"
          clearable
          placeholder="不关联犬只"
          style="max-width: 500px;"
        />

        <!-- 文件上传 -->
        <div>
          <NText strong>选择视频文件</NText>
          <NText depth="3" style="font-size: 12px; margin-left: 8px;">
            支持 MP4 / AVI / MOV / MKV，最大 2GB
          </NText>
        </div>
        <NUpload
          v-model:file-list="fileList"
          :max="1"
          :default-upload="false"
          accept=".mp4,.avi,.mov,.mkv,.webm"
          @change="handleChange"
        >
          <NButton type="primary" dashed>
            <template #icon>
              <NIcon :component="CloudUploadOutline" />
            </template>
            点击或拖拽视频文件
          </NButton>
        </NUpload>

        <!-- 上传按钮 -->
        <NSpace>
          <NButton
            type="primary"
            :loading="uploading"
            :disabled="!selectedFile || isProcessing"
            @click="handleUpload"
          >
            <template #icon>
              <NIcon :component="CloudUploadOutline" />
            </template>
            {{ uploading ? '上传中...' : '开始上传 + 推理' }}
          </NButton>
          <NButton
            v-if="currentVideoId"
            quaternary
            @click="resetUpload"
          >
            重置
          </NButton>
        </NSpace>
      </NSpace>
    </NCard>

    <!-- 推理状态 -->
    <NCard
      v-if="currentStatus"
      title="推理状态"
      size="medium"
      bordered
    >
      <NSpace vertical :size="12">
        <NSpace align="center">
          <NText>视频 ID:</NText>
          <NText strong>{{ currentVideoId }}</NText>
          <NTag :type="statusTagType" size="small" round>
            {{ statusText }}
          </NTag>
        </NSpace>

        <!-- 推理进度（P1-6：阶段 + 百分比 + 已运行时长） -->
        <div v-if="isProcessing">
          <NSpace align="center" :size="8" style="margin-bottom: 6px;">
            <NText strong>{{ stageLabel }}</NText>
            <NTag v-if="currentStatus.retry_scheduled" type="warning" size="small" round>
              {{ retryText }}
            </NTag>
            <NText depth="3" style="font-size: 12px;">
              {{ clampedPercent }}%
            </NText>
            <NText v-if="elapsedText" depth="3" style="font-size: 12px;">
              已运行 {{ elapsedText }}
            </NText>
          </NSpace>
          <NProgress
            type="line"
            status="info"
            :processing="true"
            :percentage="clampedPercent"
          />
        </div>

        <!-- 疑似卡死警告 -->
        <NAlert
          v-if="currentStatus.processing_stale"
          type="warning"
          title="处理可能已卡死"
        >
          该任务已超过 2 小时未完成，可能已卡死，请联系管理员检查 Celery worker。
        </NAlert>

        <!-- 失败信息 -->
        <NAlert
          v-if="currentStatus.status === 'failed'"
          type="error"
          :title="`推理失败${currentStatus.error_code ? ' [' + currentStatus.error_code + ']' : ''}`"
        >
          <NSpace vertical :size="4">
            <NText>{{ currentStatus.error_message || '未知错误' }}</NText>
            <NText v-if="currentStatus.error_hint" depth="2" style="font-size: 12px;">
              建议：{{ currentStatus.error_hint }}
            </NText>
          </NSpace>
        </NAlert>

        <!-- 完成操作 -->
        <NSpace v-if="isCompleted">
          <NButton type="primary" @click="goReport">
            <template #icon>
              <NIcon :component="DocumentTextOutline" />
            </template>
            查看评分报告
          </NButton>
          <NButton @click="downloadReport">
            <template #icon>
              <NIcon :component="DownloadOutline" />
            </template>
            下载 PDF
          </NButton>
        </NSpace>
      </NSpace>
    </NCard>
  </NSpace>
</template>
