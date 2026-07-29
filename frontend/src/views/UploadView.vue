<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
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
const currentStatus = ref<VideoStatusRead | null>(null)
let pollTimer: ReturnType<typeof setInterval> | null = null

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
      currentStatus.value = status
      if (status.status === 'completed') {
        stopPolling()
        message.success('推理完成，可查看报告')
      } else if (status.status === 'failed') {
        stopPolling()
        message.error(`推理失败: ${status.error_message || '未知错误'}`)
      }
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
  stopPolling()
}

onMounted(async () => {
  try {
    dogs.value = await api.listDogs()
  } catch {
    // 犬只列表加载失败不阻塞
  }
})

onUnmounted(() => {
  stopPolling()
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

        <!-- 推理进度 -->
        <NProgress
          v-if="isProcessing"
          type="line"
          status="info"
          :indeterminate="true"
          :show-indicator="false"
        />

        <!-- 失败信息 -->
        <NAlert
          v-if="currentStatus.status === 'failed'"
          type="error"
          :title="`推理失败: ${currentStatus.error_message || '未知错误'}`"
        >
          请检查视频文件是否包含清晰的工作犬画面，或联系管理员查看日志。
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
