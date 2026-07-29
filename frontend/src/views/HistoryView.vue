<script setup lang="ts">
import { h, onMounted, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import {
  NCard, NSpace, NText, NDataTable, NEmpty, NTag, NButton, NIcon, NSelect,
  NSpin, useMessage,
  type DataTableColumns, type SelectOption,
} from 'naive-ui'
import { DocumentTextOutline, DownloadOutline, RefreshOutline } from '@vicons/ionicons5'
import { api, type Video, type Scene, type Dog } from '@/api'

const message = useMessage()
const router = useRouter()

// === 数据 ===
const videos = ref<Video[]>([])
const dogs = ref<Dog[]>([])
const loading = ref(false)

// === 筛选 ===
const filterScene = ref<Scene | null>(null)
const filterStatus = ref<string | null>(null)
const filterDogId = ref<number | null>(null)

const sceneOptions: SelectOption[] = [
  { label: '全部场景', value: '' },
  { label: '科目测评', value: 'obedience_trial' },
  { label: '幼犬选育', value: 'puppy_selection' },
]

const statusOptions: SelectOption[] = [
  { label: '全部状态', value: '' },
  { label: '已上传', value: 'uploaded' },
  { label: '推理中', value: 'processing' },
  { label: '已完成', value: 'completed' },
  { label: '失败', value: 'failed' },
]

const dogOptions = computed<SelectOption[]>(() => [
  { label: '全部犬只', value: '' },
  ...dogs.value.map(d => ({
    label: `${d.name}${d.breed ? '（' + d.breed + '）' : ''}`,
    value: d.id,
  })),
])

// === 状态映射 ===
const statusTagType: Record<string, 'default' | 'info' | 'success' | 'error' | 'warning'> = {
  uploaded: 'default',
  processing: 'info',
  completed: 'success',
  failed: 'error',
}

const statusText: Record<string, string> = {
  uploaded: '已上传',
  processing: '推理中',
  completed: '已完成',
  failed: '失败',
}

const sceneText: Record<string, string> = {
  obedience_trial: '科目测评',
  puppy_selection: '幼犬选育',
}

// === 格式化 ===
function formatTime(iso: string): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { hour12: false })
}

function formatDogName(dogId: number | null | undefined): string {
  if (dogId == null) return '—'
  const d = dogs.value.find(x => x.id === dogId)
  return d ? d.name : `#${dogId}`
}

// === 表格列 ===
const columns: DataTableColumns<Video> = [
  { title: 'ID', key: 'id', width: 70 },
  { title: '场景', key: 'scene', width: 110, render: row => sceneText[row.scene] || row.scene },
  { title: '文件名', key: 'original_filename', ellipsis: { tooltip: true } },
  {
    title: '犬只',
    key: 'dog_id',
    width: 110,
    render: row => formatDogName(row.dog_id),
  },
  {
    title: '状态',
    key: 'status',
    width: 100,
    render: row =>
      h(NTag, { type: statusTagType[row.status] || 'default', size: 'small', round: true },
        () => statusText[row.status] || row.status),
  },
  { title: '上传时间', key: 'uploaded_at', width: 170, render: row => formatTime(row.uploaded_at) },
  {
    title: '操作',
    key: 'actions',
    width: 200,
    render: row =>
      h(NSpace, { size: 4 }, () => [
        h(NButton, {
          size: 'tiny',
          quaternary: true,
          type: 'primary',
          onClick: () => router.push(`/report/${row.id}`),
        }, {
          default: () => '查看',
          icon: () => h(NIcon, null, () => h(DocumentTextOutline)),
        }),
        row.status === 'completed' && row.report_path
          ? h(NButton, {
              size: 'tiny',
              quaternary: true,
              onClick: () => window.open(api.getReportUrl(row.id), '_blank'),
            }, {
              default: () => 'PDF',
              icon: () => h(NIcon, null, () => h(DownloadOutline)),
            })
          : null,
      ]),
  },
]

// === 加载 ===
async function loadVideos() {
  loading.value = true
  try {
    videos.value = await api.listVideos()
  } catch (e) {
    message.error(`加载视频列表失败: ${(e as Error).message}`)
  } finally {
    loading.value = false
  }
}

async function loadDogs() {
  try {
    dogs.value = await api.listDogs()
  } catch {
    // 不阻塞
  }
}

// === 筛选后的数据 ===
const filteredVideos = computed(() => {
  return videos.value.filter(v => {
    if (filterScene.value && v.scene !== filterScene.value) return false
    if (filterStatus.value && v.status !== filterStatus.value) return false
    if (filterDogId.value && v.dog_id !== filterDogId.value) return false
    return true
  })
})

// === 统计 ===
const stats = computed(() => {
  const total = videos.value.length
  const completed = videos.value.filter(v => v.status === 'completed').length
  const failed = videos.value.filter(v => v.status === 'failed').length
  const processing = videos.value.filter(v => v.status === 'processing').length
  return { total, completed, failed, processing }
})

onMounted(() => {
  loadDogs()
  loadVideos()
})
</script>

<template>
  <NSpace vertical :size="16">
    <!-- 统计卡片 -->
    <NCard size="small" bordered>
      <NSpace :size="32">
        <NSpace align="center" :size="8">
          <NText depth="3" style="font-size: 12px;">总数</NText>
          <NText strong style="font-size: 20px;">{{ stats.total }}</NText>
        </NSpace>
        <NSpace align="center" :size="8">
          <NText depth="3" style="font-size: 12px;">完成</NText>
          <NText strong style="font-size: 20px; color: #18a058;">{{ stats.completed }}</NText>
        </NSpace>
        <NSpace align="center" :size="8">
          <NText depth="3" style="font-size: 12px;">推理中</NText>
          <NText strong style="font-size: 20px; color: #2080f0;">{{ stats.processing }}</NText>
        </NSpace>
        <NSpace align="center" :size="8">
          <NText depth="3" style="font-size: 12px;">失败</NText>
          <NText strong style="font-size: 20px; color: #d03050;">{{ stats.failed }}</NText>
        </NSpace>
        <div style="flex: 1"></div>
        <NButton size="small" quaternary @click="loadVideos">
          <template #icon>
            <NIcon :component="RefreshOutline" />
          </template>
          刷新
        </NButton>
      </NSpace>
    </NCard>

    <!-- 视频列表 -->
    <NCard title="历史记录" size="medium" bordered>
      <!-- 筛选 -->
      <NSpace :size="12" style="margin-bottom: 16px;">
        <NSelect
          v-model:value="filterScene"
          :options="sceneOptions"
          placeholder="场景筛选"
          style="width: 160px;"
          clearable
        />
        <NSelect
          v-model:value="filterStatus"
          :options="statusOptions"
          placeholder="状态筛选"
          style="width: 140px;"
          clearable
        />
        <NSelect
          v-model:value="filterDogId"
          :options="dogOptions"
          placeholder="犬只筛选"
          style="width: 200px;"
          clearable
        />
        <NText depth="3" style="font-size: 12px; line-height: 28px;">
          共 {{ filteredVideos.length }} 条
        </NText>
      </NSpace>

      <NSpin :show="loading">
        <NDataTable
          :columns="columns"
          :data="filteredVideos"
          :bordered="false"
          :pagination="{ pageSize: 20 }"
          :row-key="(row: Video) => row.id"
        >
          <template #empty>
            <NEmpty description="暂无视频记录，请先上传">
              <template #extra>
                <NButton type="primary" @click="router.push('/upload')">
                  去上传
                </NButton>
              </template>
            </NEmpty>
          </template>
        </NDataTable>
      </NSpin>
    </NCard>
  </NSpace>
</template>
